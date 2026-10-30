"""Deploy a controller revision once at an owned, settled phase boundary."""
import argparse
import math
import os
import time
from pathlib import Path

from driversi.agents.backends.native import project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from .continuation import frozen_json


def process_identity(pid, root):
    proc = Path('/proc') / str(pid)
    try:
        args = [a.decode() for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
        stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    if not args:
        return None
    if ('driversi.rsi.experiments.continual_curriculum' not in args or 'auto' not in args
            or '--output' not in args
            or Path(args[args.index('--output')+1]).resolve() != Path(root).resolve()):
        raise ValueError('Previous PID does not own this curriculum')
    return dict(pid=pid, birth=stat[19])


def settled(root):
    from . import continual_curriculum as cc
    from .processing_resume import settled_generation
    state = read_mapping(cc.state_path(root))
    if any(c['status'] in ('dispatching', 'training', 'unknown_execution',
                          'needs_execution_reconciliation', 'asset_recovery_unknown')
           for c in state['cases']):
        raise ValueError('Native work is unresolved; no automatic relaunch')
    case = next((c for c in state['cases'] if c['id'] == state['active']), None)
    if case and case['status'] in ('generation_stopped', 'data_ready'):
        wave = Path(case['current_round']) / 'waves/wave-0001'
        if any((p/'execution').exists() and not (p/'completion.json').exists()
               for p in (wave/'resident-batches').glob('*')):
            raise ValueError('Postprocessing is not settled')
        return settled_generation(wave, allow_prior_completed=True)
    return dict(status='no_generation_replay_required')


def run(root, prior_pid, output, *, max_problems, max_transitions, deadline):
    from . import continual_curriculum as cc
    from .continual_execution import run as automatic
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_relative_to(project_root()/'exp') or not output.is_relative_to(root/'automatic'):
        raise ValueError('Handoff must belong to this campaign')
    if not math.isfinite(deadline) or not 0 < deadline-time.time() <= 86400:
        raise ValueError('Finite existing wall deadline within one day required')
    if (type(max_problems) is not int or max_problems<1 or type(max_transitions) is not int
            or not 1<=max_transitions<=64*max_problems):
        raise ValueError('Finite queue and transition budgets required')
    marker = root/'STOP_AUTOMATIC'
    if marker.exists() or output.exists():
        raise ValueError('Existing stop/handoff must not be overwritten')
    identity = process_identity(prior_pid, root)
    if identity is None:
        raise ValueError('Expected active previous executor')
    output.mkdir(parents=True)
    text = f'controller_revision_handoff:{output}:{prior_pid}'
    frozen_json(output/'started.json', dict(pid=os.getpid(), prior=identity,
        deadline=deadline, max_problems=max_problems, max_transitions=max_transitions,
        started=time.time(), recurring_monitor=False, interrupts_native_work=False))
    with marker.open('x') as f:
        f.write(text)

    def status(value, **fields):
        write_json(output/'status.json', dict(status=value, pid=os.getpid(), updated=time.time(), **fields))

    try:
        status('waiting_settled_boundary')
        while True:
            observed = process_identity(prior_pid, root)
            if observed is None:
                break
            if observed != identity:
                raise ValueError('Previous PID identity changed')
            if time.time() >= deadline:
                status('original_wall_budget_completed')
                return
            time.sleep(10)
        with cc.locked(root):
            if not marker.exists() or marker.read_text() != text:
                raise ValueError('Stop marker ownership changed')
            frozen_json(output/'settlement.json', settled(root))
            remaining = math.floor(deadline-time.time())
            if remaining < 1:
                status('original_wall_budget_completed')
                return
            (output/'boundary-stop.txt').write_text(text)
            marker.unlink()
            status('running_existing_harness', remaining_wall_seconds=remaining)
            result = automatic(root, max_problems=max_problems, max_transitions=max_transitions,
                               max_wall_seconds=remaining)
            write_json(output/'result.json', result)
            status('harness_returned', automatic_status=result['automatic_execution']['status'])
    except BaseException as exc:
        status('needs_reconciliation', error_type=type(exc).__name__)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prior-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-problems', type=int, required=True)
    parser.add_argument('--max-transitions', type=int, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    run(args.root, args.prior_pid, args.output, max_problems=args.max_problems,
        max_transitions=args.max_transitions, deadline=args.deadline)


if __name__ == '__main__':
    main()
