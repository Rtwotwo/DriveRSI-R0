"""Combine completed, admitted families without changing their executed lineage."""
import argparse
import copy
import fcntl
from pathlib import Path
import subprocess
import time

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rsi.decisions.capability_skills import identity, validate_plan, validate_sample_lineage
from driversi.rsi.decisions.trajectory_diversity import compare, describe
from driversi.rsi.experiments.continuation import frozen_json
from driversi.rsi.experiments.reactive_cohort import family_training_gate
from driversi.runtime.budget import write_json


def distinct_rows(candidates, minimum_ade_m, minimum_path_ade_m=None):
    """First admitted occurrence wins; deduplicate across generators at each source."""
    retained, selected, excluded = {}, [], []
    packages = set()
    for row in candidates:
        package = row['dataset_reference']['package_id']
        key = (row['assignment']['source_log'], row['assignment']['token'])
        decision = compare(row['descriptor'], retained.get(key, []), minimum_ade_m=minimum_ade_m,minimum_path_ade_m=minimum_path_ade_m)
        if package in packages or not decision['distinct']:
            excluded.append(dict(candidate=row['candidate'], package_id=package,
                                 reason='package_repeated' if package in packages else 'cross_strategy_duplicate',
                                 diversity=decision))
            continue
        packages.add(package)
        retained.setdefault(key, []).append(row['descriptor'])
        selected.append(dict(row, merge_diversity=decision))
    return selected, excluded


def require_profiles(sample, config):
    proof = read_mapping(sample['row']['admission']['path'])
    branch = read_mapping(proof['branch']['report'])
    visual = read_mapping(proof['visual_admission']['path'])
    if branch['trajectory_evidence']['admission_profile'] != config['trajectory_admission_profile']:
        raise ValueError('Executed trajectory profile differs from merged campaign')
    if visual['visual_quality_profile'] != config['visual_quality_profile']:
        raise ValueError('Executed visual profile differs from merged campaign')


def bounded_selection(selected, anchors, target, max_samples):
    if max_samples < target * len(anchors):
        raise ValueError('Merged sample budget cannot serve every declared source')
    counts, kept, excluded = {}, [], []
    for row in selected:
        key = (row['assignment']['source_log'], row['assignment']['token'])
        if counts.get(key, 0) >= target or len(kept) >= max_samples:
            excluded.append(dict(candidate=row['candidate'],
                                 package_id=row['dataset_reference']['package_id'],
                                 reason='qualified_but_not_selected_budget', trainable=True))
        else:
            kept.append(row)
            counts[key] = counts.get(key, 0) + 1
    return kept, excluded


def merge(config, families):
    """Reuse sealed outcomes, current admission and original generator plans."""
    from driversi_agents.chainflow.synthetic import validate_admission

    c = read_mapping(config)
    root = Path(c['output']).resolve()
    target = c['trajectory_family']['target_per_anchor']
    minimum = c['trajectory_family']['minimum_ade_m']
    if target < 10 or minimum < .5 or not families:
        raise ValueError('At least ten admitted branches and unchanged .5m diversity required')
    if len({str(Path(p).resolve()) for p in families}) != len(families):
        raise ValueError('Repeated source family')
    contexts, candidates, receipts = [], [], {}
    anchors = {}
    for fi, folder in enumerate(families):
        folder = Path(folder).resolve()
        summary_path = folder/'trajectory-family-summary.json'
        summary = read_mapping(summary_path)
        if summary['status'] not in ('target_met', 'completed_with_shortages'):
            raise ValueError('Every producer must settle before merging')
        plan = validate_plan(read_mapping(summary['plan']))
        if digest(summary['plan']) != summary['plan_sha256']:
            raise ValueError('Executed family plan changed')
        contexts.append(plan['context'])
        receipts[str(summary_path)] = digest(summary_path)
        assigned = {a['id']: a for a in plan['assignments']}
        for a in plan['assignments']:
            anchors.setdefault((a['source_log'], a['token']), f'anchor-{len(anchors)}')
        for row in summary['rows']:
            if not row.get('trainable'):
                continue
            if row['status'] != 'training_sample_ready' or not row.get('rendered'):
                raise ValueError('Only rendered, admitted samples may enter merged training')
            dest = folder/'candidates'/row['candidate']
            terminal = dest/'outcome.json'
            if digest(terminal) != read_mapping(dest/'outcome-seal.json')['sha256']:
                raise ValueError('Completed source outcome changed')
            saved = read_mapping(terminal)
            if {k: v for k, v in row.items() if k != 'rendered'} != saved:
                raise ValueError('Summary differs from actual completed source outcome')
            sample = read_mapping(row['sample_report'])
            validate_admission(sample['row'], sample['files'])
            require_profiles(sample, c)
            validate_sample_lineage(plan, [sample['row']], sample['files'])
            generation_path = Path(row['generation'])
            generation = read_mapping(generation_path)
            if generation['status'] != 'rendered':
                raise ValueError('Rendered execution receipt required')
            for path, sha in generation['files'].items():
                if digest(path) != sha:
                    raise ValueError('Rendered execution evidence changed')
            aid = row['candidate']
            candidate = f'family-{fi}-{aid}'
            candidates.append(dict(row, candidate=candidate,
                descriptor=describe(generation, candidate), assignment=assigned[aid],
                reused_execution=dict(path=str(generation_path), sha256=digest(generation_path)),
                original_family=str(folder), original_candidate=aid))
    if any(ctx != contexts[0] for ctx in contexts) or contexts[0]['model'] != c['student_model']:
        raise ValueError('Family parent/model/evaluator context differs')
    from driversi.rendering.diagnostics.numeric_admission import numeric_settings
    spatial=numeric_settings(c['visual_quality_profile']).get('minimum_path_ade_m')
    selected, excluded = distinct_rows(candidates, minimum, spatial)
    selected, budget_excluded = bounded_selection(selected, anchors, target, c['max_samples'])
    excluded.extend(budget_excluded)
    assignments, reused, rows = [], {}, []
    for row in selected:
        a = copy.deepcopy(row['assignment'])
        a.update(id=row['candidate'], anchor_assignment_id=anchors[(a['source_log'], a['token'])])
        assignments.append(a)
        reused[a['id']] = row['reused_execution']
        rows.append({**{k:v for k,v in row.items() if k not in ('assignment','reused_execution')},
                     'anchor_assignment_id': a['anchor_assignment_id']})
    plan = dict(kind='capability_skill_plan', schema_version=1, context=contexts[0],
                assignments=assignments, reused_executions=reused, automatic_training=False,
                aggregation='verified_completed_families', original_summaries=receipts)
    plan['plan_id'] = identity(plan)
    for row in rows:
        sample = read_mapping(row['sample_report'])
        validate_sample_lineage(plan, [sample['row']], sample['files'])
    summary = dict(status='completed', rows=rows, qualified=len(rows), excluded=excluded,
                   sample_reports=[r['sample_report'] for r in rows],
                   original_summaries=receipts, target_per_anchor=target,
                   selection='first_admitted_cross_strategy_ADE_deduplication',
                   generator_identity_preserved=True, promoted=False)
    gate = family_training_gate(summary, list(anchors.values()), target)
    root.mkdir(parents=True, exist_ok=True)
    frozen_json(root/'trajectory-family-plan.json', plan)
    frozen_json(root/'trajectory-family-summary.json', summary)
    frozen_json(root/'training-gate.json', gate)
    return summary, gate


def wait_for_families(families, seconds):
    end = time.monotonic()+seconds
    while True:
        if all((Path(p)/'trajectory-family-summary.json').exists() and
               read_mapping(Path(p)/'trajectory-family-summary.json')['status'] in
               ('target_met','completed_with_shortages') for p in families):
            return
        if time.monotonic() >= end:
            raise TimeoutError('Family completion wait exhausted; no training launched')
        time.sleep(min(15, max(0, end-time.monotonic())))


def training_preflight(config, shared_gpu):
    c = read_mapping(config)
    model = read_mapping(c['training_model_config'])
    settings = model['settings']
    fraction = settings.get('gpu_memory_fraction', .12)
    if not 0 < fraction <= .25:
        raise ValueError('Training and evaluation require compatible <=.25 GPU memory fraction')
    output = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total',
                                     '--format=csv,noheader,nounits'], text=True)
    memory = {int(x.split(',')[0]): tuple(map(int, x.split(',')[1:])) for x in output.splitlines()}
    for gpu in settings['distributed_training']['devices']:
        used, total = memory[gpu]
        if gpu == shared_gpu:
            if total-used < fraction*total+4096:
                raise ValueError('Authorized shared GPU has insufficient reserved headroom')
        elif used >= 500:
            raise ValueError(f'Training GPU {gpu} occupied; admitted family retained')
    return dict(shared_gpu=shared_gpu, pytorch_memory_fraction=fraction, observed_memory=memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--family', action='append', required=True)
    parser.add_argument('--wait-seconds', type=int, default=0)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--shared-gpu', type=int)
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 10800:
        raise ValueError('Bounded one-shot wait required')
    root = Path(read_mapping(args.config)['output'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/'merge.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wait_for_families(args.family, args.wait_seconds)
        summary, gate = merge(args.config, args.family)
        if args.train and gate['passed']:
            authorization = training_preflight(args.config, args.shared_gpu)
            frozen_json(root/'training-resource-preflight.json', authorization)
            from driversi.rsi.experiments.chainflow_campaign import train_pair
            done = train_pair(args.config, summary['sample_reports'], root/'chainflow-pair')
            write_json(root/'merge-completion.json', dict(gate=gate, paired_task_completed=done))


if __name__ == '__main__':
    main()
