"""Node-scoped model-service experience; never a driving-policy success label."""
import json
import math
import sqlite3
from pathlib import Path

from driversi.data.catalog.store import identity

NODES = (
    'source_selection', 'physical_generation', 'trajectory_admission',
    'visual_admission', 'joint_supervision', 'paired_training',
    'resume_verification', 'independent_evaluation', 'learning_review',
)
REPAIRS = {
    'source_selection': 'select_other_official_train_source',
    'physical_generation': 'audit_world_and_route_before_new_rollout',
    'trajectory_admission': 'revise_arrival_or_teacher_recipe_on_new_revision',
    'visual_admission': 'verify_asset_coverage_and_registration_before_rollout',
    'joint_supervision': 'revalidate_versioned_same_branch_images_and_language',
    'paired_training': 'repair_native_training_execution',
    'resume_verification': 'repair_recovery_before_measuring_gain',
    'independent_evaluation': 'complete_locked_official_evaluation',
    'learning_review': 'revise_data_intervention_then_repeat_matched_trial',
}


def review_learning(feedback, resume_checks):
    """Three decisions: valid experiment, observed gain, promotion. Never conflate."""
    valid = (feedback.get('kind') == 'augustus_paired_feedback'
             and feedback.get('status') == 'completed'
             and feedback.get('effective_recipe_verified') is True
             and feedback.get('positive_lr_added_sample_ids')
             and set(feedback['positive_lr_added_sample_ids']) == set(feedback['exposed_added_sample_ids']))
    if not valid:
        raise ValueError('Verified native paired feedback required')
    if len(resume_checks) != 2 or not all(all(x.get(k) is True for k in
            ('different_pids', 'same_compared_state')) for x in resume_checks):
        return dict(node='resume_verification', status='rejected',
                    reason='cross_process_recovery_mismatch', observed_gain=None,
                    skill_gain_credited=False, promoted=False)
    gain = feedback['gain_vs_control_pdms_points']
    parent_gain = feedback['gain_vs_parent_pdms_points']
    if any(type(x) not in (float, int) or not math.isfinite(x) for x in (gain, parent_gain)):
        raise ValueError('Finite measured gains required')
    safety = not feedback['safety_regressions']
    improved = safety and gain > 0 and parent_gain > 0
    return dict(node='learning_review', status='pilot_signal' if improved else 'no_benefit',
                reason='expand_independent_validation' if improved else
                'safety_regression' if not safety else 'no_advantage_over_control_and_parent',
                observed_gain=gain, gain_vs_parent=parent_gain, safety_passed=safety,
                skill_gain_credited=False, promoted=False,
                promotion_reason='locked_subset_single_trial_cannot_establish_stable_improvement')


class ServiceMemory:
    """Append to an existing skill DB; frozen mode cannot create or mutate it."""
    def __init__(self, path, mode):
        if mode not in ('evolve', 'frozen', 'off'):
            raise ValueError('Unknown service memory mode')
        self.path, self.mode, self.db = Path(path), mode, None
        if mode == 'evolve':
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path, timeout=30)
            self.db.execute('CREATE TABLE IF NOT EXISTS service_episodes (id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
            self.db.commit()
        elif mode == 'frozen' and self.path.exists():
            self.db = sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro', uri=True, timeout=30)

    def close(self):
        if self.db:
            self.db.close()

    def rows(self):
        if self.db is None or not self.db.execute("SELECT name FROM sqlite_master WHERE name='service_episodes'").fetchone():
            return []
        return [json.loads(x[0]) for x in self.db.execute('SELECT payload FROM service_episodes ORDER BY id')]

    def record(self, value):
        if self.mode != 'evolve':
            return dict(recorded=False, reason=self.mode+'_no_write')
        if value.get('promoted') is not False or value.get('skill_gain_credited') is not False:
            raise ValueError('Episode observation cannot promote or credit a new skill')
        if value['node'] not in NODES or not value.get('context') or not value.get('evidence'):
            raise ValueError('Node, model context and sealed evidence required')
        for p, sha in value['evidence'].items():
            from driversi.agents.backends.native import digest
            if digest(p) != sha:
                raise ValueError('Service episode evidence changed')
        key = value['program_id']
        payload = json.dumps(value, sort_keys=True, allow_nan=False)
        with self.db:
            old = self.db.execute('SELECT payload FROM service_episodes WHERE id=?', (key,)).fetchone()
            if old and old[0] != payload:
                raise ValueError('Conflicting final review for immutable program')
            self.db.execute('INSERT OR IGNORE INTO service_episodes VALUES (?,?)', (key,payload))
        return dict(recorded=old is None, episode_id=key)

    def advise(self, context, sources, *, conditions=()):
        """Scope to exact model/recipe/protocol; cross-log suggestions need shared facts."""
        sources, conditions = set(sources), set(conditions)
        rows = self.rows()
        by_id = {r["program_id"]: r for r in rows}
        resolved = set()
        for row in rows:
            prior = by_id.get(row.get("recovered_from"))
            if (prior and row.get("node") == "learning_review"
                    and row.get("status") in ("no_benefit", "pilot_signal")
                    and row.get("training_executed") is True
                    and prior.get("node") == "independent_evaluation"
                    and prior.get("training_executed") is True
                    and row["context"] == prior["context"] == context):
                resolved.add(prior["program_id"])
        matched = []
        for row in rows:
            if row["program_id"] in resolved:
                continue
            if row['context'] != context:
                continue
            # A previously trained source can inform an exact-source repair but
            # is not cross-scene transfer evidence. Conditions are facts only.
            overlap = sorted(sources & set(row['source_logs']))
            shared = sorted(conditions & set(row.get('conditions', [])))
            if not overlap and not shared:
                continue
            matched.append(dict(episode_id=row['program_id'], node=row['node'],
                                status=row['status'], same_source=bool(overlap),
                                shared_conditions=shared,
                                suggested_action='expand_independent_validation'
                                if row['node']=='learning_review' and row['status']=='pilot_signal'
                                else REPAIRS[row['node']],
                                measured_gain=row.get('observed_gain'),
                                applicability='same_source' if overlap else 'condition_overlap_unvalidated'))
        return dict(matches=matched, resolved_episodes=sorted(resolved),
                    preflight_actions=sorted({r['suggested_action'] for r in matched}),
                    mandatory_gates_unchanged=True, cause_established=False,
                    skill_superiority_established=False)
