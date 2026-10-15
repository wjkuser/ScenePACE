"""Task completion first; bounded open-room richness and one-shot appearance."""
from sage_rl.evaluation.semantics import object_matches
from collections import Counter
from sage_rl.evaluation.visual import VISUAL_DIMENSIONS, visual_reward

from sage_rl.evaluation.categories import canonical_category, category_matches


def score_task(task, rooms, objects, matched, unmatched, checks, visual, physics):
    # Partial counts carry learning signal; global legality checks do not dilute it.
    completion = []
    for req in task['objects']:
        check = next(c for c in checks if c['id'] == 'object:' + req['id'])
        actual = check['evidence']['count'] + check['evidence']['extra_matching']
        low, high = req['count']
        completion.append(min(actual / max(low, 1), 1.) if actual <= high else high / actual)
    relations = [c for c in checks if c['section'] == 'relations']
    object_score = sum(completion) / len(completion)
    relation_score = sum(c['status'] == 'pass' for c in relations) / len(relations) if relations else object_score
    furniture_checks = [c for c in checks if c['id'] == 'furniture_coverage']
    furniture_score = furniture_checks[0]['evidence']['score']
    task_score = .4 * object_score + .4 * relation_score + .2 * furniture_score
    mode = task['preferences']['object_mode']
    invalid_ids = set((physics or {}).get('unstable_objects', []))
    for c in checks:
        invalid_ids.update(c['evidence'].get('violating_ids', []))
    # Count only physically supported extras; unknown labels earn no diversity.
    from sage_rl.evaluation.checklist import supported_by
    furniture_categories = {entry["category"] for entry in task.get("furniture", {}).get("core", []) + task.get("furniture", {}).get("optional", [])}
    decorative_unmatched = [obj for obj in unmatched if not any(object_matches(c, obj) for c in furniture_categories)]
    eligible = []
    for obj in decorative_unmatched:
        # Required categories (explicit or essential) cannot earn credit by exceeding their limits.
        if any(object_matches(req['category'], obj) for req in task['objects']):
            continue
        category = (obj['_semantic'].get('diversity_group') if '_semantic' in obj
                    else canonical_category(obj['type']))
        room = next(r for r in rooms if any(o['id'] == obj['id'] for o in r.get('objects', [])))
        from sage_rl.evaluation.mesh_support import MeshUnavailable
        supported = abs(obj['position']['z'] - room['position'].get('z', 0)) <= .08
        if not supported:
            for other in objects:
                if other['id'] == obj['id']: continue
                try:
                    if supported_by(obj, other, .08):
                        supported = True; break
                except MeshUnavailable:
                    pass  # No richness credit without support evidence.
        if category and supported and obj['id'] not in invalid_ids:
            eligible.append(category)
    richness = 0.
    counts = Counter(eligible)
    if mode == 'open':
        spec = task['preferences']['richness']
        count_credit = sum(min(n, spec['per_category_cap']) for n in counts.values())
        richness = .5 * min(count_credit / spec['extra_count_target'], 1.) + .5 * min(len(counts) / spec['extra_category_target'], 1.)
        area = sum(room['dimensions']['width'] * room['dimensions']['length'] for room in rooms)
        limit = max(spec['max_total_objects'], area * spec['objects_per_m2'])
        checks.append({'id': 'capacity', 'section': 'objects', 'status': 'pass' if len(objects) <= limit else 'fail',
                       'critical': False, 'evidence': {'count': len(objects), 'limit': limit, 'area_m2': area,
                                                   'objects_per_m2': spec['objects_per_m2']}})
    visual_available = bool(visual and visual.get('success'))
    visual_raw = sum(visual['grades'][key]['mean'] for key in VISUAL_DIMENSIONS) / 40 if visual_available else 0.
    visual_score = visual_reward(visual_raw)
    base = (.6 if mode == 'strict' else .4) * task_score + (.2 * richness if mode == 'open' else 0.)
    # Concrete constraint penalties cannot be averaged away by easy pass checks.
    for c in checks:
        if c['status'] == 'fail' and c['id'] in ('no_extra_objects', 'capacity', 'furniture_capacity', 'door_access_clear'):
            base -= .2
    reward = min(1., max(0., base + .4 * visual_score))
    unknown = any(c['status'] == 'unknown' for c in checks)
    return {'method': 'checklist', 'task_id': task['task_id'], 'object_mode': mode,
            'checks': checks, 'section_scores': {'objects': object_score, 'relations': relation_score, 'furniture': furniture_score,
                'richness': richness, 'visual_raw': visual_raw, 'visual': visual_score},
            'visual_available': visual_available, 'reward_without_visual': round(min(1., max(0., base)), 6),
            'reward': round(reward, 6), 'evaluable': not unknown,
            'strict_success': not unknown and all(c['status'] == 'pass' for c in checks),
            'semantic_evidence': {o['id']: o['_semantic'] for o in objects if '_semantic' in o},
            'richness_evidence': {'eligible_category_counts': dict(counts), 'loaded_object_count': len(objects)}}
