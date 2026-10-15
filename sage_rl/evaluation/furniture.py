"""Furniture coverage and bounded exemptions from decorative object limits."""
from sage_rl.evaluation.semantics import object_matches
from sage_rl.evaluation.categories import category_matches


def assess_furniture(task, objects, matched):
    spec = task['furniture']
    entries = spec['core'] + spec['optional']
    explicit_ids = {obj['id'] for group in matched.values() for obj in group}
    exempt_ids = set()
    counts = {}
    excess_ids = []
    for entry in entries:
        category = entry['category']
        instances = [obj for obj in objects if (obj['_semantic']['furniture_role'] == category
                     if '_semantic' in obj else object_matches(category, obj))]
        counts[category] = len(instances)
        explicit = next((req for req in task['objects'] if req['category'] == category), None)
        scoped = task.get('evaluation', {}).get('requirement_contract') == 'surface_scoped_minimum_v1'
        # Minimum required count is not an unlimited furniture-capacity budget.
        limit = entry['max_count'] if scoped else (explicit['count'][1] if explicit else entry['max_count'])
        # Count explicitly required instances against the allowance, not on top of it.
        instances.sort(key=lambda obj: obj['id'] not in explicit_ids)
        exempt_ids.update(obj['id'] for obj in instances[:limit])
        excess_ids.extend(obj['id'] for obj in instances[limit:])
    missing_core = [entry['category'] for entry in spec['core'] if not counts[entry['category']]]
    covered = [entry['category'] for entry in spec['optional'] if counts[entry['category']] > 0]
    core_fraction = 1 - len(missing_core) / len(spec['core'])
    optional_fraction = min(len(covered) / spec['min_optional_types'], 1.)
    score = core_fraction * (.5 + .5 * optional_fraction)
    return {'score': score, 'missing_core': missing_core, 'covered_optional': covered,
            'target_optional_types': spec['min_optional_types'], 'counts': counts,
            'exempt_ids': sorted(exempt_ids), 'excess_ids': excess_ids}
