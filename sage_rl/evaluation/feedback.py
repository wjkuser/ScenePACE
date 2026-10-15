"""Cheap task feedback from the saved layout; no visual call or scene replay."""
from sage_rl.evaluation.semantics import object_matches
from sage_rl.evaluation.checklist import match_objects, category_matches, supported_by


def task_feedback(task, layout):
    from sage_rl.evaluation.mesh_support import attach_mesh_root, MeshUnavailable
    attach_mesh_root(layout)
    objects = [o for room in layout.get('rooms', []) for o in room.get('objects', [])]
    matched, extras = match_objects(task['objects'], objects)
    issues = []
    for req in task['objects']:
        found = matched[req['id']]
        extra = [o for o in extras if object_matches(req['category'], o)]
        count = len(found) + len(extra)
        low, high = req['count']
        if count < low:
            issues.append({'kind': 'missing', 'category': req['category'], 'count': low-count,
                           'tool': 'place_objects_in_room'})
        elif count > high:
            issues.append({'kind': 'excess', 'category': req['category'], 'count': count-high,
                           'object_ids': [o['id'] for o in extra], 'tool': 'place_objects_in_room',
                           'action': 'remove the excess objects by ID'})
    for relation in task.get('relations', []):
        if relation['predicate'] != 'supported_by':
            continue
        parents = matched[relation['object']]
        for child in matched[relation['subject']]:
            try:
                supported = any(supported_by(child, parent, .08) for parent in parents)
            except MeshUnavailable:
                issues.append({'kind': 'support_unknown', 'object_id': child['id'], 'reason': 'mesh evidence unavailable; do not infer physical failure'})
                continue
            if parents and not supported:
                issues.append({'kind': 'support', 'object_id': child['id'],
                               'supporter_ids': [p['id'] for p in parents],
                               'tool': 'move_one_object_with_condition_in_room'})
    from sage_rl.evaluation.furniture import assess_furniture
    furniture = assess_furniture(task, objects, matched)
    if furniture['score'] < 1:
        issues.append({'kind': 'furniture_coverage', 'missing_core': furniture['missing_core'],
                       'covered_optional_types': len(furniture['covered_optional']),
                       'target_optional_types': furniture['target_optional_types'],
                       'tool': 'place_objects_in_room'})
    if furniture['excess_ids']:
        issues.append({'kind': 'excess_furniture', 'object_ids': furniture['excess_ids'],
                       'tool': 'place_objects_in_room', 'action': 'remove excess furniture instances'})
    extras = [o for o in extras if o['id'] not in furniture['exempt_ids']]
    preferences = task.get('preferences', {})
    limit = preferences.get('max_extra_objects')
    if limit is not None and len(extras) > limit:
        issues.append({'kind': 'unrelated_objects', 'allowed': limit,
                       'object_ids': [o['id'] for o in extras], 'tool': 'place_objects_in_room',
                       'action': 'remove enough extra objects to meet the limit'})
    return {'source': 'current saved layout; final Isaac loading/physics checked separately',
            'object_mode': preferences.get('object_mode'), 'issues': issues,
            'recorded_object_count': len(objects),
            'note': 'Correct explicit task violations before optional decoration; do not regenerate the room.'}
