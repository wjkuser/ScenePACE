"""Combine measured room evidence without discarding multiroom validation."""

def combine_room_physics(results):
    if not results:
        raise ValueError('No room physics evidence')
    if len(results) == 1:
        return results[0]
    if any(r.get('status') != 'success' for r in results):
        raise ValueError('Incomplete room physics evidence')
    if any('loaded_object_ids' not in r or not isinstance(r.get('door_access', {}).get('pass'), bool) for r in results):
        raise ValueError('Missing loaded-instance or door-access evidence')
    merged = {'status': 'success', 'room_evidence': results}
    for key in ('loaded_object_ids', 'missing_object_ids', 'unstable_objects'):
        merged[key] = sorted({i for r in results for i in r.get(key, [])})
    merged['door_access'] = {'pass': all(r['door_access']['pass'] for r in results),
                             'rooms': [r['door_access'] for r in results]}
    return merged
