"""Opt-in evaluation of surface-scoped, minimum-count user requests.

Only evaluation tasks explicitly carrying this contract use this matcher.
Historical training tasks retain their frozen interpretation.
"""
from itertools import product
from sage_rl.evaluation.semantics import object_matches
from sage_rl.evaluation.mesh_support import query_mesh_support, MeshUnavailable


def match_scoped(task, objects):
    requirements = {r['id']: r for r in task['objects']}
    candidates = {k: [o for o in objects if object_matches(r['category'], o)]
                  for k, r in requirements.items()}
    relations = task.get('relations', [])
    parents = list(dict.fromkeys(r['object'] for r in relations))
    evidence = {}

    def support(child, parent):
        key = (child['id'], parent['id'])
        if key not in evidence:
            try:
                evidence[key] = query_mesh_support(child, parent)
            except MeshUnavailable as exc:
                evidence[key] = {'passed': None, 'error': str(exc),
                                 'child_id': key[0], 'parent_id': key[1]}
        return evidence[key]['passed'] is True

    # Find a consistent witness, instead of selecting an arbitrary same-class
    # table and rejecting objects that actually sit on a different valid table.
    best = None
    options = [candidates[k] or [None] for k in parents]
    for selection in product(*options):
        chosen = {k: ([o] if o else []) for k, o in zip(parents, selection)}
        ids = [o['id'] for o in selection if o]
        if len(ids) != len(set(ids)):
            continue
        score = 0.0
        for rel in relations:
            ps = chosen[rel['object']]
            children = [o for o in candidates[rel['subject']]
                        if o['id'] not in ids and any(support(o, p) for p in ps)]
            need = requirements[rel['subject']]['count'][0]
            score += min(len(children) / max(need, 1), 1.)
        rank = (score, len(ids))
        if best is None or rank > best[0]:
            best = (rank, chosen)
    selected = best[1] if best else {}
    used = {o['id'] for group in selected.values() for o in group}
    scopes = {}
    for key, req in requirements.items():
        if key in selected:
            scopes[key] = candidates[key]
            continue
        own_relations = [r for r in relations if r['subject'] == key]
        available = [o for o in candidates[key] if o['id'] not in used]
        if own_relations:
            available = [o for o in available if all(
                any(support(o, p) for p in selected.get(r['object'], []))
                for r in own_relations)]
        scopes[key] = available
        selected[key] = available[:req['count'][1]]
        used.update(o['id'] for o in selected[key])
    audit = {'method': 'surface_scoped_minimum_v1',
             'selected_parent_ids': {k: [o['id'] for o in selected[k]] for k in parents},
             'scoped_candidate_ids': {k: [o['id'] for o in v] for k, v in scopes.items()},
             'support_pairs': list(evidence.values())}
    unmatched = [o for o in objects if o['id'] not in used]
    return selected, unmatched, scopes, audit
