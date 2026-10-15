"""CPU mesh support queries; missing assets are unknown, never bbox successes."""
from functools import lru_cache
from pathlib import Path
import os

class MeshUnavailable(RuntimeError):
    pass


def attach_mesh_root(layout):
    root = layout.get('_mesh_root')
    if root is None:
        layout_id = layout.get('id') or layout.get('layout_id')
        if layout_id:
            root = str(Path(os.environ.get('SAGE_RESULTS_DIR', 'server/results')) / str(layout_id))
    for room in layout.get('rooms', []):
        for obj in room.get('objects', []):
            if root: obj['_mesh_root'] = root
    return layout


@lru_cache(maxsize=128)
def _load(path, size, mtime):
    import trimesh
    return trimesh.load(path, force='mesh', skip_materials=True, process=False)


def world_mesh(obj):
    import numpy as np
    from scipy.spatial.transform import Rotation
    root = obj.get('_mesh_root')
    source, sid = obj.get('source'), obj.get('source_id')
    if not root or not source or not sid:
        raise MeshUnavailable('Missing mesh root/source/source_id')
    if any(Path(str(x)).name != str(x) or x in ('.', '..') for x in (source, sid)):
        raise MeshUnavailable('Unsafe mesh identifier')
    path = Path(root) / source / (str(sid) + '.obj')
    if not path.is_file():
        raise MeshUnavailable(f'Missing mesh: {path}')
    stat = path.stat()
    try:
        mesh = _load(str(path), stat.st_size, stat.st_mtime_ns).copy()
        if not len(mesh.faces): raise ValueError('Empty mesh')
        transform = np.eye(4)
        transform[:3,:3] = Rotation.from_euler('xyz', [obj.get('rotation', {}).get(k, 0) for k in 'xyz'], degrees=True).as_matrix()
        transform[:3,3] = [obj['position'][k] for k in 'xyz']
        mesh.apply_transform(transform)
        return mesh
    except Exception as exc:
        raise MeshUnavailable(f'Cannot load mesh: {path}: {type(exc).__name__}') from exc


def query_mesh_support(child, parent, *, max_gap=.03, min_coverage=.7):
    """Sample real lower surfaces, then raycast onto the specified parent.

    Independent of historical bbox tolerance: 3cm gap, 2cm penetration allowance,
    near-horizontal support faces, and >=70% coverage of the bottom 1cm contact
    band.  Imported Blender assets do not consistently preserve outward face
    winding, so the normal test is winding-agnostic (``abs(normal.z) >= .5``).
    The ray direction and topmost hit still come from the real mesh; this is not
    a bounding-box or parent-intent success.
    """
    import numpy as np
    try:
        cm, pm = world_mesh(child), world_mesh(parent)
        lo, hi = cm.bounds
        xy = np.array([(x,y) for x in np.linspace(lo[0],hi[0],23)[1:-1]
                        for y in np.linspace(lo[1],hi[1],23)[1:-1]])
        origins = np.c_[xy, np.full(len(xy), lo[2]-.1)]
        hits, indices, _ = cm.ray.intersects_location(origins, np.tile([0,0,1], (len(xy),1)), multiple_hits=True)
        bottom = {}
        for hit, idx in zip(hits, indices):
            if idx not in bottom or hit[2] < bottom[idx][2]: bottom[idx] = hit
        if not bottom: raise MeshUnavailable('No sampled child lower surface')
        points = np.array(list(bottom.values()))
        points = points[points[:,2] <= points[:,2].min()+.01]
        origins = points.copy(); origins[:,2] += .02
        hits, indices, faces = pm.ray.intersects_location(origins, np.tile([0,0,-1], (len(points),1)), multiple_hits=True)
        nearest = {}
        for hit, idx, face in zip(hits, indices, faces):
            if idx not in nearest or hit[2] > nearest[idx][0][2]: nearest[idx] = (hit,face)
        gaps, valid = [], 0
        for idx, point in enumerate(points):
            if idx not in nearest: continue
            hit,face = nearest[idx]; gap = float(point[2]-hit[2]); gaps.append(gap)
            # OBJ/PLY exports from Blender can have the whole component wound in
            # reverse.  The ray already selects the uppermost real intersection;
            # accepting either winding avoids rejecting a geometrically touching
            # tabletop solely because its normal points downward.
            valid += int(-.02 <= gap <= max_gap and abs(pm.face_normals[face,2]) >= .5)
        coverage = valid / len(points)
        return dict(method='mesh_downward_rays_v2_winding_agnostic', child_id=child['id'], parent_id=parent['id'],
                    passed=coverage >= min_coverage, samples=len(points), coverage=coverage,
                    min_coverage=min_coverage, max_gap_m=max_gap, max_penetration_m=.02,
                    median_gap_m=float(np.median(gaps)) if gaps else None)
    except MeshUnavailable: raise
    except Exception as exc:
        raise MeshUnavailable(f'Mesh ray query failed: {type(exc).__name__}: {exc}') from exc
