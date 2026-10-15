"""Isaac RTX rendering for the generation pipeline."""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd

async def render_views(extension, scene_save_dir, room_dict_save_path, output_dir, resolution=1024, views='five'):
    omni.timeline.get_timeline_interface().stop()
    room = json.loads(Path(room_dict_save_path).read_text())
    result = extension.create_single_room_layout_scene_from_room(scene_save_dir, room_dict_save_path)
    if result.get('status') != 'success':
        raise RuntimeError(result)
    omni.timeline.get_timeline_interface().stop()
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
    app = omni.kit.app.get_app()
    for _ in range(5):
        await app.next_update_async()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for prim in stage.Traverse():
        if prim.GetName().endswith('_ceiling'):
            UsdGeom.Imageable(prim).MakeInvisible()
    light = UsdLux.DomeLight.Define(stage, '/World/SageDome')
    light.CreateIntensityAttr(800)
    sun = UsdLux.DistantLight.Define(stage, '/World/SageSun')
    sun.CreateIntensityAttr(1800)
    sun.CreateAngleAttr(12)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(25, -35, 20))
    (p, d) = (room['position'], room['dimensions'])
    center = np.array([p['x'] + d['width'] / 2, p['y'] + d['length'] / 2, p['z'] + d['height'] * 0.3])
    span = max(d['width'], d['length'])
    cameras = []
    if views in ('five', 'edges'):
        for index in range(4):
            theta = index * math.pi / 2
            eye = center + np.array([math.cos(theta) * span * 1.5, math.sin(theta) * span * 1.5, span * 1.4])
            cameras.append((f'edge_{index + 1}', eye, center, False))
    if views in ('five', 'top'):
        eye = center + np.array([0, 0, span * 2])
        target = center.copy()
        target[2] = p['z']
        cameras.append(('top', eye, target, True))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for (name, eye, target, top) in cameras:
        for wall in room['walls']:
            prim = stage.GetPrimAtPath('/World/' + wall['id'])
            if not prim.IsValid():
                continue
            (a, b) = (wall['start_point'], wall['end_point'])
            midpoint = np.array([(a['x'] + b['x']) / 2, (a['y'] + b['y']) / 2])
            outward = midpoint - center[:2]
            hide = not top and np.dot(outward, eye[:2] - center[:2]) > 1e-05
            imageable = UsdGeom.Imageable(prim)
            imageable.MakeInvisible() if hide else imageable.MakeVisible()
        camera_path = '/World/SageCamera'
        camera = UsdGeom.Camera.Define(stage, camera_path)
        transform = UsdGeom.Xformable(camera)
        transform.ClearXformOpOrder()
        up = Gf.Vec3d(0, 1, 0) if top else Gf.Vec3d(0, 0, 1)
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), up).GetInverse()
        transform.AddTransformOp().Set(matrix)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000))
        if top:
            camera.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
            camera.CreateHorizontalApertureAttr(d['width'] * 11)
            camera.CreateVerticalApertureAttr(d['length'] * 11)
            size = (resolution, max(1, int(resolution * d['length'] / d['width'])))
        else:
            camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
            camera.CreateHorizontalApertureAttr(36)
            camera.CreateVerticalApertureAttr(36 * 9 / 16)
            camera.CreateFocalLengthAttr(26)
            size = (resolution, int(resolution * 9 / 16))
        product = rep.create.render_product(camera_path, size)
        annotator = rep.AnnotatorRegistry.get_annotator('rgb')
        annotator.attach([product.path])
        try:
            for _ in range(3):
                await rep.orchestrator.step_async(rt_subframes=8, pause_timeline=True, delta_time=0.0)
            data = annotator.get_data()
            rgb = np.asarray(data)[..., :3]
            if rgb.size == 0 or rgb.std() < 1:
                raise RuntimeError(f'Blank Isaac render: {name}')
            path = output / f'{name}.png'
            Image.fromarray(rgb).save(path)
            paths.append(str(path))
        finally:
            annotator.detach([product.path])
            product.destroy()
    return {'status': 'success', 'renderer': 'Isaac Sim RTX', 'paths': paths}
