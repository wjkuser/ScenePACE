# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vectorized Isaac placement checks for one SAGE on-object tool call."""

import asyncio
import builtins
import math
import os
import re
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

"""Vectorized Isaac placement checks for one SAGE on-object tool call."""
import asyncio
import builtins
import math
import os
import re
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple
'Vectorized Isaac placement checks for one SAGE on-object tool call.'
import asyncio
import builtins
import math
import os
import re
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple
'Vectorized Isaac placement checks for one SAGE on-object tool call.'
import asyncio
import builtins
import math
import os
import re
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import omni.timeline
import omni.usd
from omni.isaac.cloner import Cloner
from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.stage import create_new_stage, create_new_stage_async
from pxr import UsdGeom
from scipy.spatial.transform import Rotation
from isaac_sim_mcp_extension.scene.utils import dict_to_room, export_single_room_layout_to_mesh_dict_list_from_room, get_single_object_mesh_info_dict
from isaac_sim_mcp_extension.usd_utils import convert_mesh_to_usd
GPU_DYNAMICS_CAPACITIES = {'gpu_max_rigid_contact_count': 2 ** 23, 'gpu_max_rigid_patch_count': 2 ** 20, 'gpu_found_lost_pairs_capacity': 2 ** 21, 'gpu_found_lost_aggregate_pairs_capacity': 2 ** 21, 'gpu_total_aggregate_pairs_capacity': 2 ** 21, 'gpu_heap_capacity': 2 ** 28, 'gpu_temp_buffer_capacity': 2 ** 26, 'gpu_collision_stack_size': 2 ** 28}
SCHEDULING_POLICIES = {'fixed_wave', 'work_conserving'}

async def _run_sync_on_main_loop(function, *args, **kwargs):
    """Let synchronous Kit APIs pump updates without re-entering an active Task."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def invoke():
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
    loop.call_soon(invoke)
    return await future

def _safe_name(value: str) -> str:
    return re.sub('[^A-Za-z0-9_]', '_', value) or 'object'

def _quat_angle_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    dots = np.abs(np.sum(first * second, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))

def _finite_placement(placement: Dict[str, Any]) -> bool:
    try:
        values = [float(placement[section][axis]) for section in ('position', 'rotation') for axis in ('x', 'y', 'z')]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))

class BatchedPlacementSimulator:
    """Build cloned room environments once, then reuse them across candidate rounds."""

    def __init__(self, room_data: Dict[str, Any], scene_save_dir: str, items: Sequence[Dict[str, Any]], parallel_width: int=16, max_candidates_per_object: int=10, simulation_steps: int=2000, equilibrium_steps: int=20, position_limit: float=0.2, rotation_limit: float=8.0, scheduling_policy: str='work_conserving') -> None:
        self.scene_save_dir = scene_save_dir
        self.layout_id = os.path.basename(os.path.normpath(scene_save_dir))
        self.room = dict_to_room(room_data)
        self.parallel_width = max(1, min(128, int(parallel_width)))
        self.max_candidates_per_object = max(1, min(100, int(max_candidates_per_object)))
        self.simulation_steps = max(1, min(10000, int(simulation_steps)))
        self.equilibrium_steps = max(2, min(200, int(equilibrium_steps)))
        self.position_limit = float(position_limit)
        self.rotation_limit = float(rotation_limit)
        self.scheduling_policy = str(scheduling_policy).strip().lower()
        if self.scheduling_policy not in SCHEDULING_POLICIES:
            raise ValueError(f'Unknown scheduling policy {scheduling_policy!r}; expected one of {sorted(SCHEDULING_POLICIES)}')
        load_started = time.perf_counter()
        self.room_meshes = export_single_room_layout_to_mesh_dict_list_from_room(self.room, self.layout_id)
        self.items: List[Dict[str, Any]] = []
        self.rejected_results: List[Dict[str, Any]] = []
        object_ids = set()
        for (item_index, item) in enumerate(items):
            object_id = str(item['object_id'])
            if object_id in object_ids:
                raise ValueError(f'Duplicate placement object id: {object_id}')
            object_ids.add(object_id)
            source = str(item['source'])
            source_id = str(item['source_id'])
            mesh_info = get_single_object_mesh_info_dict(self.scene_save_dir, source, source_id)
            if mesh_info is None:
                raise FileNotFoundError(f'Object mesh not found: {source}/{source_id}')
            mesh = mesh_info['mesh']
            placements = list(item.get('placements', []))[:self.max_candidates_per_object]
            normalized_item = {'item_index': item_index, 'object_id': object_id, 'source': source, 'source_id': source_id, 'mass': float(item.get('mass', 1.0)), 'placements': placements, 'mesh': mesh, 'mesh_center': np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0), 'prim_name': f'candidate_{item_index}_{_safe_name(object_id)}'}
            self.items.append(normalized_item)
        self.asset_load_s = time.perf_counter() - load_started
        self.scenarios: List[Dict[str, Any]] = []
        scenario_index = 0
        for item in self.items:
            for (candidate_index, placement) in enumerate(item['placements']):
                base = {'scenario_index': scenario_index, 'item_index': item['item_index'], 'object_id': item['object_id'], 'candidate_index': candidate_index, 'placement': placement}
                if _finite_placement(placement):
                    self.scenarios.append(base)
                else:
                    self.rejected_results.append({**base, 'safe': False, 'invalid': True, 'reason': 'non_finite_or_malformed_placement'})
                scenario_index += 1

    @staticmethod
    def _destroy_world() -> None:
        import omni.kit.app
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            timeline.stop()
        app = omni.kit.app.get_app()
        if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
            for _ in range(2):
                app.update()
        if World.instance() is not None:
            World.clear_instance()

    @classmethod
    async def _destroy_world_async(cls) -> None:
        import omni.kit.app
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            timeline.stop()
        app = omni.kit.app.get_app()
        for _ in range(2):
            await app.next_update_async()
        if World.instance() is not None:
            World.clear_instance()

    @staticmethod
    def _author_mesh(stage: Any, path: str, mesh_info: Dict[str, Any], *, static: Optional[bool]=None, mass: Optional[float]=None) -> None:
        mesh = mesh_info['mesh']
        convert_mesh_to_usd(stage, path, mesh.vertices, mesh.faces, 'sdf', mesh_info.get('static', False) if static is None else static, None, mass=mesh_info.get('mass', 1.0) if mass is None else mass, physics_iter=(16, 4), texture=None, usd_internal_art_reference_path=path)

    @staticmethod
    def _configure_gpu_dynamics(physics_context: Any) -> None:
        api = physics_context._physx_scene_api
        setters = {'gpu_max_rigid_contact_count': api.CreateGpuMaxRigidContactCountAttr, 'gpu_max_rigid_patch_count': api.CreateGpuMaxRigidPatchCountAttr, 'gpu_found_lost_pairs_capacity': api.CreateGpuFoundLostPairsCapacityAttr, 'gpu_found_lost_aggregate_pairs_capacity': api.CreateGpuFoundLostAggregatePairsCapacityAttr, 'gpu_total_aggregate_pairs_capacity': api.CreateGpuTotalAggregatePairsCapacityAttr, 'gpu_heap_capacity': api.CreateGpuHeapCapacityAttr, 'gpu_temp_buffer_capacity': api.CreateGpuTempBufferCapacityAttr, 'gpu_collision_stack_size': api.CreateGpuCollisionStackSizeAttr}
        for (name, value) in GPU_DYNAMICS_CAPACITIES.items():
            setters[name](value)

    def _build_world(self, width: int, *, world: Optional[World]=None, started: Optional[float]=None, reset_world: bool=True) -> Dict[str, Any]:
        if started is None:
            started = time.perf_counter()
        if world is None:
            self._destroy_world()
            create_new_stage()
            world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0, backend='numpy', device='cpu')
        physics_context = world.get_physics_context()
        if physics_context is None:
            world._init_stage(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0, physics_prim_path='/physicsScene', backend='numpy', device='cpu')
            world._setup_default_callback_fns()
            physics_context = world.get_physics_context()
        physics_context.set_broadphase_type('GPU')
        physics_context.enable_gpu_dynamics(True)
        self._configure_gpu_dynamics(physics_context)
        stage = omni.usd.get_context().get_stage()
        cloner = Cloner()
        cloner.define_base_env('/World/envs')
        source_env = '/World/envs/env_0'
        UsdGeom.Xform.Define(stage, source_env)
        dynamic_room_ids = []
        for (mesh_id, original_mesh_info) in self.room_meshes.items():
            mesh_info = original_mesh_info
            if mesh_id.startswith('door_'):
                mesh_info = {**mesh_info, 'static': True}
            self._author_mesh(stage, f'{source_env}/{mesh_id}', mesh_info)
            if not mesh_info.get('static', False) and (not mesh_id.startswith('door_')):
                dynamic_room_ids.append(mesh_id)
        for item in self.items:
            self._author_mesh(stage, f"{source_env}/{item['prim_name']}", {'mesh': item['mesh'], 'static': False}, static=False, mass=item['mass'])
        env_paths = cloner.generate_paths('/World/envs/env', width)
        env_offsets = np.zeros((width, 3), dtype=np.float64)
        cloner.clone(source_prim_path=source_env, prim_paths=env_paths, positions=env_offsets, replicate_physics=True, base_env_path='/World/envs', copy_from_source=False)
        cloner.filter_collisions(physics_context.prim_path, '/World/collisions', env_paths, global_paths=[])
        room_views = []
        for mesh_id in dynamic_room_ids:
            expected_paths = [f'{env_path}/{mesh_id}' for env_path in env_paths]
            view = world.scene.add(RigidPrimView(prim_paths_expr=expected_paths, name=f'room_{_safe_name(mesh_id)}_{width}', reset_xform_properties=True, prepare_contact_sensors=False))
            if view.prim_paths != expected_paths:
                raise RuntimeError(f'RigidPrimView order mismatch for {mesh_id}')
            room_views.append(view)
        item_views = {}
        for item in self.items:
            expected_paths = [f"{env_path}/{item['prim_name']}" for env_path in env_paths]
            view = world.scene.add(RigidPrimView(prim_paths_expr=expected_paths, name=f"candidate_{item['item_index']}_{width}", reset_xform_properties=True, prepare_contact_sensors=False))
            if view.prim_paths != expected_paths:
                raise RuntimeError(f"RigidPrimView order mismatch for {item['object_id']}")
            item_views[item['item_index']] = view
        if reset_world:
            world.reset()
        context = {'world': world, 'width': width, 'env_offsets': env_offsets, 'room_views': room_views, 'room_default_poses': [], 'item_views': item_views, 'items': {item['item_index']: item for item in self.items}, 'build_started': started, 'build_s': 0.0}
        if reset_world:
            self._capture_default_room_poses(context)
        return context

    @staticmethod
    def _capture_default_room_poses(context: Dict[str, Any]) -> None:
        room_default_poses = []
        for view in context['room_views']:
            (positions, orientations) = view.get_world_poses()
            room_default_poses.append((np.asarray(positions, dtype=np.float32).copy(), np.asarray(orientations, dtype=np.float32).copy()))
        context['room_default_poses'] = room_default_poses
        context['build_s'] = time.perf_counter() - context['build_started']

    async def _build_world_async(self, width: int) -> Dict[str, Any]:
        """Initialize Isaac through the supported persistent-extension lifecycle."""
        await self._destroy_world_async()
        started = time.perf_counter()
        await create_new_stage_async()
        world = await _run_sync_on_main_loop(World, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0, backend='numpy', device='cpu')
        await world.initialize_simulation_context_async()
        context = await _run_sync_on_main_loop(self._build_world, width, world=world, started=started, reset_world=False)
        await world.reset_async()
        self._capture_default_room_poses(context)
        return context

    @staticmethod
    def _read_active_poses(context: Dict[str, Any], active: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        room_positions = []
        room_orientations = []
        for view in context['room_views']:
            (positions, orientations) = view.get_world_poses()
            room_positions.append(np.asarray(positions, dtype=np.float64))
            room_orientations.append(np.asarray(orientations, dtype=np.float64))
        if room_positions:
            room_pos = np.stack(room_positions, axis=1)
            room_quat = np.stack(room_orientations, axis=1)
        else:
            room_pos = np.zeros((context['width'], 0, 3), dtype=np.float64)
            room_quat = np.zeros((context['width'], 0, 4), dtype=np.float64)
        candidate_pos = np.zeros((len(active), 3), dtype=np.float64)
        candidate_quat = np.zeros((len(active), 4), dtype=np.float64)
        cached = {}
        for (env_index, scenario) in enumerate(active):
            item_index = scenario['item_index']
            if item_index not in cached:
                (positions, orientations) = context['item_views'][item_index].get_world_poses()
                cached[item_index] = (np.asarray(positions, dtype=np.float64), np.asarray(orientations, dtype=np.float64))
            candidate_pos[env_index] = cached[item_index][0][env_index]
            candidate_quat[env_index] = cached[item_index][1][env_index]
        return (room_pos[:len(active)], room_quat[:len(active)], candidate_pos, candidate_quat)

    def _set_round(self, context: Dict[str, Any], active: Sequence[Dict[str, Any]], override_poses: Optional[Dict[int, Tuple[Any, Any]]]=None) -> None:
        world = context['world']
        world.reset(soft=True)
        width = context['width']
        env_offsets = context['env_offsets']
        park_local = np.array([2000.0, 2000.0, 2000.0], dtype=np.float64)
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for (item_index, view) in context['item_views'].items():
            positions = env_offsets + park_local + np.array([3.0 * item_index, 0.0, 0.0], dtype=np.float64)
            orientations = np.tile(identity, (width, 1))
            for (env_index, scenario) in enumerate(active):
                if scenario['item_index'] != item_index:
                    continue
                if override_poses and scenario['scenario_index'] in override_poses:
                    (local_position, orientation) = override_poses[scenario['scenario_index']]
                    local_position = np.asarray(local_position, dtype=np.float64)
                    orientation = np.asarray(orientation, dtype=np.float64)
                else:
                    placement = scenario['placement']
                    local_position = np.array([placement['position'][axis] for axis in ('x', 'y', 'z')], dtype=np.float64)
                    xyzw = Rotation.from_euler('xyz', [placement['rotation'][axis] for axis in ('x', 'y', 'z')], degrees=False).as_quat()
                    orientation = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
                positions[env_index] = env_offsets[env_index] + local_position
                orientations[env_index] = orientation
            view.set_world_poses(positions=positions.astype(np.float32), orientations=orientations.astype(np.float32))
            view.set_velocities(np.zeros((width, 6), dtype=np.float32))
        for (view, (positions, orientations)) in zip(context['room_views'], context['room_default_poses']):
            view.set_world_poses(positions=positions, orientations=orientations)
            view.set_velocities(np.zeros((width, 6), dtype=np.float32))
        world.play()

    def _reset_slots(self, context: Dict[str, Any], assignments: Dict[int, Optional[Tuple[Dict[str, Any], Optional[Tuple[Any, Any]], int]]]) -> None:
        """Reset selected clone slots without disturbing active neighbors."""
        if not assignments:
            return
        slots = np.asarray(sorted(assignments), dtype=np.int64)
        offsets = context['env_offsets']
        park_local = np.array([2000.0, 2000.0, 2000.0], dtype=np.float64)
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for (view, (default_positions, default_orientations)) in zip(context['room_views'], context['room_default_poses']):
            view.set_world_poses(positions=np.asarray(default_positions[slots], dtype=np.float32), orientations=np.asarray(default_orientations[slots], dtype=np.float32), indices=slots)
            view.set_velocities(np.zeros((len(slots), 6), dtype=np.float32), indices=slots)
        for (item_index, view) in context['item_views'].items():
            positions = np.asarray(offsets[slots] + park_local + np.array([3.0 * item_index, 0.0, 0.0], dtype=np.float64), dtype=np.float32)
            orientations = np.tile(identity, (len(slots), 1)).astype(np.float32)
            for (row, slot_value) in enumerate(slots):
                slot = int(slot_value)
                assignment = assignments[slot]
                if assignment is None:
                    continue
                (scenario, override_pose, _pass_index) = assignment
                if scenario['item_index'] != item_index:
                    continue
                if override_pose is not None:
                    local_position = np.asarray(override_pose[0], dtype=np.float64)
                    orientation = np.asarray(override_pose[1], dtype=np.float64)
                else:
                    placement = scenario['placement']
                    local_position = np.asarray([placement['position'][axis] for axis in ('x', 'y', 'z')], dtype=np.float64)
                    xyzw = Rotation.from_euler('xyz', [placement['rotation'][axis] for axis in ('x', 'y', 'z')], degrees=False).as_quat()
                    orientation = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
                positions[row] = offsets[slot] + local_position
                orientations[row] = orientation
            view.set_world_poses(positions=positions, orientations=orientations, indices=slots)
            view.set_velocities(np.zeros((len(slots), 6), dtype=np.float32), indices=slots)

    @staticmethod
    def _read_slot_bodies(context: Dict[str, Any], states: Dict[int, Dict[str, Any]]) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        room_positions = []
        room_orientations = []
        for view in context['room_views']:
            (positions, orientations) = view.get_world_poses()
            room_positions.append(np.asarray(positions, dtype=np.float64))
            room_orientations.append(np.asarray(orientations, dtype=np.float64))
        item_cache = {}
        for state in states.values():
            item_index = state['scenario']['item_index']
            if item_index in item_cache:
                continue
            (positions, orientations) = context['item_views'][item_index].get_world_poses()
            item_cache[item_index] = (np.asarray(positions, dtype=np.float64), np.asarray(orientations, dtype=np.float64))
        values = {}
        for (slot, state) in states.items():
            if room_positions:
                room_pos = np.stack([value[slot] for value in room_positions], axis=0)
                room_quat = np.stack([value[slot] for value in room_orientations], axis=0)
            else:
                room_pos = np.zeros((0, 3), dtype=np.float64)
                room_quat = np.zeros((0, 4), dtype=np.float64)
            item_index = state['scenario']['item_index']
            candidate_pos = item_cache[item_index][0][slot]
            candidate_quat = item_cache[item_index][1][slot]
            values[slot] = (room_pos, room_quat, candidate_pos, candidate_quat)
        return values

    def _install_assignments(self, context: Dict[str, Any], states: Dict[int, Dict[str, Any]], assignments: Dict[int, Optional[Tuple[Dict[str, Any], Optional[Tuple[Any, Any]], int]]]) -> None:
        self._reset_slots(context, assignments)
        for (slot, assignment) in assignments.items():
            if assignment is None:
                states.pop(slot, None)
                continue
            (scenario, _override_pose, pass_index) = assignment
            states[slot] = {'scenario': scenario, 'pass_index': pass_index}
        assigned_states = {slot: states[slot] for slot in assignments if slot in states}
        for (slot, values) in self._read_slot_bodies(context, assigned_states).items():
            (room_pos, room_quat, candidate_pos, candidate_quat) = values
            init_pos = np.concatenate([room_pos, candidate_pos[None, :]], axis=0)
            init_quat = np.concatenate([room_quat, candidate_quat[None, :]], axis=0)
            states[slot].update({'init_pos': init_pos, 'init_quat': init_quat, 'history_pos': [init_pos.copy()], 'history_quat': [init_quat.copy()], 'short_history': [], 'steps': 0})

    def _dynamic_snapshot(self, context: Dict[str, Any], slot: int, state: Dict[str, Any], body_values: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], all_equilibrium: bool) -> Dict[str, Any]:
        (room_pos, room_quat, candidate_pos, candidate_quat) = body_values
        body_pos = np.concatenate([room_pos, candidate_pos[None, :]], axis=0)
        body_quat = np.concatenate([room_quat, candidate_quat[None, :]], axis=0)
        position_delta = np.linalg.norm(body_pos - state['init_pos'], axis=-1)
        rotation_delta = _quat_angle_deg(body_quat, state['init_quat'])
        stable = (position_delta <= self.position_limit) & (rotation_delta <= self.rotation_limit)
        return {'existing_stable': bool(np.all(stable[:-1])), 'candidate_stable': bool(stable[-1]), 'all_equilibrium': bool(all_equilibrium), 'steps': int(state['steps']), 'position': (candidate_pos - context['env_offsets'][slot]).tolist(), 'orientation': candidate_quat.tolist(), 'position_delta': float(position_delta[-1]), 'rotation_delta': float(rotation_delta[-1])}

    def _run_scenarios_work_conserving(self, context: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], float, Dict[str, Any]]:
        """Refill a completed clone immediately from one global scenario queue."""
        started = time.perf_counter()
        world = context['world']
        world.play()
        width = context['width']
        pending = deque(self.scenarios)
        states: Dict[int, Dict[str, Any]] = {}
        results: Dict[int, Dict[str, Any]] = {}
        logical_steps = 0
        global_steps = 0
        pass_count = 0
        queue_refill_count = 0
        retry_count = 0
        initial = {}
        for slot in range(width):
            scenario = pending.popleft() if pending else None
            initial[slot] = (scenario, None, 1) if scenario is not None else None
            if scenario is not None:
                pass_count += 1
        initial_active_count = pass_count
        self._install_assignments(context, states, initial)
        while states:
            world.step(render=False)
            global_steps += 1
            bodies = self._read_slot_bodies(context, states)
            completed: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
            for (slot, state) in list(states.items()):
                state['steps'] += 1
                (room_pos, room_quat, candidate_pos, candidate_quat) = bodies[slot]
                body_pos = np.concatenate([room_pos, candidate_pos[None, :]], axis=0)
                body_quat = np.concatenate([room_quat, candidate_quat[None, :]], axis=0)
                position_delta = np.linalg.norm(body_pos - state['init_pos'], axis=-1)
                rotation_delta = _quat_angle_deg(body_quat, state['init_quat'])
                stable = (position_delta <= self.position_limit) & (rotation_delta <= self.rotation_limit)
                short = (np.linalg.norm(body_pos - state['history_pos'][0], axis=-1) < 0.001) & (_quat_angle_deg(body_quat, state['history_quat'][0]) < 0.001)
                state['history_pos'].append(body_pos.copy())
                state['history_quat'].append(body_quat.copy())
                state['short_history'].append(short)
                if len(state['history_pos']) > self.equilibrium_steps:
                    state['history_pos'].pop(0)
                    state['history_quat'].pop(0)
                if len(state['short_history']) > self.equilibrium_steps:
                    state['short_history'].pop(0)
                all_equilibrium = bool(len(state['short_history']) == self.equilibrium_steps and np.all(np.logical_and.reduce(state['short_history'])))
                existing_stable = bool(np.all(stable[:-1]))
                candidate_exempt = state['pass_index'] == 1
                stop = all_equilibrium or not existing_stable or (not candidate_exempt and (not bool(stable[-1]))) or (state['steps'] >= self.simulation_steps)
                if stop:
                    completed[slot] = (state, self._dynamic_snapshot(context, slot, state, bodies[slot], all_equilibrium))
            if not completed:
                continue
            assignments = {}
            for slot in sorted(completed):
                (state, snapshot) = completed[slot]
                logical_steps += int(snapshot['steps'])
                scenario = state['scenario']
                scenario_index = scenario['scenario_index']
                if state['pass_index'] == 1:
                    dropped = self._dropped(scenario, snapshot)
                    snapshot.update({'dropped': dropped, 'pass_index': 1})
                    if snapshot['existing_stable'] and snapshot['candidate_stable'] and (not dropped):
                        snapshot['safe'] = True
                        results[scenario_index] = snapshot
                    elif snapshot['existing_stable'] and (not dropped):
                        snapshot['safe'] = False
                        results[scenario_index] = snapshot
                        assignments[slot] = (scenario, (snapshot['position'], snapshot['orientation']), 2)
                        pass_count += 1
                        retry_count += 1
                        continue
                    else:
                        snapshot['safe'] = False
                        results[scenario_index] = snapshot
                else:
                    first_result = results[scenario_index]
                    snapshot.update({'dropped': False, 'pass_index': 2, 'first_pass_position_delta': first_result['position_delta'], 'first_pass_rotation_delta': first_result['rotation_delta'], 'first_pass_steps': first_result['steps']})
                    snapshot['safe'] = bool(snapshot['existing_stable'] and snapshot['candidate_stable'])
                    results[scenario_index] = snapshot
                next_scenario = pending.popleft() if pending else None
                if next_scenario is None:
                    assignments[slot] = None
                else:
                    assignments[slot] = (next_scenario, None, 1)
                    pass_count += 1
                    queue_refill_count += 1
            self._install_assignments(context, states, assignments)
        sim_s = time.perf_counter() - started
        utilization = logical_steps / float(global_steps * width) if global_steps else 1.0
        metrics = {'scheduling_policy': 'work_conserving', 'synchronized_rounds': False, 'round_count': 0, 'global_step_count': global_steps, 'logical_clone_steps': logical_steps, 'slot_utilization': utilization, 'initial_active_count': initial_active_count, 'pass_count': pass_count, 'queue_refill_count': queue_refill_count, 'retry_count': retry_count, 'refill_count': queue_refill_count + retry_count}
        return (results, sim_s, metrics)

    def _simulate_pass(self, context: Dict[str, Any], active: Sequence[Dict[str, Any]], *, candidate_exempt: bool, override_poses: Optional[Dict[int, Tuple[Any, Any]]]=None) -> Tuple[Dict[int, Dict[str, Any]], float]:
        started = time.perf_counter()
        self._set_round(context, active, override_poses=override_poses)
        world = context['world']
        (init_room_pos, init_room_quat, init_candidate_pos, init_candidate_quat) = self._read_active_poses(context, active)
        init_pos = np.concatenate([init_room_pos, init_candidate_pos[:, None, :]], axis=1)
        init_quat = np.concatenate([init_room_quat, init_candidate_quat[:, None, :]], axis=1)
        history_pos = [init_pos.copy()]
        history_quat = [init_quat.copy()]
        short_history: List[np.ndarray] = []
        done = np.zeros(len(active), dtype=bool)
        snapshots: Dict[int, Dict[str, Any]] = {}
        for step_index in range(self.simulation_steps):
            world.step(render=False)
            (room_pos, room_quat, candidate_pos, candidate_quat) = self._read_active_poses(context, active)
            body_pos = np.concatenate([room_pos, candidate_pos[:, None, :]], axis=1)
            body_quat = np.concatenate([room_quat, candidate_quat[:, None, :]], axis=1)
            position_delta = np.linalg.norm(body_pos - init_pos, axis=-1)
            rotation_delta = _quat_angle_deg(body_quat, init_quat)
            stable = (position_delta <= self.position_limit) & (rotation_delta <= self.rotation_limit)
            short = (np.linalg.norm(body_pos - history_pos[0], axis=-1) < 0.001) & (_quat_angle_deg(body_quat, history_quat[0]) < 0.001)
            history_pos.append(body_pos.copy())
            history_quat.append(body_quat.copy())
            short_history.append(short)
            if len(history_pos) > self.equilibrium_steps:
                history_pos.pop(0)
                history_quat.pop(0)
            if len(short_history) > self.equilibrium_steps:
                short_history.pop(0)
            longterm = np.logical_and.reduce(short_history) if len(short_history) == self.equilibrium_steps else np.zeros_like(stable)
            all_equilibrium = np.all(longterm, axis=1)
            existing_stable = np.all(stable[:, :-1], axis=1)
            stop = all_equilibrium | ~existing_stable
            if not candidate_exempt:
                stop |= ~stable[:, -1]
            newly_done = np.flatnonzero(stop & ~done)
            for env_index in newly_done:
                scenario = active[int(env_index)]
                snapshots[scenario['scenario_index']] = {'existing_stable': bool(existing_stable[env_index]), 'candidate_stable': bool(stable[env_index, -1]), 'all_equilibrium': bool(all_equilibrium[env_index]), 'steps': step_index + 1, 'position': (candidate_pos[env_index] - context['env_offsets'][env_index]).tolist(), 'orientation': candidate_quat[env_index].tolist(), 'position_delta': float(position_delta[env_index, -1]), 'rotation_delta': float(rotation_delta[env_index, -1])}
            done |= stop
            if bool(np.all(done)):
                break
        if not bool(np.all(done)):
            (room_pos, room_quat, candidate_pos, candidate_quat) = self._read_active_poses(context, active)
            body_pos = np.concatenate([room_pos, candidate_pos[:, None, :]], axis=1)
            body_quat = np.concatenate([room_quat, candidate_quat[:, None, :]], axis=1)
            position_delta = np.linalg.norm(body_pos - init_pos, axis=-1)
            rotation_delta = _quat_angle_deg(body_quat, init_quat)
            stable = (position_delta <= self.position_limit) & (rotation_delta <= self.rotation_limit)
            for env_index in np.flatnonzero(~done):
                scenario = active[int(env_index)]
                snapshots[scenario['scenario_index']] = {'existing_stable': bool(np.all(stable[env_index, :-1])), 'candidate_stable': bool(stable[env_index, -1]), 'all_equilibrium': False, 'steps': self.simulation_steps, 'position': (candidate_pos[env_index] - context['env_offsets'][env_index]).tolist(), 'orientation': candidate_quat[env_index].tolist(), 'position_delta': float(position_delta[env_index, -1]), 'rotation_delta': float(rotation_delta[env_index, -1])}
        return (snapshots, time.perf_counter() - started)

    def _dropped(self, scenario: Dict[str, Any], result: Dict[str, Any]) -> bool:
        item = self.items[scenario['item_index']]
        quat = np.asarray(result['orientation'], dtype=np.float64)
        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        center_z = float((rotation.as_matrix() @ item['mesh_center'])[2] + result['position'][2])
        return center_z < float(scenario['placement']['position']['z']) - 0.01

    def _run_scenarios(self, context: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], float, int]:
        results: Dict[int, Dict[str, Any]] = {}
        sim_s = 0.0
        rounds = 0
        width = context['width']
        for chunk_start in range(0, len(self.scenarios), width):
            active = self.scenarios[chunk_start:chunk_start + width]
            (first, duration) = self._simulate_pass(context, active, candidate_exempt=True)
            sim_s += duration
            rounds += 1
            retry = []
            retry_poses = {}
            for scenario in active:
                result = first[scenario['scenario_index']]
                dropped = self._dropped(scenario, result)
                result.update({'dropped': dropped, 'pass_index': 1})
                if result['existing_stable'] and result['candidate_stable'] and (not dropped):
                    result['safe'] = True
                    results[scenario['scenario_index']] = result
                elif result['existing_stable'] and (not dropped):
                    retry.append(scenario)
                    retry_poses[scenario['scenario_index']] = (result['position'], result['orientation'])
                    result['safe'] = False
                    results[scenario['scenario_index']] = result
                else:
                    result['safe'] = False
                    results[scenario['scenario_index']] = result
            if retry:
                (second, duration) = self._simulate_pass(context, retry, candidate_exempt=False, override_poses=retry_poses)
                sim_s += duration
                rounds += 1
                for scenario in retry:
                    first_result = results[scenario['scenario_index']]
                    result = second[scenario['scenario_index']]
                    result.update({'dropped': False, 'pass_index': 2, 'first_pass_position_delta': first_result['position_delta'], 'first_pass_rotation_delta': first_result['rotation_delta'], 'first_pass_steps': first_result['steps']})
                    result['safe'] = bool(result['existing_stable'] and result['candidate_stable'])
                    results[scenario['scenario_index']] = result
        return (results, sim_s, rounds)

    def _run_scheduled_scenarios(self, context: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], float, Dict[str, Any]]:
        if self.scheduling_policy == 'work_conserving':
            return self._run_scenarios_work_conserving(context)
        (results, sim_s, rounds) = self._run_scenarios(context)
        pass_count = sum((int(result['pass_index']) for result in results.values()))
        return (results, sim_s, {'scheduling_policy': 'fixed_wave', 'synchronized_rounds': True, 'round_count': rounds, 'pass_count': pass_count})

    def _normalize_result(self, scenario: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        position = np.asarray(result['position'], dtype=np.float64)
        quat = np.asarray(result['orientation'], dtype=np.float64)
        finite = bool(np.all(np.isfinite(position)) and np.all(np.isfinite(quat)))
        safe = bool(result.get('safe')) and finite
        if finite:
            euler = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz', degrees=False)
        else:
            euler = np.zeros(3, dtype=np.float64)
        normalized_position = float(result['position_delta']) / self.position_limit
        normalized_rotation = float(result['rotation_delta']) / self.rotation_limit
        if result.get('pass_index') == 2:
            normalized_position += float(result.get('first_pass_position_delta', 0.0)) / self.position_limit
            normalized_rotation += float(result.get('first_pass_rotation_delta', 0.0)) / self.rotation_limit
        stability_cost = normalized_position + normalized_rotation + (1.0 if result.get('pass_index') == 2 else 0.0) + 0.01 * float(result['steps']) / self.simulation_steps
        return {'scenario_index': scenario['scenario_index'], 'object_id': scenario['object_id'], 'candidate_index': scenario['candidate_index'], 'safe': safe, 'invalid': not finite, 'position': {axis: float(position[index]) for (index, axis) in enumerate(('x', 'y', 'z'))}, 'rotation': {axis: float(euler[index]) for (index, axis) in enumerate(('x', 'y', 'z'))}, 'position_delta_m': float(result['position_delta']), 'rotation_delta_deg': float(result['rotation_delta']), 'pass_index': int(result['pass_index']), 'simulation_steps': int(result['steps']), 'stability_cost': float(stability_cost) if finite else math.inf, 'dropped': bool(result.get('dropped', False)), 'existing_stable': bool(result.get('existing_stable', False)), 'candidate_stable': bool(result.get('candidate_stable', False))}

    def run(self) -> Dict[str, Any]:
        if not self.scenarios:
            results = sorted(self.rejected_results, key=lambda value: value['scenario_index'])
            return {'status': 'success', 'message': 'No finite placement candidates to simulate', 'results': results, 'metrics': {'asset_load_s': self.asset_load_s, 'build_s': 0.0, 'sim_s': 0.0, 'total_s': self.asset_load_s, 'parallel_width': 0, 'round_count': 0, 'scheduling_policy': self.scheduling_policy, 'candidate_count': len(results), 'safe_count': 0, 'scene_build_count': 0}}
        width = min(self.parallel_width, len(self.scenarios))
        context = None
        total_started = time.perf_counter()
        try:
            context = self._build_world(width)
            (raw_results, sim_s, scheduler_metrics) = self._run_scheduled_scenarios(context)
            normalized = [self._normalize_result(scenario, raw_results[scenario['scenario_index']]) for scenario in self.scenarios]
            normalized.extend(self.rejected_results)
            normalized.sort(key=lambda value: value['scenario_index'])
            return {'status': 'success', 'message': 'Batched placement simulation completed', 'results': normalized, 'metrics': {'asset_load_s': self.asset_load_s, 'build_s': context['build_s'], 'sim_s': sim_s, 'total_s': self.asset_load_s + time.perf_counter() - total_started, 'parallel_width': width, 'candidate_count': len(normalized), 'safe_count': sum((bool(value.get('safe')) for value in normalized)), 'scene_build_count': 1, 'gpu_dynamics_capacities': GPU_DYNAMICS_CAPACITIES, **scheduler_metrics}}
        finally:
            self._destroy_world()

    async def run_async(self) -> Dict[str, Any]:
        """Run inside an Isaac extension without blocking its async initialization."""
        if not self.scenarios:
            return self.run()
        width = min(self.parallel_width, len(self.scenarios))
        context = None
        total_started = time.perf_counter()
        try:
            context = await self._build_world_async(width)
            (raw_results, sim_s, scheduler_metrics) = await _run_sync_on_main_loop(self._run_scheduled_scenarios, context)
            normalized = [self._normalize_result(scenario, raw_results[scenario['scenario_index']]) for scenario in self.scenarios]
            normalized.extend(self.rejected_results)
            normalized.sort(key=lambda value: value['scenario_index'])
            return {'status': 'success', 'message': 'Batched placement simulation completed', 'results': normalized, 'metrics': {'asset_load_s': self.asset_load_s, 'build_s': context['build_s'], 'sim_s': sim_s, 'total_s': self.asset_load_s + time.perf_counter() - total_started, 'parallel_width': width, 'candidate_count': len(normalized), 'safe_count': sum((bool(value.get('safe')) for value in normalized)), 'scene_build_count': 1, 'gpu_dynamics_capacities': GPU_DYNAMICS_CAPACITIES, **scheduler_metrics}}
        finally:
            await self._destroy_world_async()
