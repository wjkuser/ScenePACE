# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import omni
from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
import numpy as np

def get_prim(prim_path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f'Prim at path {prim_path} is not valid.')
        return None
    return prim

def get_all_prim_paths(ids):
    prim_paths = [f'/World/{id}' for id in ids]
    return prim_paths

def get_all_prims_with_paths(ids):
    prim_paths = get_all_prim_paths(ids)
    prims = []
    for prim_path in prim_paths:
        prim = get_prim(prim_path)
        prims.append(prim)
    return (prims, prim_paths)

def get_all_prims_with_prim_paths(prim_paths):
    prims = []
    for prim_path in prim_paths:
        prim = get_prim(prim_path)
        assert prim is not None, f'Prim at path {prim_path} is not valid.'
        prims.append(prim)
    return prims

def extract_position_orientation(transform):
    position = Gf.Vec3d(transform.ExtractTranslation())
    rotation = transform.ExtractRotationQuat()
    orientation = Gf.Quatd(rotation.GetReal(), *rotation.GetImaginary())
    return (position, orientation)

def quaternion_angle(q1, q2):
    """
    Calculate the angle between two quaternions.

    Parameters:
    q1, q2: Lists or arrays of shape [w, x, y, z] representing quaternions

    Returns:
    angle: The angle in radians between the two quaternions
    """
    q1 = np.array(q1)
    q2 = np.array(q2)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    q_rel = np.array([q2[0] * q1_inv[0] - q2[1] * q1_inv[1] - q2[2] * q1_inv[2] - q2[3] * q1_inv[3], q2[0] * q1_inv[1] + q2[1] * q1_inv[0] + q2[2] * q1_inv[3] - q2[3] * q1_inv[2], q2[0] * q1_inv[2] - q2[1] * q1_inv[3] + q2[2] * q1_inv[0] + q2[3] * q1_inv[1], q2[0] * q1_inv[3] + q2[1] * q1_inv[2] - q2[2] * q1_inv[1] + q2[3] * q1_inv[0]])
    angle = 2 * np.arccos(min(abs(q_rel[0]), 1.0))
    return angle * 180 / np.pi

def start_simulation_and_track(prims, prim_paths, simulation_steps=2000, longterm_equilibrium_steps=20, stable_position_limit=0.2, stable_rotation_limit=8.0, early_stop_unstable_exemption_prim_paths=[]):
    import omni
    import omni.kit.app
    app = omni.kit.app.get_app()
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        timeline.stop()
    timeline.set_current_time(0.0)
    import time
    time.sleep(0.1)
    traced_data_all = {}
    init_data = {}
    timeline.play()
    elapsed_steps = 0
    init = True
    early_stop = False
    while not early_stop and elapsed_steps < simulation_steps:
        current_time_code = Usd.TimeCode.Default()
        traced_data_frame_prims = []
        for prim in prims:
            xform = UsdGeom.Xformable(prim)
            transform = xform.ComputeLocalToWorldTransform(current_time_code)
            traced_data_frame_prim = extract_position_orientation(transform)
            traced_data_frame_prims.append(traced_data_frame_prim)
        for (prim_i, (position, orientation)) in enumerate(traced_data_frame_prims):
            prim_path = prim_paths[prim_i]
            traced_data = traced_data_all.get(prim_path, [])
            if init:
                init_data[prim_path] = {}
                init_data[prim_path]['position'] = [position[0], position[1], position[2]]
                init_data[prim_path]['orientation'] = [orientation.GetReal(), orientation.GetImaginary()[0], orientation.GetImaginary()[1], orientation.GetImaginary()[2]]
                relative_position = 0.0
                relative_orientation = 0.0
                position_cur = np.array([init_data[prim_path]['position'][0], init_data[prim_path]['position'][1], init_data[prim_path]['position'][2]])
                orientation_cur = np.array([init_data[prim_path]['orientation'][0], init_data[prim_path]['orientation'][1], init_data[prim_path]['orientation'][2], init_data[prim_path]['orientation'][3]])
            else:
                position_cur = np.array([position[0], position[1], position[2]])
                position_init = np.array([init_data[prim_path]['position'][0], init_data[prim_path]['position'][1], init_data[prim_path]['position'][2]])
                orientation_cur = np.array([orientation.GetReal(), orientation.GetImaginary()[0], orientation.GetImaginary()[1], orientation.GetImaginary()[2]])
                orientation_init = np.array([init_data[prim_path]['orientation'][0], init_data[prim_path]['orientation'][1], init_data[prim_path]['orientation'][2], init_data[prim_path]['orientation'][3]])
                position_last = traced_data[0]['position_last']
                orientation_last = traced_data[0]['orientation_last']
                relative_position_last = position_cur - position_last
                relative_orientation_last = quaternion_angle(orientation_cur, orientation_last)
                relative_position_last = float(np.linalg.norm(relative_position_last))
                relative_orientation_last = float(relative_orientation_last)
                relative_position = position_cur - position_init
                relative_orientation = quaternion_angle(orientation_cur, orientation_init)
                relative_position = float(np.linalg.norm(relative_position))
                relative_orientation = float(relative_orientation)
            traced_data.append({'position': position_cur.copy(), 'orientation': orientation_cur.copy(), 'd_position': relative_position, 'd_orientation': relative_orientation, 'position_last': position_cur.copy(), 'orientation_last': orientation_cur.copy()})
            if traced_data[-1]['d_position'] > stable_position_limit or traced_data[-1]['d_orientation'] > stable_rotation_limit:
                traced_data[-1]['stable'] = False
            else:
                traced_data[-1]['stable'] = True
            if not init:
                traced_data[-1]['relative_position_last'] = relative_position_last
                traced_data[-1]['relative_orientation_last'] = relative_orientation_last
                if relative_position_last < 0.001 and relative_orientation_last < 0.001:
                    traced_data[-1]['shortterm_equilibrium'] = True
                else:
                    traced_data[-1]['shortterm_equilibrium'] = False
            if len(traced_data) > longterm_equilibrium_steps:
                traced_data.pop(0)
                longterm_equilibrium = True
                for trace_item in traced_data:
                    longterm_equilibrium = longterm_equilibrium and trace_item['shortterm_equilibrium']
                traced_data[-1]['longterm_equilibrium'] = longterm_equilibrium
            else:
                traced_data[-1]['longterm_equilibrium'] = False
            traced_data_all[prim_path] = traced_data
        all_longterm_equilibrium = True
        for (prim_path, traced_data) in traced_data_all.items():
            all_longterm_equilibrium = all_longterm_equilibrium and traced_data[-1]['longterm_equilibrium']
        if all_longterm_equilibrium:
            print('early stop: all longterm equilibrium')
            early_stop = True
        existing_stable = True
        for (prim_path, traced_data) in traced_data_all.items():
            if prim_path not in early_stop_unstable_exemption_prim_paths and (not traced_data[-1]['stable']):
                print(f'early stop: unstable prim: {prim_path}')
                existing_stable = False
                break
        if not existing_stable:
            early_stop = True
        if init:
            init = False
        app.update()
        current_time = timeline.get_current_time()
        time_step = 1.0 / 60.0
        timeline.set_current_time(current_time + time_step)
        elapsed_steps += 1
        print(f'\relapsed steps: {elapsed_steps:05d}/{simulation_steps:05d}', end='')
    traced_data_all_final = {}
    for (prim_path, traced_data) in traced_data_all.items():
        traced_data_all_final[prim_path] = {}
        traced_data_all_final[prim_path]['final_position'] = np.array(traced_data[-1]['position']).reshape(3)
        traced_data_all_final[prim_path]['final_orientation'] = np.array(traced_data[-1]['orientation']).reshape(4)
        traced_data_all_final[prim_path]['stable'] = traced_data[-1]['stable']
        traced_data_all_final[prim_path]['initial_position'] = np.array(init_data[prim_path]['position']).reshape(3)
        traced_data_all_final[prim_path]['initial_orientation'] = np.array(init_data[prim_path]['orientation']).reshape(4)
        position_list = [np.array(traced_data[trace_idx]['position']).reshape(3) for trace_idx in range(len(traced_data))]
        orientation_list = [np.array(traced_data[trace_idx]['orientation']).reshape(4) for trace_idx in range(len(traced_data))]
        traced_data_all_final[prim_path]['position_traj'] = np.array(position_list).reshape(-1, 3).astype(np.float32)
        traced_data_all_final[prim_path]['orientation_traj'] = np.array(orientation_list).reshape(-1, 4).astype(np.float32)
    timeline.stop()
    return traced_data_all_final

def start_simulation_and_track_groups(group_prims, group_prim_paths, simulation_steps=6000, longterm_equilibrium_steps=10, stable_position_limit=0.2, stable_rotation_limit=8.0, group_early_stop_unstable_exemption_prim_paths={}):
    """
    This function is used to start the simulation and track the prims of each group.
    The group_prims is a dict from group_id to a list of prims, and the group_prim_paths is a dict from group_id to a list of prim paths.
    group_early_stop_unstable_exemption_prim_paths is a dict from group_id to a list of prim paths that are exempted from the early stop.

    All groups are simulated and tracked together.
    The simulation will stop when it reaches the simulation_steps or all groups reach early stop criteria.
    """
    assert len(group_prims.keys()) == len(group_prim_paths.keys()), 'The number of groups and group prim paths must be the same'
    import omni
    import omni.kit.app
    app = omni.kit.app.get_app()
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        timeline.stop()
    timeline.set_current_time(0.0)
    import time
    time.sleep(0.1)
    all_prims = []
    all_prim_paths = []
    prim_path_to_group_id = {}
    for (group_id, prims) in group_prims.items():
        prim_paths = group_prim_paths[group_id]
        assert len(prims) == len(prim_paths), f'Group {group_id} has mismatched prims and prim_paths lengths'
        all_prims.extend(prims)
        all_prim_paths.extend(prim_paths)
        for prim_path in prim_paths:
            prim_path_to_group_id[prim_path] = group_id
    all_early_stop_unstable_exemption_prim_paths = set()
    for (group_id, exemption_paths) in group_early_stop_unstable_exemption_prim_paths.items():
        all_early_stop_unstable_exemption_prim_paths.update(exemption_paths)
    traced_data_all = {}
    init_data = {}
    timeline.play()
    elapsed_steps = 0
    init = True
    early_stop = False
    while not early_stop and elapsed_steps < simulation_steps:
        current_time_code = Usd.TimeCode.Default()
        traced_data_frame_prims = []
        for prim in all_prims:
            xform = UsdGeom.Xformable(prim)
            transform = xform.ComputeLocalToWorldTransform(current_time_code)
            traced_data_frame_prim = extract_position_orientation(transform)
            traced_data_frame_prims.append(traced_data_frame_prim)
        for (prim_i, (position, orientation)) in enumerate(traced_data_frame_prims):
            prim_path = all_prim_paths[prim_i]
            traced_data = traced_data_all.get(prim_path, [])
            if init:
                init_data[prim_path] = {}
                init_data[prim_path]['position'] = [position[0], position[1], position[2]]
                init_data[prim_path]['orientation'] = [orientation.GetReal(), orientation.GetImaginary()[0], orientation.GetImaginary()[1], orientation.GetImaginary()[2]]
                relative_position = 0.0
                relative_orientation = 0.0
                position_cur = np.array([init_data[prim_path]['position'][0], init_data[prim_path]['position'][1], init_data[prim_path]['position'][2]])
                orientation_cur = np.array([init_data[prim_path]['orientation'][0], init_data[prim_path]['orientation'][1], init_data[prim_path]['orientation'][2], init_data[prim_path]['orientation'][3]])
            else:
                position_cur = np.array([position[0], position[1], position[2]])
                position_init = np.array([init_data[prim_path]['position'][0], init_data[prim_path]['position'][1], init_data[prim_path]['position'][2]])
                orientation_cur = np.array([orientation.GetReal(), orientation.GetImaginary()[0], orientation.GetImaginary()[1], orientation.GetImaginary()[2]])
                orientation_init = np.array([init_data[prim_path]['orientation'][0], init_data[prim_path]['orientation'][1], init_data[prim_path]['orientation'][2], init_data[prim_path]['orientation'][3]])
                position_last = traced_data[0]['position_last']
                orientation_last = traced_data[0]['orientation_last']
                relative_position_last = position_cur - position_last
                relative_orientation_last = quaternion_angle(orientation_cur, orientation_last)
                relative_position_last = float(np.linalg.norm(relative_position_last))
                relative_orientation_last = float(relative_orientation_last)
                relative_position = position_cur - position_init
                relative_orientation = quaternion_angle(orientation_cur, orientation_init)
                relative_position = float(np.linalg.norm(relative_position))
                relative_orientation = float(relative_orientation)
            traced_data.append({'position': position_cur.copy(), 'orientation': orientation_cur.copy(), 'd_position': relative_position, 'd_orientation': relative_orientation, 'position_last': position_cur.copy(), 'orientation_last': orientation_cur.copy()})
            if traced_data[-1]['d_position'] > stable_position_limit or traced_data[-1]['d_orientation'] > stable_rotation_limit:
                traced_data[-1]['stable'] = False
            else:
                traced_data[-1]['stable'] = True
            if not init:
                traced_data[-1]['relative_position_last'] = relative_position_last
                traced_data[-1]['relative_orientation_last'] = relative_orientation_last
                if relative_position_last < 0.001 and relative_orientation_last < 0.001:
                    traced_data[-1]['shortterm_equilibrium'] = True
                else:
                    traced_data[-1]['shortterm_equilibrium'] = False
            if len(traced_data) > longterm_equilibrium_steps:
                traced_data.pop(0)
                longterm_equilibrium = True
                for trace_item in traced_data:
                    longterm_equilibrium = longterm_equilibrium and trace_item['shortterm_equilibrium']
                traced_data[-1]['longterm_equilibrium'] = longterm_equilibrium
            else:
                traced_data[-1]['longterm_equilibrium'] = False
            traced_data_all[prim_path] = traced_data
        all_groups_meet_early_stop = True
        for group_id in group_prims.keys():
            group_prim_paths_list = group_prim_paths[group_id]
            group_exemption_paths = set(group_early_stop_unstable_exemption_prim_paths.get(group_id, []))
            group_longterm_equilibrium = True
            for prim_path in group_prim_paths_list:
                if prim_path in traced_data_all:
                    group_longterm_equilibrium = group_longterm_equilibrium and traced_data_all[prim_path][-1]['longterm_equilibrium']
            group_stable = True
            for prim_path in group_prim_paths_list:
                if prim_path in traced_data_all and prim_path not in group_exemption_paths:
                    if not traced_data_all[prim_path][-1]['stable']:
                        group_stable = False
                        break
            group_meets_early_stop = group_longterm_equilibrium or not group_stable
            if not group_meets_early_stop:
                all_groups_meet_early_stop = False
                break
        if all_groups_meet_early_stop:
            early_stop = True
        if init:
            init = False
        app.update()
        current_time = timeline.get_current_time()
        time_step = 1.0 / 60.0
        timeline.set_current_time(current_time + time_step)
        elapsed_steps += 1
    traced_data_all_final = {}
    for group_id in group_prims.keys():
        traced_data_all_final[group_id] = {}
    for (prim_path, traced_data) in traced_data_all.items():
        group_id = prim_path_to_group_id[prim_path]
        traced_data_all_final[group_id][prim_path] = {}
        traced_data_all_final[group_id][prim_path]['final_position'] = np.array(traced_data[-1]['position']).reshape(3)
        traced_data_all_final[group_id][prim_path]['final_orientation'] = np.array(traced_data[-1]['orientation']).reshape(4)
        traced_data_all_final[group_id][prim_path]['stable'] = traced_data[-1]['stable']
        traced_data_all_final[group_id][prim_path]['initial_position'] = np.array(init_data[prim_path]['position']).reshape(3)
        traced_data_all_final[group_id][prim_path]['initial_orientation'] = np.array(init_data[prim_path]['orientation']).reshape(4)
    timeline.stop()
    return traced_data_all_final
