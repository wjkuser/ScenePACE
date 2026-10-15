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
"""
MIT License

Copyright (c) 2023-2025 omni-mcp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
'Extension module for Isaac Sim MCP.'
import asyncio
import carb
import tempfile
import omni.usd
import threading
import time
import socket
import json
import traceback
import sys
import gc
from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils, Sdf, UsdShade
import omni
import omni.kit.commands
import omni.physx as _physx
import omni.timeline
from typing import Dict, Any, List, Optional, Union
from omni.isaac.core.prims import XFormPrim
import numpy as np
from omni.isaac.core import World
from isaac_sim_mcp_extension.usd import USDLoader
import requests
from isaac_sim_mcp_extension.usd_utils import convert_mesh_to_usd, convert_mesh_to_usd_simple, door_frame_to_usd
from isaac_sim_mcp_extension.sim_utils import get_all_prims_with_paths, get_all_prims_with_prim_paths, start_simulation_and_track, start_simulation_and_track_groups
from pxr import Usd, UsdUtils
from isaac_sim_mcp_extension.scene.utils import dict_to_floor_plan, dict_to_room, export_layout_to_mesh_dict_list, export_single_room_layout_to_mesh_dict_list, export_single_room_layout_to_mesh_dict_list_from_room, get_single_object_mesh_info_dict, apply_object_transform_direct, export_layout_to_mesh_dict_list_no_object_transform
import os
from tqdm import tqdm
from datetime import datetime
import hashlib
import pdb

def slurm_job_id_to_port(job_id, port_start=8080, port_end=40000):
    """
    Hash-based mapping function to convert SLURM job ID to a port number.
    
    Args:
        job_id (str or int): SLURM job ID
        port_start (int): Starting port number (default: 8080)
        port_end (int): Ending port number (default: 40000)
    
    Returns:
        int: Mapped port number within the specified range
    """
    job_id_str = str(job_id)
    hash_obj = hashlib.md5(job_id_str.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    port_range = port_end - port_start + 1
    mapped_port = port_start + hash_int % port_range
    return mapped_port
try:
    import omni.isaac.core.utils.prims as prim_utils
except ModuleNotFoundError:
    import isaacsim.core.utils.prims as prim_utils
from scipy.spatial.transform import Rotation as R
import sys
isaac_ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.islink(isaac_ext_dir):
    isaac_ext_dir = os.readlink(isaac_ext_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(isaac_ext_dir))))
SERVER_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(isaac_ext_dir)))

class MCPExtension(omni.ext.IExt):

    def __init__(self) -> None:
        """Initialize the extension."""
        super().__init__()
        self.ext_id = None
        self.running = False
        self.host = None
        self.port = None
        self.socket = None
        self.server_thread = None
        self._usd_context = None
        self._physx_interface = None
        self._timeline = None
        self._window = None
        self._status_label = None
        self._server_thread = None
        self._models = None
        self._settings = carb.settings.get_settings()
        self._image_url_cache = {}
        self._text_prompt_cache = {}
        self.track_ids = []

    def get_port(self):
        configured_port = os.environ.get('ISAAC_MCP_PORT')
        if configured_port:
            return int(configured_port)
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        port = slurm_job_id_to_port(slurm_job_id)
        return port

    def on_startup(self, ext_id: str):
        """Initialize extension and UI elements"""
        print('trigger  on_startup for: ', ext_id)
        print('settings: ', self._settings.get('/exts/omni.kit.pipapi'))
        self.port = self.get_port()
        self.host = 'localhost'
        if not hasattr(self, 'running'):
            self.running = False
        self.ext_id = ext_id
        self._usd_context = omni.usd.get_context()
        self._start()

    def on_shutdown(self):
        print('trigger  on_shutdown for: ', self.ext_id)
        self._models = {}
        gc.collect()
        self._stop()

    def _start(self):
        if self.running:
            print('Server is already running')
            return
        self.running = True
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()
            print(f'Isaac Sim MCP server started on {self.host}:{self.port}')
        except Exception as e:
            print(f'Failed to start server: {str(e)}')
            self.stop()

    def _stop(self):
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        if self.server_thread:
            try:
                if self.server_thread.is_alive():
                    self.server_thread.join(timeout=1.0)
            except:
                pass
            self.server_thread = None
        print('Isaac Sim MCP server stopped')

    def _server_loop(self):
        """Main server loop in a separate thread"""
        print('Server thread started')
        self.socket.settimeout(1.0)
        if not hasattr(self, 'running'):
            self.running = False
        while self.running:
            try:
                try:
                    (client, address) = self.socket.accept()
                    print(f'Connected to client: {address}')
                    client_thread = threading.Thread(target=self._handle_client, args=(client,))
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f'Error accepting connection: {str(e)}')
                    time.sleep(0.5)
            except Exception as e:
                print(f'Error in server loop: {str(e)}')
                if not self.running:
                    break
                time.sleep(0.5)
        print('Server thread stopped')

    def _handle_client(self, client):
        """Handle connected client"""
        print('Client handler started')
        client.settimeout(None)
        buffer = b''
        try:
            while self.running:
                try:
                    data = client.recv(16384)
                    if not data:
                        print('Client disconnected')
                        break
                    buffer += data
                    try:
                        command = json.loads(buffer.decode('utf-8'))
                        buffer = b''

                        async def execute_unlocked(command=command):
                            try:
                                if command.get('type') == 'test_object_placements_batch_in_single_room':
                                    response = await self.execute_command_async(command)
                                elif command.get('type') == 'render_room_views':
                                    from isaac_sim_mcp_extension.room_views import render_views
                                    result = await render_views(self, **command.get('params', {}))
                                    response = {'status': 'success', 'result': result}
                                else:
                                    loop = asyncio.get_running_loop()
                                    future = loop.create_future()

                                    def run_sync():
                                        try:
                                            future.set_result(self.execute_command(command))
                                        except Exception as error:
                                            future.set_exception(error)
                                    loop.call_soon(run_sync)
                                    response = await future
                                response_json = json.dumps(response)
                                print('response_json: ', response_json)
                                try:
                                    client.sendall(response_json.encode('utf-8'))
                                except:
                                    print('Failed to send response - client disconnected')
                            except Exception as e:
                                print(f'Error executing command: {str(e)}')
                                traceback.print_exc()
                                try:
                                    error_response = {'status': 'error', 'message': str(e)}
                                    client.sendall(json.dumps(error_response).encode('utf-8'))
                                except:
                                    pass
                            return None

                        async def execute_wrapper(execute_unlocked=execute_unlocked):
                            if not hasattr(self, '_scene_command_lock'):
                                self._scene_command_lock = asyncio.Lock()
                            async with self._scene_command_lock:
                                return await execute_unlocked()
                        from omni.kit.async_engine import run_coroutine
                        task = run_coroutine(execute_wrapper())
                    except json.JSONDecodeError:
                        pass
                except Exception as e:
                    print(f'Error receiving data: {str(e)}')
                    break
        except Exception as e:
            print(f'Error in client handler: {str(e)}')
        finally:
            try:
                client.close()
            except:
                pass
            print('Client handler stopped')

    def execute_command(self, command):
        """Execute a command in the main thread"""
        try:
            cmd_type = command.get('type')
            params = command.get('params', {})
            if cmd_type in ['create_object', 'modify_object', 'delete_object']:
                self._usd_context = omni.usd.get_context()
                self._execute_command_internal(command)
            else:
                return self._execute_command_internal(command)
        except Exception as e:
            print(f'Error executing command: {str(e)}')
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    async def execute_command_async(self, command):
        """Execute commands that require Isaac's extension async lifecycle."""
        try:
            cmd_type = command.get('type')
            if cmd_type != 'test_object_placements_batch_in_single_room':
                return self.execute_command(command)
            params = command.get('params', {})
            print(f'Executing async handler for {cmd_type}')
            result = await self.test_object_placements_batch_in_single_room_async(**params)
            if result and result.get('status') == 'success':
                return {'status': 'success', 'result': result}
            return {'status': 'error', 'message': (result or {}).get('message', 'Unknown error')}
        except Exception as error:
            print(f'Error executing async command: {error}')
            traceback.print_exc()
            return {'status': 'error', 'message': str(error)}

    def _execute_command_internal(self, command):
        """Internal command execution with proper context"""
        print('command: ', command, file=sys.stderr)
        cmd_type = command.get('type')
        params = command.get('params', {})
        handlers = {'execute_script': self.execute_script, 'get_scene_info': self.get_scene_info, 'omini_kit_command': self.omini_kit_command, 'create_room_layout_scene': self.create_room_layout_scene, 'create_single_room_layout_scene': self.create_single_room_layout_scene, 'create_room_groups_layouts': self.create_room_groups_layouts, 'simulate_the_scene_groups': self.simulate_the_scene_groups, 'create_single_room_layout_scene_from_room': self.create_single_room_layout_scene_from_room, 'get_room_layout_scene_usd': self.get_room_layout_scene_usd, 'simulate_the_scene': self.simulate_the_scene, 'validate_room_stability': self.validate_room_stability, 'test_object_placements_in_single_room': self.test_object_placements_in_single_room, 'get_room_layout_scene_usd_separate': self.get_room_layout_scene_usd_separate, 'get_room_layout_scene_usd_separate_from_layout': self.get_room_layout_scene_usd_separate_from_layout}
        handler = handlers.get(cmd_type)
        if handler:
            try:
                print(f'Executing handler for {cmd_type}')
                result = handler(**params)
                print(f'Handler execution complete: /n', result)
                if result and result.get('status') == 'success':
                    return {'status': 'success', 'result': result}
                else:
                    return {'status': 'error', 'message': result.get('message', 'Unknown error')}
            except Exception as e:
                print(f'Error in handler: {str(e)}')
                traceback.print_exc()
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': f'Unknown command type: {cmd_type}'}

    def execute_script(self, code: str):
        """Execute a scene inspection script in Isaac."""
        try:
            local_ns = {}
            local_ns['omni'] = omni
            local_ns['carb'] = carb
            local_ns['Usd'] = Usd
            local_ns['UsdGeom'] = UsdGeom
            local_ns['Sdf'] = Sdf
            local_ns['Gf'] = Gf
            exec(code, local_ns)
            result = None
            return {'status': 'success', 'message': 'Script executed successfully', 'result': result}
        except Exception as e:
            carb.log_error(f'Error executing script: {e}')
            import traceback
            carb.log_error(traceback.format_exc())
            return {'status': 'error', 'message': str(e), 'traceback': traceback.format_exc()}

    def get_scene_info(self):
        self._stage = omni.usd.get_context().get_stage()
        assert self._stage is not None
        stage_path = self._stage.GetRootLayer().realPath
        assets_root_path = os.environ.get('ISAAC_ASSETS_ROOT', '')
        return {'status': 'success', 'message': 'pong', 'assets_root_path': assets_root_path}

    def omini_kit_command(self, command: str, prim_type: str) -> Dict[str, Any]:
        omni.kit.commands.execute(command, prim_type=prim_type)
        print('command executed')
        return {'status': 'success', 'message': 'command executed'}

    def create_room_layout_scene(self, scene_save_dir: str):
        """
        Create a room layout scene from a dictionary of mesh information.
        """
        try:
            current_layout_id = os.path.basename(scene_save_dir)
            json_file_path = os.path.join(scene_save_dir, f'{current_layout_id}.json')
            try:
                with open(json_file_path, 'r') as f:
                    layout_data = json.load(f)
            except FileNotFoundError:
                return json.dumps({'success': False, 'error': f'JSON file not found: {json_file_path}'})
            floor_plan = dict_to_floor_plan(layout_data)
            current_layout = floor_plan
            mesh_info_dict = export_layout_to_mesh_dict_list(current_layout)
            stage = Usd.Stage.CreateInMemory()
            world_base_prim = UsdGeom.Xform.Define(stage, '/World')
            stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
            collision_approximation = 'sdf'
            self.track_ids = []
            door_ids = []
            door_frame_ids = []
            for mesh_id in mesh_info_dict:
                if mesh_id.startswith('wall_room_') or mesh_id.startswith('window_') or mesh_id.startswith('floor_'):
                    usd_internal_path = f'/World/{mesh_id}'
                elif mesh_id.startswith('door_'):
                    if mesh_id.endswith('_frame'):
                        door_frame_ids.append(mesh_id)
                    else:
                        door_ids.append(mesh_id)
                    continue
                else:
                    self.track_ids.append(mesh_id)
                    usd_internal_path = f'/World/{mesh_id}'
                mesh_dict = mesh_info_dict[mesh_id]
                mesh_obj_i = mesh_dict['mesh']
                static = mesh_dict['static']
                mass = mesh_dict.get('mass', 1.0)
                articulation = mesh_dict.get('articulation', None)
                texture = mesh_dict.get('texture', None)
                stage = convert_mesh_to_usd(stage, usd_internal_path, mesh_obj_i.vertices, mesh_obj_i.faces, collision_approximation, static, articulation, mass=mass, physics_iter=(16, 1), apply_debug_torque=False, debug_torque_value=30.0, texture=texture, usd_internal_art_reference_path=f'/World/{mesh_id}')
            door_ids = sorted(door_ids)
            door_frame_ids = sorted(door_frame_ids)
            for (door_id, door_frame_id) in zip(door_ids, door_frame_ids):
                usd_internal_path_door = f'/World/{door_id}'
                usd_internal_path_door_frame = f'/World/{door_frame_id}'
                mesh_dict_door = mesh_info_dict[door_id]
                mesh_obj_door = mesh_dict_door['mesh']
                articulation_door = mesh_dict_door.get('articulation', None)
                texture_door = mesh_dict_door.get('texture', None)
                mesh_dict_door_frame = mesh_info_dict[door_frame_id]
                mesh_obj_door_frame = mesh_dict_door_frame['mesh']
                texture_door_frame = mesh_dict_door_frame.get('texture', None)
                stage = door_frame_to_usd(stage, usd_internal_path_door, usd_internal_path_door_frame, mesh_obj_door, mesh_obj_door_frame, articulation_door, texture_door, texture_door_frame)
            cache = UsdUtils.StageCache.Get()
            stage_id = cache.Insert(stage).ToLongInt()
            omni.usd.get_context().attach_stage_with_callback(stage_id)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            return {'status': 'success', 'message': f'Room layout scene created successfully'}
        except Exception as e:
            print(f'Error creating room layout scene: {str(e)}')
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def create_single_room_layout_scene(self, scene_save_dir: str, room_id: str):
        """
        Create a room layout scene from a dictionary of mesh information.
        """
        try:
            current_layout_id = os.path.basename(scene_save_dir)
            json_file_path = os.path.join(scene_save_dir, f'{current_layout_id}.json')
            try:
                with open(json_file_path, 'r') as f:
                    layout_data = json.load(f)
            except FileNotFoundError:
                return json.dumps({'success': False, 'error': f'JSON file not found: {json_file_path}'})
            floor_plan = dict_to_floor_plan(layout_data)
            current_layout = floor_plan
            mesh_info_dict = export_single_room_layout_to_mesh_dict_list(current_layout, room_id)
            stage = Usd.Stage.CreateInMemory()
            world_base_prim = UsdGeom.Xform.Define(stage, '/World')
            stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
            collision_approximation = 'sdf'
            self.track_ids = []
            door_ids = []
            door_frame_ids = []
            print(f'mesh_info_dict: {mesh_info_dict.keys()}')
            for mesh_id in mesh_info_dict:
                if mesh_id.startswith('wall_room_') or mesh_id.startswith('window_') or mesh_id.startswith('floor_'):
                    usd_internal_path = f'/World/{mesh_id}'
                elif mesh_id.startswith('door_'):
                    if mesh_id.endswith('_frame'):
                        door_frame_ids.append(mesh_id)
                    else:
                        door_ids.append(mesh_id)
                    continue
                else:
                    self.track_ids.append(mesh_id)
                    usd_internal_path = f'/World/{mesh_id}'
                mesh_dict = mesh_info_dict[mesh_id]
                mesh_obj_i = mesh_dict['mesh']
                static = mesh_dict['static']
                articulation = mesh_dict.get('articulation', None)
                texture = mesh_dict.get('texture', None)
                mass = mesh_dict.get('mass', 1.0)
                print(f'usd_internal_path: {usd_internal_path}')
                stage = convert_mesh_to_usd(stage, usd_internal_path, mesh_obj_i.vertices, mesh_obj_i.faces, collision_approximation, static, articulation, mass=mass, physics_iter=(16, 4), apply_debug_torque=False, debug_torque_value=30.0, texture=texture, usd_internal_art_reference_path=f'/World/{mesh_id}')
            door_ids = sorted(door_ids)
            door_frame_ids = sorted(door_frame_ids)
            for (door_id, door_frame_id) in zip(door_ids, door_frame_ids):
                usd_internal_path_door = f'/World/{door_id}'
                usd_internal_path_door_frame = f'/World/{door_frame_id}'
                mesh_dict_door = mesh_info_dict[door_id]
                mesh_obj_door = mesh_dict_door['mesh']
                articulation_door = mesh_dict_door.get('articulation', None)
                texture_door = mesh_dict_door.get('texture', None)
                mesh_dict_door_frame = mesh_info_dict[door_frame_id]
                mesh_obj_door_frame = mesh_dict_door_frame['mesh']
                texture_door_frame = mesh_dict_door_frame.get('texture', None)
                stage = door_frame_to_usd(stage, usd_internal_path_door, usd_internal_path_door_frame, mesh_obj_door, mesh_obj_door_frame, articulation_door, texture_door, texture_door_frame)
            cache = UsdUtils.StageCache.Get()
            stage_id = cache.Insert(stage).ToLongInt()
            omni.usd.get_context().attach_stage_with_callback(stage_id)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            return {'status': 'success', 'message': f'Room layout scene created successfully'}
        except Exception as e:
            print(f'Error creating room layout scene: {str(e)}')
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def create_single_room_layout_scene_from_room(self, scene_save_dir: str, room_dict_save_path: str):
        """
        Create a room layout scene from a dictionary of mesh information.
        """
        try:
            try:
                with open(room_dict_save_path, 'r') as f:
                    room_data = json.load(f)
            except FileNotFoundError:
                return json.dumps({'success': False, 'error': f'JSON file not found: {room_dict_save_path}'})
            layout_id = os.path.basename(scene_save_dir)
            room = dict_to_room(room_data)
            mesh_info_dict = export_single_room_layout_to_mesh_dict_list_from_room(room, layout_id)
            stage = Usd.Stage.CreateInMemory()
            world_base_prim = UsdGeom.Xform.Define(stage, '/World')
            stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
            collision_approximation = 'sdf'
            self.track_ids = []
            door_ids = []
            door_frame_ids = []
            for mesh_id in mesh_info_dict:
                if mesh_id.startswith('wall_room_') or mesh_id.startswith('window_') or mesh_id.startswith('floor_'):
                    usd_internal_path = f'/World/{mesh_id}'
                elif mesh_id.startswith('door_'):
                    if mesh_id.endswith('_frame'):
                        door_frame_ids.append(mesh_id)
                    else:
                        door_ids.append(mesh_id)
                    continue
                else:
                    self.track_ids.append(mesh_id)
                    usd_internal_path = f'/World/{mesh_id}'
                mesh_dict = mesh_info_dict[mesh_id]
                mesh_obj_i = mesh_dict['mesh']
                static = mesh_dict['static']
                articulation = mesh_dict.get('articulation', None)
                texture = mesh_dict.get('texture', None)
                mass = mesh_dict.get('mass', 1.0)
                stage = convert_mesh_to_usd(stage, usd_internal_path, mesh_obj_i.vertices, mesh_obj_i.faces, collision_approximation, static, articulation, mass=mass, physics_iter=(16, 1), apply_debug_torque=False, debug_torque_value=30.0, texture=texture, usd_internal_art_reference_path=f'/World/{mesh_id}')
            door_ids = sorted(door_ids)
            door_frame_ids = sorted(door_frame_ids)
            for (door_id, door_frame_id) in zip(door_ids, door_frame_ids):
                usd_internal_path_door = f'/World/{door_id}'
                usd_internal_path_door_frame = f'/World/{door_frame_id}'
                mesh_dict_door = mesh_info_dict[door_id]
                mesh_obj_door = mesh_dict_door['mesh']
                articulation_door = mesh_dict_door.get('articulation', None)
                texture_door = mesh_dict_door.get('texture', None)
                mesh_dict_door_frame = mesh_info_dict[door_frame_id]
                mesh_obj_door_frame = mesh_dict_door_frame['mesh']
                texture_door_frame = mesh_dict_door_frame.get('texture', None)
                stage = door_frame_to_usd(stage, usd_internal_path_door, usd_internal_path_door_frame, mesh_obj_door, mesh_obj_door_frame, articulation_door, texture_door, texture_door_frame)
            cache = UsdUtils.StageCache.Get()
            stage_id = cache.Insert(stage).ToLongInt()
            omni.usd.get_context().attach_stage_with_callback(stage_id)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            return {'status': 'success', 'message': f'Room layout scene created successfully'}
        except Exception as e:
            print(f'Error creating room layout scene: {str(e)}')
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def create_room_groups_layouts(self, scene_save_dir: str, room_list_save_path: str):
        """
        Create a room layout scene from a dictionary of mesh information.
        """
        try:
            layout_id = os.path.basename(scene_save_dir)
            with open(room_list_save_path, 'r') as f:
                room_dict_save_paths = json.load(f)
            mesh_info_dict_groups = {}
            group_offset = {}
            current_x_offset = 0.0
            gap = 1.0
            for (room_i, room_dict_save_path) in enumerate(room_dict_save_paths):
                with open(room_dict_save_path, 'r') as f:
                    room_data = json.load(f)
                room = dict_to_room(room_data)
                group_id = os.path.splitext(os.path.basename(room_dict_save_path))[0]
                mesh_info_dict = export_single_room_layout_to_mesh_dict_list_from_room(room, layout_id)
                mesh_info_dict_groups[group_id] = mesh_info_dict
                (room_corner_x, room_corner_y) = (room.position.x, room.position.y)
                (room_x_length, room_y_length) = (room.dimensions.width, room.dimensions.length)
                group_offset[group_id] = (current_x_offset - room_corner_x, 0.0 - room_corner_y, 0.0)
                current_x_offset += room_x_length + gap
            gc.collect()
            stage = Usd.Stage.CreateInMemory()
            world_base_prim = UsdGeom.Xform.Define(stage, '/World')
            stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
            collision_approximation = 'sdf'
            self.group_track_prims = {}
            self.group_ids = []
            for (group_id, mesh_info_dict) in mesh_info_dict_groups.items():
                group_base_prim = UsdGeom.Xform.Define(stage, f'/World/{group_id}')
                group_base_prim.AddTranslateOp().Set(value=group_offset[group_id])
                self.group_track_prims[group_id] = []
                self.group_ids.append(group_id)
                for mesh_id in mesh_info_dict:
                    if mesh_id.startswith('wall_room_') or mesh_id.startswith('window_') or mesh_id.startswith('floor_'):
                        usd_internal_path = f'/World/{group_id}/{mesh_id}'
                    elif mesh_id.startswith('door_'):
                        continue
                    else:
                        usd_internal_path = f'/World/{group_id}/{mesh_id}'
                        self.group_track_prims[group_id].append(usd_internal_path)
                    mesh_dict = mesh_info_dict[mesh_id]
                    mesh_obj_i = mesh_dict['mesh']
                    static = mesh_dict['static']
                    articulation = mesh_dict.get('articulation', None)
                    texture = mesh_dict.get('texture', None)
                    mass = mesh_dict.get('mass', 1.0)
                    stage = convert_mesh_to_usd(stage, usd_internal_path, mesh_obj_i.vertices, mesh_obj_i.faces, collision_approximation, static, articulation, mass=mass, physics_iter=(16, 4), apply_debug_torque=False, debug_torque_value=30.0, texture=texture, usd_internal_art_reference_path=f'/World/{group_id}/{mesh_id}')
            cache = UsdUtils.StageCache.Get()
            stage_id = cache.Insert(stage).ToLongInt()
            omni.usd.get_context().attach_stage_with_callback(stage_id)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            return {'status': 'success', 'message': f'Room layout scene created successfully'}
        except Exception as e:
            print(f'Error creating room layout scene: {str(e)}')
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    async def test_object_placements_batch_in_single_room_async(self, room_dict_save_path: str, placements_batch: List[Dict[str, Any]], parallel_width: int=16, max_candidates_per_object: int=10, simulation_steps: int=2000, equilibrium_steps: int=20, position_limit: float=0.2, rotation_limit: float=8.0, scheduling_policy: str='work_conserving'):
        """Evaluate every on-object candidate with one reusable cloned room build."""
        try:
            with open(room_dict_save_path, 'r') as room_file:
                room_data = json.load(room_file)
            from isaac_sim_mcp_extension.batched_placement import BatchedPlacementSimulator
            simulator = BatchedPlacementSimulator(room_data=room_data, scene_save_dir=os.path.dirname(room_dict_save_path), items=placements_batch, parallel_width=parallel_width, max_candidates_per_object=max_candidates_per_object, simulation_steps=simulation_steps, equilibrium_steps=equilibrium_steps, position_limit=position_limit, rotation_limit=rotation_limit, scheduling_policy=scheduling_policy)
            return await simulator.run_async()
        except Exception as error:
            print(f'Error in batched placement simulation: {error}')
            traceback.print_exc()
            return {'status': 'error', 'message': str(error)}

    def test_object_placements_in_single_room(self, room_dict_save_path: str, placements_info_path: str, only_need_one: bool=False):
        """
        Create a room layout scene from a dictionary of mesh information.
        """
        try:
            print(f'room_dict_save_path: {room_dict_save_path}, placements_info_path: {placements_info_path}')
            try:
                with open(room_dict_save_path, 'r') as f:
                    room_data = json.load(f)
            except FileNotFoundError:
                return json.dumps({'success': False, 'error': f'JSON file not found: {room_dict_save_path}'})
            target_room = dict_to_room(room_data)
            scene_save_dir = os.path.dirname(room_dict_save_path)
            layout_id = os.path.basename(scene_save_dir)
            mesh_info_dict = export_single_room_layout_to_mesh_dict_list_from_room(target_room, layout_id)
            stage = Usd.Stage.CreateInMemory()
            world_base_prim = UsdGeom.Xform.Define(stage, '/World')
            stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
            collision_approximation = 'sdf'
            track_ids = []
            door_ids = []
            door_frame_ids = []
            for mesh_id in mesh_info_dict:
                if mesh_id.startswith('wall_room_') or mesh_id.startswith('window_') or mesh_id.startswith('floor_'):
                    usd_internal_path = f'/World/{mesh_id}'
                elif mesh_id.startswith('door_'):
                    if mesh_id.endswith('_frame'):
                        door_frame_ids.append(mesh_id)
                    else:
                        door_ids.append(mesh_id)
                    continue
                else:
                    track_ids.append(mesh_id)
                    usd_internal_path = f'/World/{mesh_id}'
                mesh_dict = mesh_info_dict[mesh_id]
                mesh_obj_i = mesh_dict['mesh']
                static = mesh_dict['static']
                articulation = mesh_dict.get('articulation', None)
                texture = mesh_dict.get('texture', None)
                mass = mesh_dict.get('mass', 1.0)
                stage = convert_mesh_to_usd(stage, usd_internal_path, mesh_obj_i.vertices, mesh_obj_i.faces, collision_approximation, static, articulation, mass=mass, physics_iter=(16, 4), apply_debug_torque=False, debug_torque_value=30.0, texture=texture, usd_internal_art_reference_path=f'/World/{mesh_id}')
            door_ids = sorted(door_ids)
            door_frame_ids = sorted(door_frame_ids)
            for (door_id, door_frame_id) in zip(door_ids, door_frame_ids):
                usd_internal_path_door = f'/World/{door_id}'
                usd_internal_path_door_frame = f'/World/{door_frame_id}'
                mesh_dict_door = mesh_info_dict[door_id]
                mesh_obj_door = mesh_dict_door['mesh']
                articulation_door = mesh_dict_door.get('articulation', None)
                texture_door = mesh_dict_door.get('texture', None)
                mesh_dict_door_frame = mesh_info_dict[door_frame_id]
                mesh_obj_door_frame = mesh_dict_door_frame['mesh']
                texture_door_frame = mesh_dict_door_frame.get('texture', None)
                stage = door_frame_to_usd(stage, usd_internal_path_door, usd_internal_path_door_frame, mesh_obj_door, mesh_obj_door_frame, articulation_door, texture_door, texture_door_frame)
            with open(placements_info_path, 'r') as f:
                placements_info = json.load(f)
            placements = placements_info['placements']
            object_info = placements_info['object']
            placed_object_mass = object_info.get('mass', 1.0)
            mesh_info_dict = get_single_object_mesh_info_dict(scene_save_dir, object_info['source'], object_info['source_id'])
            if mesh_info_dict is None:
                return {'status': 'error', 'message': f"Object mesh not found: {object_info['source']}/{object_info['source_id']}"}
            mesh_id = 'object_to_place'
            articulation = None
            static = False
            texture = mesh_info_dict['texture']
            track_ids.append(mesh_id)
            object_to_place_prim_path = f'/World/{mesh_id}'
            safe_placements = []
            max_attempts = 10
            if len(placements) > max_attempts:
                placements = placements[:max_attempts]
                print(f'Warning: Only testing the first {max_attempts} placements due to time constraints')
            for placement in tqdm(placements):
                translation_matrix_initial = np.eye(4)
                translation_matrix_initial[:3, 3] = np.array([placement['position']['x'], placement['position']['y'], placement['position']['z']])
                rotation_matrix_initial = np.eye(4)
                rotation_matrix_initial[:3, :3] = R.from_euler('xyz', [placement['rotation']['x'], placement['rotation']['y'], placement['rotation']['z']], degrees=False).as_matrix()
                transform_matrix_initial = translation_matrix_initial @ rotation_matrix_initial
                transformed_mesh = apply_object_transform_direct(mesh_info_dict['mesh'], placement['position'], placement['rotation'], degrees=False)
                stage = convert_mesh_to_usd(stage, object_to_place_prim_path, transformed_mesh.vertices, transformed_mesh.faces, collision_approximation, static, articulation, mass=placed_object_mass, physics_iter=(16, 4), apply_debug_torque=False, debug_torque_value=30.0, texture=texture, usd_internal_art_reference_path=object_to_place_prim_path)
                cache = UsdUtils.StageCache.Get()
                stage_id = cache.Insert(stage).ToLongInt()
                omni.usd.get_context().attach_stage_with_callback(stage_id)
                UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
                print(f'start first simulation test')
                (prims, prim_paths) = get_all_prims_with_paths(track_ids)
                early_stop_unstable_exemption_prim_paths = [object_to_place_prim_path]
                traced_data_all = start_simulation_and_track(prims, prim_paths, early_stop_unstable_exemption_prim_paths=early_stop_unstable_exemption_prim_paths)
                unstable_addition = False
                for (prim_path, traced_data) in traced_data_all.items():
                    if prim_path not in early_stop_unstable_exemption_prim_paths and (not traced_data['stable']):
                        unstable_addition = True
                        print(f'unstable_addition in first simulation test: {prim_path}')
                        break
                traced_data_object_to_place = traced_data_all[object_to_place_prim_path]
                position_simulated_relative = traced_data_object_to_place['final_position']
                orientation_simulated_relative = traced_data_object_to_place['final_orientation']
                rotation_matrix_simulated_relative = np.eye(4)
                rotation_matrix_simulated_relative[:3, :3] = R.from_quat(orientation_simulated_relative, scalar_first=True).as_matrix()
                translation_matrix_simulated_relative = np.eye(4)
                translation_matrix_simulated_relative[:3, 3] = position_simulated_relative
                transform_matrix_simulated_relative = translation_matrix_simulated_relative @ rotation_matrix_simulated_relative
                transform_matrix_total = transform_matrix_simulated_relative @ transform_matrix_initial
                position_simulated = transform_matrix_total[:3, 3]
                rotation_simulated = R.from_matrix(transform_matrix_total[:3, :3]).as_euler('xyz', degrees=False)
                transformed_mesh = apply_object_transform_direct(mesh_info_dict['mesh'], {'x': position_simulated[0], 'y': position_simulated[1], 'z': position_simulated[2]}, {'x': rotation_simulated[0], 'y': rotation_simulated[1], 'z': rotation_simulated[2]}, degrees=False)
                mesh_center_z = (transformed_mesh.vertices[:, 2].max() + transformed_mesh.vertices[:, 2].min()) / 2
                if mesh_center_z < placement['position']['z'] - 0.01:
                    print(f"unstable_addition: {mesh_center_z} < {placement['position']['z'] - 0.01}")
                    unstable_addition = True
                if not unstable_addition and traced_data_object_to_place['stable']:
                    safe_placements.append({'position': {'x': float(position_simulated[0]), 'y': float(position_simulated[1]), 'z': float(position_simulated[2])}, 'rotation': {'x': float(rotation_simulated[0]), 'y': float(rotation_simulated[1]), 'z': float(rotation_simulated[2])}})
                    if only_need_one:
                        break
                    continue
                if not unstable_addition:
                    stage = convert_mesh_to_usd(stage, object_to_place_prim_path, transformed_mesh.vertices, transformed_mesh.faces, collision_approximation, static, articulation, mass=placed_object_mass, physics_iter=(16, 4), apply_debug_torque=False, debug_torque_value=30.0, texture=texture, usd_internal_art_reference_path=object_to_place_prim_path)
                    cache = UsdUtils.StageCache.Get()
                    stage_id = cache.Insert(stage).ToLongInt()
                    omni.usd.get_context().attach_stage_with_callback(stage_id)
                    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
                    (prims, prim_paths) = get_all_prims_with_paths(track_ids)
                    print(f'start second simulation test')
                    traced_data_all = start_simulation_and_track(prims, prim_paths)
                    all_stable = True
                    for (prim_path, traced_data) in traced_data_all.items():
                        if not traced_data['stable']:
                            print(f'unstable_addition in second simulation test: {prim_path}')
                            all_stable = False
                            break
                    if all_stable:
                        safe_placements.append({'position': {'x': float(position_simulated[0]), 'y': float(position_simulated[1]), 'z': float(position_simulated[2])}, 'rotation': {'x': float(rotation_simulated[0]), 'y': float(rotation_simulated[1]), 'z': float(rotation_simulated[2])}})
                        if only_need_one:
                            break
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                save_path = f.name
                json.dump(safe_placements, f, indent=4)
            return {'status': 'success', 'message': 'Safe placements saved successfully', 'safe_placements_path': save_path}
        except Exception as e:
            print(f'Error creating room layout scene: {str(e)}')
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def get_room_layout_scene_usd(self, scene_save_dir: str, usd_file_path: str):
        """
        Create a room layout scene from a dictionary of mesh information.
        """
        try:
            current_layout_id = os.path.basename(scene_save_dir)
            json_file_path = os.path.join(scene_save_dir, f'{current_layout_id}.json')
            try:
                with open(json_file_path, 'r') as f:
                    layout_data = json.load(f)
            except FileNotFoundError:
                return json.dumps({'success': False, 'error': f'JSON file not found: {json_file_path}'})
            floor_plan = dict_to_floor_plan(layout_data)
            current_layout = floor_plan
            mesh_info_dict = export_layout_to_mesh_dict_list(current_layout)
            if os.path.exists(usd_file_path):
                os.remove(usd_file_path)
            stage = Usd.Stage.CreateNew(usd_file_path)
            collision_approximation = 'sdf'
            world_base_prim = UsdGeom.Xform.Define(stage, '/World')
            stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
            door_ids = []
            door_frame_ids = []
            for mesh_id in mesh_info_dict:
                if mesh_id.startswith('door_'):
                    if mesh_id.endswith('_frame'):
                        door_frame_ids.append(mesh_id)
                    else:
                        door_ids.append(mesh_id)
                    continue
                else:
                    usd_internal_path = f'/World/{mesh_id}'
                mesh_dict = mesh_info_dict[mesh_id]
                mesh_obj_i = mesh_dict['mesh']
                static = mesh_dict['static']
                articulation = mesh_dict.get('articulation', None)
                texture = mesh_dict.get('texture', None)
                mass = mesh_dict.get('mass', 1.0)
                stage = convert_mesh_to_usd(stage, usd_internal_path, mesh_obj_i.vertices, mesh_obj_i.faces, collision_approximation, static, articulation, physics_iter=(16, 4), apply_debug_torque=False, debug_torque_value=30.0, texture=texture, usd_internal_art_reference_path=f'/World/{mesh_id}')
            door_ids = sorted(door_ids)
            door_frame_ids = sorted(door_frame_ids)
            for (door_id, door_frame_id) in zip(door_ids, door_frame_ids):
                usd_internal_path_door = f'/World/{door_id}'
                usd_internal_path_door_frame = f'/World/{door_frame_id}'
                mesh_dict_door = mesh_info_dict[door_id]
                mesh_obj_door = mesh_dict_door['mesh']
                articulation_door = mesh_dict_door.get('articulation', None)
                texture_door = mesh_dict_door.get('texture', None)
                mesh_dict_door_frame = mesh_info_dict[door_frame_id]
                mesh_obj_door_frame = mesh_dict_door_frame['mesh']
                texture_door_frame = mesh_dict_door_frame.get('texture', None)
                stage = door_frame_to_usd(stage, usd_internal_path_door, usd_internal_path_door_frame, mesh_obj_door, mesh_obj_door_frame, articulation_door, texture_door, texture_door_frame)
            stage.Save()
            success = UsdUtils.CreateNewUsdzPackage(f'{usd_file_path}', usd_file_path.replace('.usd', '.usdz'))
            if success:
                print(f"Successfully created USDZ: {usd_file_path.replace('.usd', '.usdz')}")
            else:
                print('Failed to create USDZ.')
            return {'status': 'success', 'message': f'Room layout scene created successfully'}
        except Exception as e:
            print(f'Error creating room layout scene: {str(e)}')
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def save_usd_with_ids(self, usd_file_path, mesh_info_dict, room_base_ids):
        stage = Usd.Stage.CreateNew(usd_file_path)
        collision_approximation = 'sdf'
        world_base_prim = UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
        for mesh_id in room_base_ids:
            if mesh_id.startswith('door_'):
                continue
            else:
                usd_internal_path = f'/World/{mesh_id}'
            mesh_dict = mesh_info_dict[mesh_id]
            mesh_obj_i = mesh_dict['mesh']
            static = mesh_dict['static']
            articulation = mesh_dict.get('articulation', None)
            texture = mesh_dict.get('texture', None)
            mass = mesh_dict.get('mass', 1.0)
            stage = convert_mesh_to_usd(stage, usd_internal_path, mesh_obj_i.vertices, mesh_obj_i.faces, collision_approximation, static, articulation, mass=mass, physics_iter=(16, 4), apply_debug_torque=False, debug_torque_value=30.0, texture=texture, usd_internal_art_reference_path=f'/World/{mesh_id}', add_damping=True)
        stage.Save()
        success = UsdUtils.CreateNewUsdzPackage(f'{usd_file_path}', usd_file_path.replace('.usd', '.usdz'))
        if success:
            print(f"Successfully created USDZ: {usd_file_path.replace('.usd', '.usdz')}")
        else:
            print('Failed to create USDZ.')

    def save_door_frame_to_usd(self, usd_file_path, mesh_info_dict_door, mesh_info_dict_door_frame, door_id, door_frame_id):
        stage = Usd.Stage.CreateNew(usd_file_path)
        world_base_prim = UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath('/World'))
        usd_internal_path_door = f'/World/{door_id}'
        usd_internal_path_door_frame = f'/World/{door_frame_id}'
        mesh_dict_door = mesh_info_dict_door
        mesh_obj_door = mesh_dict_door['mesh']
        articulation_door = mesh_dict_door.get('articulation', None)
        texture_door = mesh_dict_door.get('texture', None)
        mesh_dict_door_frame = mesh_info_dict_door_frame
        mesh_obj_door_frame = mesh_dict_door_frame['mesh']
        texture_door_frame = mesh_dict_door_frame.get('texture', None)
        stage = door_frame_to_usd(stage, usd_internal_path_door, usd_internal_path_door_frame, mesh_obj_door, mesh_obj_door_frame, articulation_door, texture_door, texture_door_frame)
        stage.Save()
        success = UsdUtils.CreateNewUsdzPackage(f'{usd_file_path}', usd_file_path.replace('.usd', '.usdz'))
        if success:
            print(f"Successfully created USDZ: {usd_file_path.replace('.usd', '.usdz')}")
        else:
            print('Failed to create USDZ.')

    def get_room_layout_scene_usd_separate(self, scene_save_dir: str, usd_collection_dir: str):
        """
        Create a room layout scene from a dictionary of mesh information.
        """
        try:
            current_layout_id = os.path.basename(scene_save_dir)
            json_file_path = os.path.join(scene_save_dir, f'{current_layout_id}.json')
            try:
                with open(json_file_path, 'r') as f:
                    layout_data = json.load(f)
            except FileNotFoundError:
                return json.dumps({'success': False, 'error': f'JSON file not found: {json_file_path}'})
            floor_plan = dict_to_floor_plan(layout_data)
            current_layout = floor_plan
            mesh_info_dict = export_layout_to_mesh_dict_list(current_layout)
            rigid_object_property_dict = {}
            rigid_object_transform_dict = {}
            os.makedirs(usd_collection_dir, exist_ok=True)
            room_base_ids = [mesh_id for mesh_id in mesh_info_dict.keys() if mesh_id.startswith('door_') or mesh_id.startswith('wall_room_') or mesh_id.startswith('window_') or mesh_id.startswith('floor_')]
            rigid_object_ids = [mesh_id for mesh_id in mesh_info_dict.keys() if mesh_id not in room_base_ids]
            door_ids = []
            door_frame_ids = []
            for room_base_id in room_base_ids:
                if room_base_id.startswith('door_'):
                    if room_base_id.endswith('_frame'):
                        door_frame_ids.append(room_base_id)
                    else:
                        door_ids.append(room_base_id)
                    continue
                usd_file_path = f'{usd_collection_dir}/{room_base_id}.usd'
                self.save_usd_with_ids(usd_file_path, mesh_info_dict, [room_base_id])
            for rigid_object_id in rigid_object_ids:
                usd_file_path = f'{usd_collection_dir}/{rigid_object_id}.usd'
                self.save_usd_with_ids(usd_file_path, mesh_info_dict, [rigid_object_id])
                rigid_object_property_dict[rigid_object_id] = {'static': mesh_info_dict[rigid_object_id]['static'], 'mass': mesh_info_dict[rigid_object_id]['mass']}
                rigid_object_transform_dict[rigid_object_id] = mesh_info_dict[rigid_object_id]['transform']
            for (door_id, door_frame_id) in zip(door_ids, door_frame_ids):
                self.save_door_frame_to_usd(usd_file_path=f'{usd_collection_dir}/{door_id}.usd', mesh_info_dict_door=mesh_info_dict[door_id], mesh_info_dict_door_frame=mesh_info_dict[door_frame_id], door_id=door_id, door_frame_id=door_frame_id)
            with open(os.path.join(usd_collection_dir, 'rigid_object_property_dict.json'), 'w') as f:
                json.dump(rigid_object_property_dict, f, indent=4)
            with open(os.path.join(usd_collection_dir, 'rigid_object_transform_dict.json'), 'w') as f:
                json.dump(rigid_object_transform_dict, f, indent=4)
            return {'status': 'success', 'message': f'Room layout scene created successfully'}
        except Exception as e:
            print(f'Error creating room layout scene: {str(e)}')
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def get_room_layout_scene_usd_separate_from_layout(self, layout_json_path: str, usd_collection_dir: str):
        """
        Create a room layout scene from a dictionary of mesh information.
        """
        try:
            try:
                with open(layout_json_path, 'r') as f:
                    layout_data = json.load(f)
            except FileNotFoundError:
                return json.dumps({'success': False, 'error': f'JSON file not found: {layout_json_path}'})
            floor_plan = dict_to_floor_plan(layout_data)
            current_layout = floor_plan
            mesh_info_dict = export_layout_to_mesh_dict_list_no_object_transform(current_layout)
            rigid_object_property_dict = {}
            rigid_object_transform_dict = {}
            os.makedirs(usd_collection_dir, exist_ok=True)
            room_base_ids = [mesh_id for mesh_id in mesh_info_dict.keys() if mesh_id.startswith('door_') or mesh_id.startswith('wall_room_') or mesh_id.startswith('window_') or mesh_id.startswith('floor_')]
            rigid_object_ids = [mesh_id for mesh_id in mesh_info_dict.keys() if mesh_id not in room_base_ids]
            door_ids = []
            door_frame_ids = []
            for room_base_id in room_base_ids:
                if room_base_id.startswith('door_'):
                    if room_base_id.endswith('_frame'):
                        door_frame_ids.append(room_base_id)
                    else:
                        door_ids.append(room_base_id)
                    continue
                usd_file_path = f'{usd_collection_dir}/{room_base_id}.usd'
                self.save_usd_with_ids(usd_file_path, mesh_info_dict, [room_base_id])
            for rigid_object_id in rigid_object_ids:
                usd_file_path = f'{usd_collection_dir}/{rigid_object_id}.usd'
                rigid_object_property_dict[rigid_object_id] = {'static': mesh_info_dict[rigid_object_id]['static'], 'mass': mesh_info_dict[rigid_object_id]['mass']}
                rigid_object_transform_dict[rigid_object_id] = mesh_info_dict[rigid_object_id]['transform']
                mesh_info_dict[rigid_object_id]['static'] = False
                self.save_usd_with_ids(usd_file_path, mesh_info_dict, [rigid_object_id])
            for (door_id, door_frame_id) in zip(door_ids, door_frame_ids):
                self.save_door_frame_to_usd(usd_file_path=f'{usd_collection_dir}/{door_id}.usd', mesh_info_dict_door=mesh_info_dict[door_id], mesh_info_dict_door_frame=mesh_info_dict[door_frame_id], door_id=door_id, door_frame_id=door_frame_id)
            with open(os.path.join(usd_collection_dir, 'rigid_object_property_dict.json'), 'w') as f:
                json.dump(rigid_object_property_dict, f, indent=4)
            with open(os.path.join(usd_collection_dir, 'rigid_object_transform_dict.json'), 'w') as f:
                json.dump(rigid_object_transform_dict, f, indent=4)
            return {'status': 'success', 'message': f'Room layout scene created successfully'}
        except Exception as e:
            print(f'Error creating room layout scene: {str(e)}')
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def validate_room_stability(self, scene_save_dir: str, room_id: str):
        """Load and simulate one room atomically without editing its layout JSON."""
        created = self.create_single_room_layout_scene(scene_save_dir, room_id)
        if not isinstance(created, dict) or created.get('status') != 'success':
            return {'status': 'error', 'message': str(created)}
        result = self.simulate_the_scene()
        if result is None:
            return {'status': 'error', 'message': 'Isaac simulation returned no result'}
        stage = omni.usd.get_context().get_stage()
        loaded_ids = []
        for object_id in self.track_ids:
            prim = stage.GetPrimAtPath(f'/World/{object_id}')
            if prim.IsValid() and any((child.IsA(UsdGeom.Mesh) and len(UsdGeom.Mesh(child).GetPointsAttr().Get() or []) > 0 for child in Usd.PrimRange(prim))):
                loaded_ids.append(object_id)
        result['loaded_object_ids'] = loaded_ids
        result['missing_object_ids'] = sorted(set(self.track_ids) - set(loaded_ids))
        result['validation_settings'] = {'max_steps': 2000, 'equilibrium_steps': 20, 'position_threshold': 0.2, 'rotation_threshold_degrees': 8.0}
        return result

    def simulate_the_scene(self):
        """
        Simulate the scene.
        """
        try:
            stage = omni.usd.get_context().get_stage()
            (prims, prim_paths) = get_all_prims_with_paths(self.track_ids)
            traced_data_all = start_simulation_and_track(prims, prim_paths)
            unstable_prims = []
            unstable_object_ids = []
            for (object_id, (prim_path, traced_data)) in zip(self.track_ids, traced_data_all.items()):
                if not traced_data['stable']:
                    unstable_prims.append(os.path.basename(prim_path))
                    unstable_object_ids.append(object_id)
            if len(unstable_prims) > 0:
                next_step_message = f"\nThe scene is unstable. Please check the following prims: {unstable_prims}; \nSuggestions: \n1. use move_one_object_with_condition_in_room(room_id, condition) to adjust the placement of those unstable objects.\n2. use place_objects_in_room(room_id, 'remove [object_description]') to remove the unstable objects.\n"
            else:
                next_step_message = 'The scene is stable. You can continue to the next step.'
            return {'status': 'success', 'message': 'Scene simulated successfully!', 'unstable_objects': unstable_object_ids, 'next_step': next_step_message}
        except Exception as e:
            print(f'Error simulating the scene: {str(e)}')

    def simulate_the_scene_groups(self):
        """
        Simulate the scene.
        """
        try:
            stage = omni.usd.get_context().get_stage()
            group_prims = {}
            group_prims_paths = {}
            for group_id in self.group_ids:
                prims = get_all_prims_with_prim_paths(self.group_track_prims[group_id])
                prim_paths = self.group_track_prims[group_id]
                group_prims[group_id] = prims
                group_prims_paths[group_id] = prim_paths
            group_traced_data_all = start_simulation_and_track_groups(group_prims, group_prims_paths)
            group_stable_list = []
            for group_id in self.group_ids:
                stable = True
                traced_data_all = group_traced_data_all[group_id]
                for (prim_path, traced_data) in traced_data_all.items():
                    if not traced_data['stable']:
                        stable = False
                        break
                group_stable_list.append(stable)
            return {'status': 'success', 'message': 'Scene simulated successfully!', 'group_stable_list': json.dumps(group_stable_list)}
        except Exception as e:
            print(f'Error simulating the scene: {str(e)}')
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def render_room(self, room_id: str):
        try:
            stage = omni.usd.get_context().get_stage()
        except Exception as e:
            print(f'Error rendering the room: {str(e)}')
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def transform(self, prim_path, position=(0, 0, 50), scale=(10, 10, 10)):
        """
        Transform a USD model by applying position and scale.
        
        Args:
            prim_path (str): Path to the USD prim to transform
            position (tuple, optional): The position to set (x, y, z)
            scale (tuple, optional): The scale to set (x, y, z)
            
        Returns:
            dict: Result information
        """
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim:
                return {'status': 'error', 'message': f'Prim not found at path: {prim_path}'}
            loader = USDLoader()
            xformable = loader.transform(prim=prim, position=position, scale=scale)
            return {'status': 'success', 'message': f'Model at {prim_path} transformed successfully', 'position': position, 'scale': scale}
        except Exception as e:
            print(f'Error transforming model: {str(e)}')
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
