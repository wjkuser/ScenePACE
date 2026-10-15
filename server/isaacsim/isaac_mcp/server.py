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
import time
from mcp.server.fastmcp import FastMCP, Context, Image
import socket
import json
import asyncio
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List
import os
from pathlib import Path
import base64
from urllib.parse import urlparse
import sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('IsaacMCPServer')

@dataclass
class IsaacConnection:
    host: str
    port: int
    sock: socket.socket = None

    def connect(self) -> bool:
        """Connect to the Isaac addon socket server"""
        if self.sock:
            return True
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f'Connected to Isaac at {self.host}:{self.port}')
            return True
        except Exception as e:
            logger.error(f'Failed to connect to Isaac: {str(e)}')
            self.sock = None
            return False

    def disconnect(self):
        """Disconnect from the Isaac addon"""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f'Error disconnecting from Isaac: {str(e)}')
            finally:
                self.sock = None

    def receive_full_response(self, sock, buffer_size=16384):
        """Receive the complete response, potentially in multiple chunks"""
        chunks = []
        sock.settimeout(300.0)
        try:
            while True:
                try:
                    logger.info('Waiting for data from Isaac')
                    time.sleep(1.0)
                    chunk = sock.recv(buffer_size)
                    if not chunk:
                        if not chunks:
                            raise Exception('Connection closed before receiving any data')
                        break
                    chunks.append(chunk)
                    try:
                        data = b''.join(chunks)
                        json.loads(data.decode('utf-8'))
                        logger.info(f'Received complete response ({len(data)} bytes)')
                        return data
                    except json.JSONDecodeError:
                        continue
                except socket.timeout:
                    logger.warning('Socket timeout during chunked receive')
                    break
                except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                    logger.error(f'Socket connection error during receive: {str(e)}')
                    raise
        except socket.timeout:
            logger.warning('Socket timeout during chunked receive')
        except Exception as e:
            logger.error(f'Error during receive: {str(e)}')
            raise
        if chunks:
            data = b''.join(chunks)
            logger.info(f'Returning data after receive completion ({len(data)} bytes)')
            try:
                json.loads(data.decode('utf-8'))
                return data
            except json.JSONDecodeError:
                raise Exception('Incomplete JSON response received')
        else:
            raise Exception('No data received')

    def send_command(self, command_type: str, params: Dict[str, Any]=None) -> Dict[str, Any]:
        """Send a command to Isaac and return the response"""
        if not self.sock and (not self.connect()):
            raise ConnectionError('Not connected to Isaac')
        command = {'type': command_type, 'params': params or {}}
        try:
            logger.info(f'Sending command: {command_type} with params: {params}')
            self.sock.sendall(json.dumps(command).encode('utf-8'))
            logger.info(f'Command sent, waiting for response...')
            self.sock.settimeout(300.0)
            response_data = self.receive_full_response(self.sock)
            logger.info(f'Received {len(response_data)} bytes of data')
            response = json.loads(response_data.decode('utf-8'))
            logger.info(f"Response parsed, status: {response.get('status', 'unknown')}")
            if response.get('status') == 'error':
                logger.error(f"Isaac error: {response.get('message')}")
                raise Exception(response.get('message', 'Unknown error from Isaac'))
            return response.get('result', {})
        except socket.timeout:
            logger.error('Socket timeout while waiting for response from Isaac')
            self.sock = None
            raise Exception('Timeout waiting for Isaac response - try simplifying your request')
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            logger.error(f'Socket connection error: {str(e)}')
            self.sock = None
            raise Exception(f'Connection to Isaac lost: {str(e)}')
        except json.JSONDecodeError as e:
            logger.error(f'Invalid JSON response from Isaac: {str(e)}')
            if 'response_data' in locals() and response_data:
                logger.error(f'Raw response (first 200 bytes): {response_data[:200]}')
            raise Exception(f'Invalid response from Isaac: {str(e)}')
        except Exception as e:
            logger.error(f'Error communicating with Isaac: {str(e)}')
            self.sock = None
            raise Exception(f'Communication error with Isaac: {str(e)}')
_isaac_connection = None
import hashlib

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

def get_port():
    configured_port = os.environ.get('ISAAC_MCP_PORT')
    if configured_port:
        port = int(configured_port)
        print(f'Isaacsim MCP server port: {port} (ISAAC_MCP_PORT)', file=sys.stderr)
        return port
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    port = slurm_job_id_to_port(slurm_job_id)
    print(f'Isaacsim MCP server port: {port}', file=sys.stderr)
    return port

def get_isaac_connection():
    """Get or create a persistent Isaac connection"""
    global _isaac_connection
    if _isaac_connection is not None:
        try:
            return _isaac_connection
        except Exception as e:
            logger.warning(f'Existing connection is no longer valid: {str(e)}')
            try:
                _isaac_connection.disconnect()
            except:
                pass
            _isaac_connection = None
    if _isaac_connection is None:
        _isaac_connection = IsaacConnection(host='localhost', port=get_port())
        if not _isaac_connection.connect():
            logger.error('Failed to connect to Isaac')
            _isaac_connection = None
            raise Exception('Could not connect to Isaac. Make sure the Isaac addon is running.')
        logger.info('Created new persistent connection to Isaac')
    return _isaac_connection

def get_scene_info() -> str:
    """Ping status of Isaac Sim Extension Server"""
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command('get_scene_info')
        print('result: ', result)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f'Error getting scene info from Isaac: {str(e)}')
        return {'status': 'error', 'error': str(e), 'message': 'Error getting scene info'}

def create_physics_scene(objects: List[Dict[str, Any]]=[], floor: bool=True, gravity: List[float]=[0, -0.981, 0], scene_name: str='physics_scene') -> Dict[str, Any]:
    """Create a physics scene with multiple objects. Before create physics scene, you need to call get_scene_info() first to verify availability of connection.
    
    Args:
        objects: List of objects to create. Each object should have at least 'type' and 'position'. 
        objects  = [
        {"path": "/World/Cube", "type": "Cube", "size": 20, "position": [0, 100, 0]},
        {"path": "/World/Sphere", "type": "Sphere", "radius": 5, "position": [5, 200, 0]},
        {"path": "/World/Cone", "type": "Cone", "height": 8, "radius": 3, "position": [-5, 150, 0]}
         ]
        floor: Whether to create a floor. deafult is True
        gravity: The gravity vector. Default is [0, 0, -981.0] (cm/s^2).
        scene_name: The name of the scene. deafult is "physics_scene"
        
    Returns:
        Dictionary with result information.
    """
    params = {'objects': objects, 'floor': floor}
    if gravity is not None:
        params['gravity'] = gravity
    if scene_name is not None:
        params['scene_name'] = scene_name
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command('create_physics_scene', params)
        return f"create_physics_scene successfully: {result.get('result', '')}, {result.get('message', '')}"
    except Exception as e:
        logger.error(f'Error create_physics_scene: {str(e)}')
        return {'status': 'error', 'error': str(e), 'message': 'Error create_physics_scene'}

def omni_kit_command(command: str='CreatePrim', prim_type: str='Sphere') -> str:
    """Execute an Omni Kit command.
    
    Args:
        command: The Omni Kit command to execute.
        prim_type: The primitive type for the command.
        
    Returns:
        String with result information.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command('omini_kit_command', {'command': command, 'prim_type': prim_type})
        return f"Omni Kit command executed successfully: {result.get('message', '')}"
    except Exception as e:
        logger.error(f'Error executing Omni Kit command: {str(e)}')
        return {'status': 'error', 'error': str(e), 'message': 'Error executing Omni Kit command'}

def execute_script(code: str) -> str:
    """Execute a scene inspection script in Isaac."""
    try:
        isaac = get_isaac_connection()
        print('code: ', code)
        result = isaac.send_command('execute_script', {'code': code})
        print('result: ', result)
        return result
    except Exception as e:
        logger.error(f'Error executing code: {str(e)}')
        return {'status': 'error', 'error': str(e), 'message': 'Error executing code'}

def create_room_layout_scene(scene_save_dir: str) -> str:
    """
    Create a room layout scene from a dictionary of mesh information.

    Args:
        scene_save_dir: The directory to save the scene layout.

    Returns:
        String with result information.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command('create_room_layout_scene', {'scene_save_dir': scene_save_dir})
        return result
    except Exception as e:
        logger.error(f'Error create_room_layout_scene: {str(e)}')
        return f"""error: {'status': "error", "error": str(e), "message": "Error create_room_layout_scene"}"""

def create_single_room_layout_scene(scene_save_dir: str, room_id: str) -> str:
    """
    Create a single room layout scene from a dictionary of mesh information.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command('create_single_room_layout_scene', {'scene_save_dir': scene_save_dir, 'room_id': room_id})
        return result
    except Exception as e:
        logger.error(f'Error create_single_room_layout_scene: {str(e)}')
        return {'status': 'error', 'error': str(e), 'message': 'Error create_single_room_layout_scene'}

def create_single_room_layout_scene_from_room(scene_save_dir: str, room_dict_save_path: str) -> str:
    """
    Create a single room layout scene from a dictionary of mesh information.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command('create_single_room_layout_scene_from_room', {'scene_save_dir': scene_save_dir, 'room_dict_save_path': room_dict_save_path})
        return result
    except Exception as e:
        logger.error(f'Error create_single_room_layout_scene: {str(e)}')
        return {'status': 'error', 'error': str(e), 'message': 'Error create_single_room_layout_scene'}

def create_room_groups_layouts(scene_save_dir: str, room_list_save_path: str) -> str:
    """
    Create a single room layout scene from a dictionary of mesh information.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command('create_room_groups_layouts', {'scene_save_dir': scene_save_dir, 'room_list_save_path': room_list_save_path})
        return result
    except Exception as e:
        logger.error(f'Error create_room_groups_layouts: {str(e)}')
        return {'status': 'error', 'error': str(e), 'message': 'Error create_room_groups_layouts'}

def get_room_layout_scene_usd(scene_save_dir: str, usd_file_path: str) -> str:
    """
    Get the USD file of the room layout scene.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command('get_room_layout_scene_usd', {'scene_save_dir': scene_save_dir, 'usd_file_path': usd_file_path})
        return result
    except Exception as e:
        logger.error(f'Error get_room_layout_scene_usd: {str(e)}')
        return f"""error: {'status': "error", "error": str(e), "message": "Error get_room_layout_scene_usd"}"""

def get_room_layout_scene_usd_separate(scene_save_dir: str, usd_collection_dir: str) -> str:
    """
    Get the USD file of the room layout scene.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command('get_room_layout_scene_usd_separate', {'scene_save_dir': scene_save_dir, 'usd_collection_dir': usd_collection_dir})
        return result
    except Exception as e:
        logger.error(f'Error get_room_layout_scene_usd_separate: {str(e)}')
        return f"""error: {'status': "error", "error": str(e), "message": "Error get_room_layout_scene_usd_separate"}"""

def get_room_layout_scene_usd_separate_from_layout(layout_json_path: str, usd_collection_dir: str) -> str:
    """
    Get the USD file of the room layout scene.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command('get_room_layout_scene_usd_separate_from_layout', {'layout_json_path': layout_json_path, 'usd_collection_dir': usd_collection_dir})
        return result
    except Exception as e:
        logger.error(f'Error get_room_layout_scene_usd_separate: {str(e)}')
        return f"""error: {'status': "error", "error": str(e), "message": "Error get_room_layout_scene_usd_separate"}"""

def validate_room_stability(scene_save_dir: str, room_id: str) -> dict:
    """Validate a room in one atomic service request, without editing JSON."""
    return get_isaac_connection().send_command('validate_room_stability', {'scene_save_dir': scene_save_dir, 'room_id': room_id})

def simulate_the_scene() -> str:
    """
    Simulate the scene. Before simulate the scene, you need to call create_room_layout_scene(scene_save_dir) first to create the scene.


    Returns:
        String with the result of the simulation. Each item in the dictionary is a key-value pair.
        The key is the object id in the simulation.

        The value is a dictionary with the following keys:

    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command('simulate_the_scene')
        return result
    except Exception as e:
        logger.error(f'Error simulate_the_scene: {str(e)}')
        return str({'status': 'error', 'error': str(e), 'message': 'Error simulate_the_scene'})

def simulate_the_scene_groups() -> str:
    """
    Simulate the scene. Before simulate the scene, you need to call create_room_layout_scene(scene_save_dir) first to create the scene.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command('simulate_the_scene_groups')
        return result
    except Exception as e:
        logger.error(f'Error simulate_the_scene_groups: {str(e)}')
        return str({'status': 'error', 'error': str(e), 'message': 'Error simulate_the_scene_groups'})

def transform(prim_path: str, position: List[float]=[0, 0, 50], scale: List[float]=[10, 10, 10]) -> str:
    """
    Transform a USD model by applying position and scale.
    
    Args:
        prim_path (str): Path to the USD prim to transform
        position (list, optional): The position to set [x, y, z]
        scale (list, optional): The scale to set [x, y, z]
        
    Returns:
        String with transformation result
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command('transform', {'prim_path': prim_path, 'position': position, 'scale': scale})
        if result.get('status') == 'success':
            return f'Successfully transformed model at {prim_path} to position {position} and scale {scale}'
        else:
            return f"Error transforming model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f'Error transforming model: {str(e)}')
        return f'Error transforming model: {str(e)}'

def test_object_placements_in_single_room(room_dict_save_path: str, placements_info_path: str, only_need_one: bool=False):
    """
    Test object placements in a single room.
    """
    try:
        isaac = get_isaac_connection()
        print(f'room_dict_save_path: {room_dict_save_path}, placements_info_path: {placements_info_path}')
        result = isaac.send_command('test_object_placements_in_single_room', {'room_dict_save_path': room_dict_save_path, 'placements_info_path': placements_info_path, 'only_need_one': only_need_one})
        return result
    except Exception as e:
        logger.error(f'Error test_object_placements_in_single_room: {str(e)}')
        return f"""error: {'status': "error", "error": str(e), "message": "Error test_object_placements_in_single_room"}"""

def test_object_placements_batch_in_single_room(room_dict_save_path: str, placements_batch: List[Dict[str, Any]], parallel_width: int=16, max_candidates_per_object: int=10, simulation_steps: int=2000, equilibrium_steps: int=20, position_limit: float=0.2, rotation_limit: float=8.0, scheduling_policy: str='work_conserving'):
    """Test all on-object candidates using one reusable Isaac room build."""
    try:
        isaac = get_isaac_connection()
        return isaac.send_command('test_object_placements_batch_in_single_room', {'room_dict_save_path': room_dict_save_path, 'placements_batch': placements_batch, 'parallel_width': parallel_width, 'max_candidates_per_object': max_candidates_per_object, 'simulation_steps': simulation_steps, 'equilibrium_steps': equilibrium_steps, 'position_limit': position_limit, 'rotation_limit': rotation_limit, 'scheduling_policy': scheduling_policy})
    except Exception as error:
        logger.error('Error test_object_placements_batch_in_single_room: %s', error)
        return {'status': 'error', 'error': str(error), 'message': 'Error test_object_placements_batch_in_single_room'}
