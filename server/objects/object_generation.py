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
ssh -L 14412:gpu-h100-0272:14412 hongchix@pdx
"""
import requests
import json
import os
import sys
import time
from pathlib import Path
import sys
import numpy as np
import trimesh
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SERVER_ROOT_DIR
try:
    from key import SERVER_URL
    from objects.load_glb import load_glb_advanced
    from objects.object_attribute_inference import infer_attributes_from_claude, estimate_front_from_mesh
except ImportError:
    sys.path.insert(0, SERVER_ROOT_DIR)
    from key import SERVER_URL
    from objects.load_glb import load_glb_advanced
    from objects.object_attribute_inference import infer_attributes_from_claude, estimate_front_from_mesh
import torch
from pytorch3d.io import save_obj

class TrellisClient:

    def __init__(self, server_url=SERVER_URL):
        self.server_url = server_url

    def health_check(self):
        """Check if the server is running"""
        try:
            response = requests.get(f'{self.server_url}/health', timeout=10)
            response.raise_for_status()
            health = response.json()
            return health if health.get('status') == 'healthy' and health.get('service') == 'sage-trellis' else None
        except Exception as e:
            print(f'Health check failed: {e}', file=sys.stderr)
            return None

    def generate_model(self, input_text, seed=1, output_file='generated_model.glb'):
        """Generate 3D model and save GLB file using two-phase protocol with retry logic"""
        max_retries = 3
        job_id = None
        for attempt in range(max_retries):
            if attempt > 0:
                print(f'\n⟳ Retry attempt {attempt}/{max_retries - 1}...', file=sys.stderr)
            try:
                print(f"Generating 3D model for: '{input_text}'", file=sys.stderr)
                payload = {'input_text': input_text, 'seed': seed}
                print('Waiting for server to be ready...', file=sys.stderr)
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    health = self.health_check()
                    if health:
                        break
                    time.sleep(1)
                else:
                    raise TimeoutError(f'TRELLIS health check timed out: {self.server_url}')
                print('Server is ready. Submitting generation request...', file=sys.stderr)
                total_trials = 0
                while job_id is None:
                    try:
                        response = requests.post(f'{self.server_url}/generate', json=payload, timeout=10)
                        if response.status_code == 202:
                            ack_data = response.json()
                            job_id = ack_data.get('job_id')
                            print(f'✓ Request acknowledged. Job ID: {job_id}', file=sys.stderr)
                            print(f"Message: {ack_data.get('message', 'Processing started')}", file=sys.stderr)
                            break
                        else:
                            print(f'Server returned unexpected status: {response.status_code}', file=sys.stderr)
                            print(response.text, file=sys.stderr)
                    except Exception as e:
                        print(f'Error submitting request: {e}', file=sys.stderr)
                    total_trials += 1
                    if total_trials > 10:
                        raise Exception('Failed to submit request after 10 trials.')
                    time.sleep(2)
                if not job_id:
                    raise Exception('Failed to get job ID from server.')
                print(f'Waiting for generation to complete (Job ID: {job_id}, this may take several minutes)...', file=sys.stderr)
                poll_count = 0
                max_polls = int(os.environ.get('TRELLIS_MAX_POLLS', '200'))
                while poll_count < max_polls:
                    try:
                        if poll_count > 0 and poll_count % 10 == 0:
                            status_response = requests.get(f'{self.server_url}/job/{job_id}', timeout=10)
                            if status_response.status_code == 200:
                                print('✓ Generation completed successfully!', file=sys.stderr)
                                with open(output_file, 'wb') as f:
                                    f.write(status_response.content)
                                return True
                            elif status_response.status_code == 202:
                                status_data = status_response.json()
                                if poll_count % 10 == 0:
                                    print(f"  Status: {status_data.get('status', 'processing')}... (poll {poll_count})", file=sys.stderr)
                            elif status_response.status_code == 500:
                                error_data = status_response.json()
                                print(f"✗ Generation failed: {error_data.get('error', 'Unknown error')}", file=sys.stderr)
                                job_id = None
                                break
                            elif status_response.status_code == 404:
                                print(f'✗ Job not found on server', file=sys.stderr)
                                job_id = None
                                break
                        else:
                            time.sleep(1)
                    except Exception as e:
                        if poll_count % 10 == 0:
                            print(f'  Polling error (will retry): {e}', file=sys.stderr)
                    poll_count += 1
                    time.sleep(1)
                if poll_count >= max_polls:
                    print(f'✗ Polling window exhausted after {max_polls} polls; retain job {job_id}', file=sys.stderr)
            except Exception as e:
                print(f'Generation failed: {e}', file=sys.stderr)
        print(f'✗ Failed after {max_retries} attempts', file=sys.stderr)
        return False

def merge_vertices(mesh_dict):
    """
    Merge vertices of the mesh.
    Args:
        mesh_dict: A dictionary containing the mesh data.
        keys:
            "mesh": trimesh.Trimesh
            "tex_coords": {
                "vts": numpy.ndarray
                "fts": numpy.ndarray
            }
            "texture": numpy.ndarray
    Returns:
        A dictionary exactly the same as the input mesh_dict, but with the merged mesh.
    """
    import numpy as np
    from scipy.spatial import KDTree
    mesh = mesh_dict['mesh']
    tex_coords = mesh_dict['tex_coords']
    texture = mesh_dict['texture']
    original_vertices = mesh.vertices.copy()
    original_faces = mesh.faces.copy()
    original_vts = tex_coords['vts'].copy()
    original_fts = tex_coords['fts'].copy()
    merged_mesh = mesh.copy()
    merged_mesh.merge_vertices(digits_vertex=6, merge_tex=True)
    merged_vertices = merged_mesh.vertices
    merged_faces = merged_mesh.faces
    original_tree = KDTree(original_vertices)
    (distances, vertex_mapping) = original_tree.query(merged_vertices)
    merged_triangles = merged_vertices[merged_faces.reshape(-1)].reshape(-1, 3, 3)
    original_triangles = original_vertices[original_faces.reshape(-1)].reshape(-1, 3, 3)

    def get_all_orderings(triangles):
        """Get all 6 possible vertex orderings for each triangle"""
        orderings = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])
        all_ordered = triangles[:, None, orderings].reshape(-1, 3, 3)
        triangle_indices = np.repeat(np.arange(len(triangles)), 6)
        ordering_indices = np.tile(np.arange(6), len(triangles))
        return (all_ordered, triangle_indices, ordering_indices)
    (original_all_ordered, orig_tri_indices, orig_order_indices) = get_all_orderings(original_triangles)
    original_flattened = original_all_ordered.reshape(-1, 9)
    merged_flattened = merged_triangles.reshape(-1, 9)
    triangle_tree = KDTree(original_flattened)
    (tri_distances, tri_mapping_indices) = triangle_tree.query(merged_flattened)
    face_mapping = orig_tri_indices[tri_mapping_indices]
    new_fts = original_fts[face_mapping]
    new_vts = original_vts
    updated_mesh_dict = {'mesh': merged_mesh, 'tex_coords': {'vts': new_vts, 'fts': new_fts}, 'texture': texture}
    return updated_mesh_dict

def get_lcc_mesh(mesh):
    verts = mesh.vertices
    faces = mesh.faces
    edges = mesh.edges_sorted.reshape((-1, 2))
    components = trimesh.graph.connected_components(edges, min_len=1, engine='scipy')
    largest_cc = np.argmax(np.array([comp.shape[0] for comp in components]).reshape(-1), axis=0)
    verts_map = components[largest_cc].reshape(-1)
    verts_map = np.sort(np.unique(verts_map))
    keep = np.zeros(verts.shape[0]).astype(np.bool_)
    keep[verts_map] = True
    filter_mapping = np.arange(keep.shape[0])[keep]
    filter_unmapping = -np.ones(keep.shape[0])
    filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    verts_lcc = verts[keep]
    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = np.logical_and(keep_0, keep_1)
    keep_faces = np.logical_and(keep_faces, keep_2)
    faces_lcc = faces[keep_faces]
    faces_map = keep_faces
    faces_lcc[:, 0] = filter_unmapping[faces_lcc[:, 0]]
    faces_lcc[:, 1] = filter_unmapping[faces_lcc[:, 1]]
    faces_lcc[:, 2] = filter_unmapping[faces_lcc[:, 2]]
    return (verts_lcc, faces_lcc, verts_map, faces_map)

def extract_max_connected_component(mesh_dict):
    """
    Extract the maximum connected component from the mesh.
    Args:
        mesh_dict: A dictionary containing the mesh data.
        keys:
            "mesh": trimesh.Trimesh
            "tex_coords": {
                "vts": numpy.ndarray
                "fts": numpy.ndarray
            }
            "texture": numpy.ndarray

    Returns:
        A dictionary containing the mesh data of the maximum connected component.
    """
    mesh = mesh_dict['mesh']
    tex_coords = mesh_dict['tex_coords']
    texture = mesh_dict['texture']
    (verts_lcc, faces_lcc, verts_map, faces_map) = get_lcc_mesh(mesh)
    fts = tex_coords['fts']
    fts_new = fts[faces_map]
    mesh = trimesh.Trimesh(vertices=verts_lcc, faces=faces_lcc)
    return {'mesh': mesh, 'tex_coords': {'vts': tex_coords['vts'], 'fts': fts_new}, 'texture': texture}
client = TrellisClient()
health = client.health_check()
if health:
    print('✓ Server is running and healthy', file=sys.stderr)
    print(f"GPU available: {health.get('gpu_available', 'Unknown')}", file=sys.stderr)
else:
    print("✗ Cannot connect to server. Make sure it's running and SSH tunnel is active.", file=sys.stderr)

def generate_model_from_text(input_text, output_path, reference_object_size=None, estimate_front=True):
    success = client.generate_model(input_text=input_text, seed=random.randint(0, 1000000), output_file=output_path)
    if success:
        print('✓ Model generation completed successfully!', file=sys.stderr)
        mesh_dict = load_glb_advanced(output_path)
        mesh_dict = merge_vertices(mesh_dict)
        if reference_object_size is not None:
            size_text_description = f' Reference approximate size (Not the exact shape of the object, but the approximate shape of the object): {reference_object_size[0] / 100.0}m x {reference_object_size[1] / 100.0}m x {reference_object_size[2] / 100.0}m.'
        else:
            size_text_description = ''
        mesh = mesh_dict['mesh']
        mesh_vertices = mesh.vertices.copy()
        mesh_vertices[:, 1] = mesh.vertices[:, 2]
        mesh_vertices[:, 2] = mesh.vertices[:, 1]
        mesh.vertices = mesh_vertices
        mesh.vertices[:, 0] = mesh.vertices[:, 0] - 0.5 * (mesh.vertices[:, 0].max() + mesh.vertices[:, 0].min())
        mesh.vertices[:, 1] = mesh.vertices[:, 1] - 0.5 * (mesh.vertices[:, 1].max() + mesh.vertices[:, 1].min())
        mesh.vertices[:, 2] = mesh.vertices[:, 2] - mesh.vertices[:, 2].min()
        mesh.faces = mesh.faces[:, [0, 2, 1]].copy()
        mesh_dict['tex_coords']['fts'] = mesh_dict['tex_coords']['fts'][:, [0, 2, 1]].copy()
        if estimate_front:
            mesh_dict = estimate_front_from_mesh(mesh_dict)
        height_list = []
        width_list = []
        length_list = []
        num_inference_height = 3
        for _ in range(num_inference_height):
            object_attributes = infer_attributes_from_claude(mesh_dict, caption=input_text)
            height_list.append(object_attributes['height'])
            width_list.append(object_attributes['width'])
            length_list.append(object_attributes['length'])
        height_mean = float(np.mean(np.array(height_list)))
        object_attributes['height'] = height_mean
        width_mean = float(np.mean(np.array(width_list)))
        object_attributes['width'] = width_mean
        length_mean = float(np.mean(np.array(length_list)))
        object_attributes['length'] = length_mean
        mesh_dict['object_attributes'] = object_attributes
        scale_factor_height = height_mean / float(mesh_dict['mesh'].vertices[:, 2].max() - mesh_dict['mesh'].vertices[:, 2].min())
        scale_factor_xy = max(width_mean, length_mean) / max(float(mesh_dict['mesh'].vertices[:, 0].max() - mesh_dict['mesh'].vertices[:, 0].min()), float(mesh_dict['mesh'].vertices[:, 1].max() - mesh_dict['mesh'].vertices[:, 1].min()))
        scale_factor = np.mean([scale_factor_height, scale_factor_xy])
        mesh_dict['mesh'].vertices = mesh_dict['mesh'].vertices * scale_factor
        mesh_dict['mesh'].vertices[:, 2] = mesh_dict['mesh'].vertices[:, 2] + 0.001
        return mesh_dict
    else:
        raise Exception('Model generation failed.')
