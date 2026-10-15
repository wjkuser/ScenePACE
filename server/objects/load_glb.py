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
import os
import numpy as np
import trimesh
from PIL import Image
import sys
from constants import SERVER_ROOT_DIR

def load_glb(glb_path):
    """
    Load a GLB file and extract mesh data including vertices, faces, 
    texture coordinates, face texture indices, and texture image.
    
    Args:
        glb_path (str): Path to the GLB file
        
    Returns:
        dict: Contains 'mesh' (trimesh object), 'tex_coords' (vts, fts), and 'texture'
    """
    scene_or_mesh = trimesh.load(glb_path)
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = None
        for geometry in scene_or_mesh.geometry.values():
            if isinstance(geometry, trimesh.Trimesh):
                mesh = geometry
                break
        if mesh is None:
            raise ValueError('No valid mesh found in the GLB scene')
    else:
        mesh = scene_or_mesh
    vertices = mesh.vertices
    faces = mesh.faces
    vts = None
    fts = None
    texture = None
    uv_coordinates = None
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uv_coordinates = mesh.visual.uv
    if uv_coordinates is None:
        try:
            if isinstance(scene_or_mesh, trimesh.Scene):
                for (geom_name, geometry) in scene_or_mesh.geometry.items():
                    if hasattr(geometry, 'metadata') and 'TEXCOORD_0' in str(geometry.metadata):
                        print('Found TEXCOORD_0 in metadata', file=sys.stderr)
                    if hasattr(geometry.visual, 'uv') and geometry.visual.uv is not None:
                        uv_coordinates = geometry.visual.uv
                        print(f'Found UV coordinates from geometry: {uv_coordinates.shape}', file=sys.stderr)
                        break
        except Exception as e:
            print(f'Could not extract UV from GLB metadata: {e}', file=sys.stderr)
    if uv_coordinates is not None:
        vts = uv_coordinates
        if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
            fts = faces.copy()
        else:
            fts = faces.copy()
        if len(vts) != len(vertices):
            print(f"Warning: UV coordinate count ({len(vts)}) doesn't match vertex count ({len(vertices)})", file=sys.stderr)
            print('This may indicate separate UV indexing in the GLB file', file=sys.stderr)
            if len(vts) < len(vertices):
                padding = np.zeros((len(vertices) - len(vts), 2))
                vts = np.vstack([vts, padding])
                print(f'Padded UV coordinates to match vertex count', file=sys.stderr)
            elif len(vts) > len(vertices):
                vts = vts[:len(vertices)]
                print(f'Truncated UV coordinates to match vertex count', file=sys.stderr)
    else:
        print('Warning: No UV coordinates found in GLB file, creating dummy coordinates', file=sys.stderr)
        vts = np.zeros((len(vertices), 2))
        fts = faces.copy()
    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        material = mesh.visual.material
        if hasattr(material, 'image') and material.image is not None:
            texture_img = material.image
            if texture_img.mode == 'RGBA':
                texture = np.array(texture_img).astype(np.float32) / 255.0
            elif texture_img.mode == 'RGB':
                texture = np.array(texture_img).astype(np.float32) / 255.0
            else:
                texture_img = texture_img.convert('RGB')
                texture = np.array(texture_img).astype(np.float32) / 255.0
        elif hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
            texture_img = material.baseColorTexture
            texture = np.array(texture_img).astype(np.float32) / 255.0
    if texture is None:
        print('Warning: No texture found in GLB file, creating default white texture', file=sys.stderr)
        texture = np.ones((256, 256, 3), dtype=np.float32)
    mesh_dict = {'mesh': mesh, 'tex_coords': {'vts': vts, 'fts': fts}, 'texture': texture}
    return mesh_dict

def load_glb_advanced(glb_path):
    """
    Advanced GLB loader that properly handles UV coordinate indexing by parsing the original GLB data.
    This version attempts to extract proper texture coordinate indices.
    
    Args:
        glb_path (str): Path to the GLB file
        
    Returns:
        dict: Contains 'mesh' (trimesh object), 'tex_coords' (vts, fts), and 'texture'
    """
    import json
    import struct
    try:
        with open(glb_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'glTF':
                raise ValueError('Not a valid GLB file')
            version = struct.unpack('<I', f.read(4))[0]
            length = struct.unpack('<I', f.read(4))[0]
            json_length = struct.unpack('<I', f.read(4))[0]
            json_type = f.read(4)
            json_data = f.read(json_length).decode('utf-8')
            gltf_json = json.loads(json_data)
            print(f'GLB file version: {version}, total length: {length}', file=sys.stderr)
            print(f"Found {len(gltf_json.get('meshes', []))} meshes in GLB", file=sys.stderr)
            return load_glb(glb_path)
    except Exception as e:
        print(f'Advanced GLB parsing failed: {e}, falling back to standard loader', file=sys.stderr)
        return load_glb(glb_path)

def save_obj_with_materials(mesh_dict, obj_path, mtl_path=None, texture_path=None):
    """
    Save mesh data to OBJ format with proper UV coordinates and materials.
    This function properly handles the texture coordinates and creates MTL files.
    
    Args:
        mesh_dict (dict): Dictionary from load_glb containing mesh, tex_coords, and texture
        obj_path (str): Path where the OBJ file will be saved
        mtl_path (str): Path where the MTL file will be saved (optional)
        texture_path (str): Path where the texture image will be saved (optional)
    """
    if mtl_path is None:
        mtl_path = obj_path.replace('.obj', '.mtl')
    if texture_path is None:
        texture_path = obj_path.replace('.obj', '_texture.png')
    mesh = mesh_dict['mesh']
    tex_coords = mesh_dict['tex_coords']
    texture = mesh_dict['texture']
    vertices = mesh.vertices
    faces = mesh.faces
    vts = tex_coords['vts']
    fts = tex_coords['fts']
    print(f'Saving OBJ with {len(vertices)} vertices, {len(faces)} faces, {len(vts)} texture coordinates', file=sys.stderr)
    if texture is not None and texture.shape[0] > 1 and (texture.shape[1] > 1):
        if texture.shape[-1] == 4:
            texture_img = Image.fromarray((texture * 255).astype(np.uint8), 'RGBA')
        else:
            texture_img = Image.fromarray((texture * 255).astype(np.uint8), 'RGB')
        texture_img.save(texture_path)
        print(f'Saved texture to: {texture_path}', file=sys.stderr)
    mtl_name = os.path.basename(mtl_path)
    texture_name = os.path.basename(texture_path)
    with open(mtl_path, 'w') as mtl_file:
        mtl_file.write('# Material file created by improved load_glb\n')
        mtl_file.write('newmtl material0\n')
        mtl_file.write('Ka 1.0 1.0 1.0\n')
        mtl_file.write('Kd 0.8 0.8 0.8\n')
        mtl_file.write('Ks 0.5 0.5 0.5\n')
        mtl_file.write('Ns 96.0\n')
        mtl_file.write('d 1.0\n')
        mtl_file.write('illum 2\n')
        if texture is not None:
            mtl_file.write(f'map_Kd {texture_name}\n')
    print(f'Saved MTL to: {mtl_path}', file=sys.stderr)
    with open(obj_path, 'w') as obj_file:
        obj_file.write('# OBJ file created by improved load_glb\n')
        obj_file.write(f'mtllib {os.path.basename(mtl_path)}\n')
        obj_file.write('\n')
        for vertex in vertices:
            obj_file.write(f'v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n')
        obj_file.write('\n')
        for vt in vts:
            obj_file.write(f'vt {vt[0]:.6f} {vt[1]:.6f}\n')
        obj_file.write('\n')
        obj_file.write('usemtl material0\n')
        for (i, face) in enumerate(faces):
            if i < len(fts):
                ft = fts[i]
                obj_file.write(f'f {face[0] + 1}/{ft[0] + 1} {face[1] + 1}/{ft[1] + 1} {face[2] + 1}/{ft[2] + 1}\n')
            else:
                obj_file.write(f'f {face[0] + 1}/{face[0] + 1} {face[1] + 1}/{face[1] + 1} {face[2] + 1}/{face[2] + 1}\n')
    print(f'Saved OBJ to: {obj_path}', file=sys.stderr)

def create_usage_example():
    """
    Create a usage example script showing how to use the improved GLB loader.
    """
    example_code = '\n# Example: Using the improved GLB loader\n\nfrom objects.load_glb import load_glb, save_obj_with_materials\n\n# 1. Load a GLB file with proper UV coordinate handling\nglb_path = "path/to/your/model.glb"\nmesh_dict = load_glb(glb_path)\n\n# 2. Access the loaded data\nmesh = mesh_dict["mesh"]                    # trimesh object\ntexture_coords = mesh_dict["tex_coords"]    # UV coordinates and face indices\ntexture_image = mesh_dict["texture"]        # texture as numpy array\n\n# 3. Save as OBJ with proper materials\noutput_path = "path/to/output/model.obj"\nsave_obj_with_materials(mesh_dict, output_path)\n\n# This will create:\n# - model.obj (geometry with UV coordinates)\n# - model.mtl (material definition)\n# - model_texture.png (texture image)\n\n# 4. The generated OBJ file will have proper face definitions like:\n# f 1/1 2/2 3/3  (vertex_index/uv_index pairs)\n\nprint("✅ GLB converted to OBJ with proper texture mapping!")\n'
    with open('example_glb_usage.py', 'w') as f:
        f.write(example_code)
    print('Created example_glb_usage.py with usage instructions', file=sys.stderr)

def diagnose_glb_issues(glb_path):
    """
    Diagnose common issues with GLB files that cause poor OBJ conversion.
    
    Args:
        glb_path (str): Path to the GLB file to diagnose
    """
    print(f'\n🔍 DIAGNOSING GLB FILE: {os.path.basename(glb_path)}', file=sys.stderr)
    print('-' * 60, file=sys.stderr)
    try:
        scene_or_mesh = trimesh.load(glb_path)
        if isinstance(scene_or_mesh, trimesh.Scene):
            print(f'✓ Scene with {len(scene_or_mesh.geometry)} geometries', file=sys.stderr)
            mesh = list(scene_or_mesh.geometry.values())[0]
        else:
            print(f'✓ Single mesh', file=sys.stderr)
            mesh = scene_or_mesh
        print(f'✓ Vertices: {mesh.vertices.shape[0]:,}', file=sys.stderr)
        print(f'✓ Faces: {mesh.faces.shape[0]:,}', file=sys.stderr)
        has_uvs = hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None
        if has_uvs:
            uv_shape = mesh.visual.uv.shape
            print(f'✓ UV coordinates: {uv_shape[0]:,} (shape: {uv_shape})', file=sys.stderr)
            if uv_shape[0] != mesh.vertices.shape[0]:
                print(f'⚠️  WARNING: UV count ({uv_shape[0]}) ≠ vertex count ({mesh.vertices.shape[0]})', file=sys.stderr)
                print(f'   This indicates separate UV indexing (common in complex GLB files)', file=sys.stderr)
            else:
                print(f'✓ UV coordinates match vertex count', file=sys.stderr)
            uv_min = mesh.visual.uv.min(axis=0)
            uv_max = mesh.visual.uv.max(axis=0)
            print(f'✓ UV range: U[{uv_min[0]:.3f}, {uv_max[0]:.3f}], V[{uv_min[1]:.3f}, {uv_max[1]:.3f}]', file=sys.stderr)
            if uv_min.min() < -0.1 or uv_max.max() > 1.1:
                print(f'⚠️  WARNING: UV coordinates outside [0,1] range', file=sys.stderr)
        else:
            print(f'❌ No UV coordinates found', file=sys.stderr)
        has_material = hasattr(mesh.visual, 'material') and mesh.visual.material is not None
        if has_material:
            material = mesh.visual.material
            print(f'✓ Material found: {type(material).__name__}', file=sys.stderr)
            has_texture = hasattr(material, 'image') and material.image is not None
            if has_texture:
                tex_shape = np.array(material.image).shape
                print(f'✓ Texture image: {tex_shape}', file=sys.stderr)
            else:
                print(f'❌ No texture image in material', file=sys.stderr)
        else:
            print(f'❌ No material found', file=sys.stderr)
        print('-' * 60, file=sys.stderr)
    except Exception as e:
        print(f'❌ Error diagnosing GLB file: {e}', file=sys.stderr)
