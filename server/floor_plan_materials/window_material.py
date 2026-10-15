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
import numpy as np
import os
import torch
import trimesh
import cv2
from PIL import Image
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from floor_plan_materials.material_generator import material_generate_from_prompt, generate_texture_map_from_prompt_and_sketch, generate_texture_map_from_prompt_and_sketch_and_image, generate_texture_map_from_prompt_and_color, generate_texture_map_from_prompt_and_color_and_sketch
from constants import SERVER_ROOT_DIR

class WindowMaterialGenerator:

    def __init__(self):
        self.window_type_grids = {'fixed': (1, 1), 'slider': (2, 1), 'hung': (1, 2), 'bay_small': (2, 2), 'bay_medium': (3, 2), 'bay_large': (3, 3), 'picture': (1, 1), 'casement': (1, 1), 'double_hung': (1, 2), 'single_hung': (1, 2)}
        self.default_glass_color = [0.8, 0.9, 1.0]
        self.default_frame_color = [0.3, 0.3, 0.3]

    def generate_window_by_type(self, window_type, color_glass=None, color_frame=None):
        """
        Generate window material by window type name.
        
        Args:
            window_type: String window type ("fixed", "slider", "hung", "bay_small", etc.)
            color_glass: RGB color for glass regions (optional, uses default if None)
            color_frame: RGB color for frame regions (optional, uses default if None)
            
        Returns:
            dict: Same as generate_window_material
        """
        if window_type not in self.window_type_grids:
            raise ValueError(f'Unknown window type: {window_type}. Available types: {list(self.window_type_grids.keys())}')
        grid_xy = self.window_type_grids[window_type]
        if color_glass is None:
            color_glass = self.default_glass_color
        if color_frame is None:
            color_frame = self.default_frame_color
        return self.generate_window_material(grid_xy, color_glass, color_frame)

    def generate_window_material(self, window_grid_xy, color_glass, color_frame):
        """
        Generate window material with glass and frame regions.
        
        Args:
            window_grid_xy: Tuple (nx, ny) defining the grid layout
                           - Bay: (n, m) grid  
                           - Hung: (1, 2) grid
                           - Slider: (2, 1) grid
                           - Fixed: (1, 1) grid
            color_glass: RGB color for glass regions (3-element array/list)
            color_frame: RGB color for frame regions (3-element array/list)
            
        Returns:
            dict: {
                "texture_image": numpy array of texture image,
                "vts": texture coordinates (N, 2),
                "fts": face texture indices (M, 3)
            }
        """
        (nx, ny) = window_grid_xy
        if nx <= 0 or ny <= 0:
            raise ValueError(f'Grid dimensions must be positive: got {window_grid_xy}')
        color_glass = np.array(color_glass, dtype=np.float32)
        color_frame = np.array(color_frame, dtype=np.float32)
        if color_glass.shape != (3,) or color_frame.shape != (3,):
            raise ValueError('Colors must be 3-element arrays (RGB)')
        color_glass = np.clip(color_glass, 0.0, 1.0)
        color_frame = np.clip(color_frame, 0.0, 1.0)
        window_area_width = 512
        window_area_height = 512
        edge_strip_width = 64
        edge_strip_height = 64
        total_texture_width = window_area_width + edge_strip_width
        total_texture_height = window_area_height + edge_strip_height
        window_texture_image = np.ones((total_texture_height, total_texture_width, 3), dtype=np.float32)
        window_texture_image[:, :, :] = color_frame
        frame_thickness_ratio = 0.05
        cell_width = window_area_width // nx
        cell_height = window_area_height // ny
        frame_thickness_x = max(1, int(cell_width * frame_thickness_ratio))
        frame_thickness_y = max(1, int(cell_height * frame_thickness_ratio))
        window_texture_image[:window_area_height, :window_area_width, :] = color_frame
        for i in range(nx):
            for j in range(ny):
                x_start = i * cell_width + frame_thickness_x
                x_end = (i + 1) * cell_width - frame_thickness_x
                y_start = j * cell_height + frame_thickness_y
                y_end = (j + 1) * cell_height - frame_thickness_y
                x_start = max(0, min(x_start, window_area_width - 1))
                x_end = max(x_start + 1, min(x_end, window_area_width))
                y_start = max(0, min(y_start, window_area_height - 1))
                y_end = max(y_start + 1, min(y_end, window_area_height))
                window_texture_image[y_start:y_end, x_start:x_end, :] = color_glass
        for j in range(1, ny):
            y_center = j * cell_height
            y_start = max(0, y_center - frame_thickness_y // 2)
            y_end = min(window_area_height, y_center + frame_thickness_y // 2)
            window_texture_image[y_start:y_end, :window_area_width, :] = color_frame
        for i in range(1, nx):
            x_center = i * cell_width
            x_start = max(0, x_center - frame_thickness_x // 2)
            x_end = min(window_area_width, x_center + frame_thickness_x // 2)
            window_texture_image[:window_area_height, x_start:x_end, :] = color_frame
        window_texture_image[:window_area_height, window_area_width:, :] = color_frame
        window_texture_image[window_area_height:, :window_area_width, :] = color_frame
        window_texture_image[window_area_height:, window_area_width:, :] = color_frame
        temp_box = trimesh.creation.box([1, 1, 1])
        face_normals = temp_box.face_normals
        window_u_max = window_area_width / total_texture_width
        window_v_max = window_area_height / total_texture_height
        edge_right_u_min = window_area_width / total_texture_width
        edge_right_u_max = 1.0
        edge_bottom_v_min = window_area_height / total_texture_height
        edge_bottom_v_max = 1.0
        unique_vts = []
        unique_fts = []
        for (face_idx, (face, normal)) in enumerate(zip(temp_box.faces, face_normals)):
            face_vertices = temp_box.vertices[face]
            if abs(normal[1]) > 0.9:
                (min_x, max_x) = (face_vertices[:, 0].min(), face_vertices[:, 0].max())
                (min_z, max_z) = (face_vertices[:, 2].min(), face_vertices[:, 2].max())
                tex_coords = np.array([[(face_vertices[i, 0] - min_x) / (max_x - min_x) * window_u_max, (face_vertices[i, 2] - min_z) / (max_z - min_z) * window_v_max] for i in range(3)])
            elif abs(normal[0]) > 0.9:
                (min_y, max_y) = (face_vertices[:, 1].min(), face_vertices[:, 1].max())
                (min_z, max_z) = (face_vertices[:, 2].min(), face_vertices[:, 2].max())
                tex_coords = np.array([[edge_right_u_min + (face_vertices[i, 1] - min_y) / (max_y - min_y) * (edge_right_u_max - edge_right_u_min), (face_vertices[i, 2] - min_z) / (max_z - min_z) * window_v_max] for i in range(3)])
            elif abs(normal[2]) > 0.9:
                (min_x, max_x) = (face_vertices[:, 0].min(), face_vertices[:, 0].max())
                (min_y, max_y) = (face_vertices[:, 1].min(), face_vertices[:, 1].max())
                tex_coords = np.array([[(face_vertices[i, 0] - min_x) / (max_x - min_x) * window_u_max, edge_bottom_v_min + (face_vertices[i, 1] - min_y) / (max_y - min_y) * (edge_bottom_v_max - edge_bottom_v_min)] for i in range(3)])
            else:
                tex_coords = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            start_idx = len(unique_vts)
            unique_vts.extend(tex_coords)
            unique_fts.append([start_idx, start_idx + 1, start_idx + 2])
        vts = np.array(unique_vts, dtype=np.float32)
        fts = np.array(unique_fts, dtype=np.int32)
        vts[:, 1] = 1 - vts[:, 1]
        return {'texture_image': window_texture_image, 'vts': vts, 'fts': fts, 'grid_layout': window_grid_xy, 'dimensions': (total_texture_width, total_texture_height)}

    def generate_window_material_color_diffusion(self, window_grid_xy, color_glass, color_frame):
        """
        Generate window material with glass and frame regions using color-based texture generation.
        
        Args:
            window_grid_xy: Tuple (nx, ny) defining the grid layout
                           - Bay: (n, m) grid  
                           - Hung: (1, 2) grid
                           - Slider: (2, 1) grid
                           - Fixed: (1, 1) grid
            color_glass: RGB color for glass regions (3-element array/list)
            color_frame: RGB color for frame regions (3-element array/list)
            
        Returns:
            dict: {
                "texture_image": numpy array of texture image,
                "vts": texture coordinates (N, 2),
                "fts": face texture indices (M, 3)
            }
        """
        (nx, ny) = window_grid_xy
        if nx <= 0 or ny <= 0:
            raise ValueError(f'Grid dimensions must be positive: got {window_grid_xy}')
        color_glass = np.array(color_glass, dtype=np.float32)
        color_frame = np.array(color_frame, dtype=np.float32)
        if color_glass.shape != (3,) or color_frame.shape != (3,):
            raise ValueError('Colors must be 3-element arrays (RGB)')
        color_glass = np.clip(color_glass, 0.0, 1.0)
        color_frame = np.clip(color_frame, 0.0, 1.0)
        window_area_width = 512
        window_area_height = 512
        edge_strip_width = 64
        edge_strip_height = 64
        total_texture_width = window_area_width + edge_strip_width
        total_texture_height = window_area_height + edge_strip_height
        glass_mask = np.zeros((total_texture_height, total_texture_width), dtype=bool)
        frame_mask = np.ones((total_texture_height, total_texture_width), dtype=bool)
        frame_thickness_ratio = 0.05
        cell_width = window_area_width // nx
        cell_height = window_area_height // ny
        frame_thickness_x = max(1, int(cell_width * frame_thickness_ratio))
        frame_thickness_y = max(1, int(cell_height * frame_thickness_ratio))
        for i in range(nx):
            for j in range(ny):
                x_start = i * cell_width + frame_thickness_x
                x_end = (i + 1) * cell_width - frame_thickness_x
                y_start = j * cell_height + frame_thickness_y
                y_end = (j + 1) * cell_height - frame_thickness_y
                x_start = max(0, min(x_start, window_area_width - 1))
                x_end = max(x_start + 1, min(x_end, window_area_width))
                y_start = max(0, min(y_start, window_area_height - 1))
                y_end = max(y_start + 1, min(y_end, window_area_height))
                glass_mask[y_start:y_end, x_start:x_end] = True
                frame_mask[y_start:y_end, x_start:x_end] = False
        for j in range(1, ny):
            y_center = j * cell_height
            y_start = max(0, y_center - frame_thickness_y // 2)
            y_end = min(window_area_height, y_center + frame_thickness_y // 2)
            glass_mask[y_start:y_end, :window_area_width] = False
            frame_mask[y_start:y_end, :window_area_width] = True
        for i in range(1, nx):
            x_center = i * cell_width
            x_start = max(0, x_center - frame_thickness_x // 2)
            x_end = min(window_area_width, x_center + frame_thickness_x // 2)
            glass_mask[:window_area_height, x_start:x_end] = False
            frame_mask[:window_area_height, x_start:x_end] = True
        glass_texture_pil = generate_texture_map_from_prompt_and_color('glass', color_glass.tolist())
        frame_texture_pil = generate_texture_map_from_prompt_and_color('frame', color_frame.tolist())
        glass_texture_pil = glass_texture_pil.resize((total_texture_width, total_texture_height), Image.LANCZOS)
        frame_texture_pil = frame_texture_pil.resize((total_texture_width, total_texture_height), Image.LANCZOS)
        glass_texture = np.array(glass_texture_pil).astype(np.float32) / 255.0
        frame_texture = np.array(frame_texture_pil).astype(np.float32) / 255.0
        window_texture_image = frame_texture.copy()
        glass_mask_3d = np.stack([glass_mask, glass_mask, glass_mask], axis=2)
        window_texture_image[glass_mask_3d] = glass_texture[glass_mask_3d]
        temp_box = trimesh.creation.box([1, 1, 1])
        face_normals = temp_box.face_normals
        window_u_max = window_area_width / total_texture_width
        window_v_max = window_area_height / total_texture_height
        edge_right_u_min = window_area_width / total_texture_width
        edge_right_u_max = 1.0
        edge_bottom_v_min = window_area_height / total_texture_height
        edge_bottom_v_max = 1.0
        unique_vts = []
        unique_fts = []
        for (face_idx, (face, normal)) in enumerate(zip(temp_box.faces, face_normals)):
            face_vertices = temp_box.vertices[face]
            if abs(normal[1]) > 0.9:
                (min_x, max_x) = (face_vertices[:, 0].min(), face_vertices[:, 0].max())
                (min_z, max_z) = (face_vertices[:, 2].min(), face_vertices[:, 2].max())
                tex_coords = np.array([[(face_vertices[i, 0] - min_x) / (max_x - min_x) * window_u_max, (face_vertices[i, 2] - min_z) / (max_z - min_z) * window_v_max] for i in range(3)])
            elif abs(normal[0]) > 0.9:
                (min_y, max_y) = (face_vertices[:, 1].min(), face_vertices[:, 1].max())
                (min_z, max_z) = (face_vertices[:, 2].min(), face_vertices[:, 2].max())
                tex_coords = np.array([[edge_right_u_min + (face_vertices[i, 1] - min_y) / (max_y - min_y) * (edge_right_u_max - edge_right_u_min), (face_vertices[i, 2] - min_z) / (max_z - min_z) * window_v_max] for i in range(3)])
            elif abs(normal[2]) > 0.9:
                (min_x, max_x) = (face_vertices[:, 0].min(), face_vertices[:, 0].max())
                (min_y, max_y) = (face_vertices[:, 1].min(), face_vertices[:, 1].max())
                tex_coords = np.array([[(face_vertices[i, 0] - min_x) / (max_x - min_x) * window_u_max, edge_bottom_v_min + (face_vertices[i, 1] - min_y) / (max_y - min_y) * (edge_bottom_v_max - edge_bottom_v_min)] for i in range(3)])
            else:
                tex_coords = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            start_idx = len(unique_vts)
            unique_vts.extend(tex_coords)
            unique_fts.append([start_idx, start_idx + 1, start_idx + 2])
        vts = np.array(unique_vts, dtype=np.float32)
        fts = np.array(unique_fts, dtype=np.int32)
        vts[:, 1] = 1 - vts[:, 1]
        return {'texture_image': window_texture_image, 'vts': vts, 'fts': fts, 'grid_layout': window_grid_xy, 'dimensions': (total_texture_width, total_texture_height)}

    def generate_window_material_from_prompt(self, window_grid_xy, prompt):
        """
        Generate window material with glass and frame regions using texture generation from prompt and sketch.
        
        Args:
            window_grid_xy: Tuple (nx, ny) defining the grid layout
                           - Bay: (n, m) grid  
                           - Hung: (1, 2) grid
                           - Slider: (2, 1) grid
                           - Fixed: (1, 1) grid
            prompt: Text prompt for texture generation
            
        Returns:
            dict: {
                "texture_image": numpy array of texture image,
                "vts": texture coordinates (N, 2),
                "fts": face texture indices (M, 3)
            }
        """
        (nx, ny) = window_grid_xy
        if nx <= 0 or ny <= 0:
            raise ValueError(f'Grid dimensions must be positive: got {window_grid_xy}')
        total_texture_width = 512
        total_texture_height = 512
        window_area_width = total_texture_width
        window_area_height = total_texture_height
        frame_thickness_ratio = 0.01
        cell_width = window_area_width // nx
        cell_height = window_area_height // ny
        frame_thickness_x = max(1, int(cell_width * frame_thickness_ratio))
        frame_thickness_y = max(1, int(cell_height * frame_thickness_ratio))
        edge_mask = np.zeros((window_area_height, window_area_width), dtype=np.uint8)
        for i in range(nx):
            for j in range(ny):
                x_start = i * cell_width + frame_thickness_x
                x_end = (i + 1) * cell_width - frame_thickness_x
                y_start = j * cell_height + frame_thickness_y
                y_end = (j + 1) * cell_height - frame_thickness_y
                x_start = max(0, min(x_start, window_area_width - 1))
                x_end = max(x_start + 1, min(x_end, window_area_width))
                y_start = max(0, min(y_start, window_area_height - 1))
                y_end = max(y_start + 1, min(y_end, window_area_height))
                edge_mask[y_start:y_end, x_start:x_end] = 255
        edges = cv2.Canny(edge_mask, 50, 150)
        edges[1:5, :] = 255
        edges[-5:-1, :] = 255
        edges[:, 1:5] = 255
        edges[:, -5:-1] = 255
        full_edge_map = edges
        texture_pil = generate_texture_map_from_prompt_and_sketch(prompt, full_edge_map)
        window_texture_image = np.array(texture_pil).astype(np.float32) / 255.0
        if window_texture_image.shape[:2] != (total_texture_height, total_texture_width):
            texture_pil_resized = texture_pil.resize((total_texture_width, total_texture_height), Image.LANCZOS)
            window_texture_image = np.array(texture_pil_resized).astype(np.float32) / 255.0
        temp_box = trimesh.creation.box([1, 1, 1])
        face_normals = temp_box.face_normals
        window_u_min = 0.0
        window_u_max = 1.0
        window_v_min = 0.0
        window_v_max = 1.0
        unique_vts = []
        unique_fts = []
        for (face_idx, (face, normal)) in enumerate(zip(temp_box.faces, face_normals)):
            face_vertices = temp_box.vertices[face]
            if abs(normal[1]) > 0.9:
                (min_x, max_x) = (face_vertices[:, 0].min(), face_vertices[:, 0].max())
                (min_z, max_z) = (face_vertices[:, 2].min(), face_vertices[:, 2].max())
                tex_coords = np.array([[(face_vertices[i, 0] - min_x) / (max_x - min_x), (face_vertices[i, 2] - min_z) / (max_z - min_z)] for i in range(3)])
            elif abs(normal[0]) > 0.9:
                (min_y, max_y) = (face_vertices[:, 1].min(), face_vertices[:, 1].max())
                (min_z, max_z) = (face_vertices[:, 2].min(), face_vertices[:, 2].max())
                tex_coords = np.array([[(face_vertices[i, 1] - min_y) / (max_y - min_y), (face_vertices[i, 2] - min_z) / (max_z - min_z)] for i in range(3)])
            elif abs(normal[2]) > 0.9:
                (min_x, max_x) = (face_vertices[:, 0].min(), face_vertices[:, 0].max())
                (min_y, max_y) = (face_vertices[:, 1].min(), face_vertices[:, 1].max())
                tex_coords = np.array([[(face_vertices[i, 0] - min_x) / (max_x - min_x), (face_vertices[i, 1] - min_y) / (max_y - min_y)] for i in range(3)])
            else:
                tex_coords = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            start_idx = len(unique_vts)
            unique_vts.extend(tex_coords)
            unique_fts.append([start_idx, start_idx + 1, start_idx + 2])
        vts = np.array(unique_vts, dtype=np.float32)
        fts = np.array(unique_fts, dtype=np.int32)
        vts[:, 1] = 1 - vts[:, 1]
        return {'texture_image': window_texture_image, 'vts': vts, 'fts': fts, 'grid_layout': window_grid_xy, 'dimensions': (total_texture_width, total_texture_height)}

    def save_window_material(self, window_material, save_path_prefix):
        """
        Save window material to disk.
        
        Args:
            window_material: Result from generate_window_material
            save_path_prefix: Path prefix for saving files
            
        Returns:
            dict: Paths to saved files
        """
        import compress_pickle
        texture_path = f'{save_path_prefix}_texture.png'
        texture_image = window_material['texture_image']
        texture_image_pil = Image.fromarray((texture_image * 255).astype(np.uint8))
        texture_image_pil.save(texture_path)
        coords_path = f'{save_path_prefix}_tex_coords.pkl'
        tex_coords_dict = {'vts': window_material['vts'], 'fts': window_material['fts'], 'grid_layout': window_material['grid_layout'], 'dimensions': window_material['dimensions']}
        compress_pickle.dump(tex_coords_dict, coords_path)
        return {'texture_path': texture_path, 'coords_path': coords_path}

    def get_available_window_types(self):
        """Get list of available window types."""
        return list(self.window_type_grids.keys())
