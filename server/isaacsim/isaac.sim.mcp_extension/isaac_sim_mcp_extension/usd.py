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
import os
import carb
import omni.usd
import omni
from pathlib import Path
from pxr import UsdShade, Sdf, UsdGeom, Gf
from omni.isaac.core.utils.stage import add_reference_to_stage
import json
import requests

class USDLoader:
    """
    A class to load USD models and textures from local directories,
    with methods to bind materials and transform models.
    """

    def __init__(self):
        """
        Initialize the USDLoader class with default values.
        """
        self.usd_prim = None
        self.material = None
        self.working_dir = Path(os.environ.get('USD_WORKING_DIR', 'outputs/usd'))
        self.stage = omni.usd.get_context().get_stage()

    def load_usd_model(self, abs_path=None, task_id=None):
        """
        Load a USD model from either an absolute path or by task_id.
        
        Args:
            abs_path (str, optional): Absolute path to the USD file
            task_id (str, optional): Task ID to load model from working_dir
            
        Returns:
            str: Path to the loaded USD prim
        """
        if not (abs_path or task_id):
            raise ValueError('Either abs_path or task_id must be provided')
        if task_id:
            usd_path = self.working_dir / task_id / 'output.usd'
        else:
            usd_path = Path(abs_path)
        if not usd_path.exists():
            raise FileNotFoundError(f'USD file not found at: {usd_path}')
        if task_id:
            prim_id = task_id[-5:]
        else:
            prim_id = usd_path.stem[:5]
        usd_prim_path = f'/World/model_{prim_id}'
        self.usd_prim = add_reference_to_stage(str(usd_path), usd_prim_path)
        print(f'Loaded USD model from {usd_path} at {usd_prim_path}')
        return usd_prim_path

    def load_texture_and_create_material(self, abs_path=None, task_id=None):
        """
        Load a texture from either an absolute path or by task_id and create a material with it.
        
        Args:
            abs_path (str, optional): Absolute path to the texture file
            task_id (str, optional): Task ID to load texture from working_dir
            
        Returns:
            tuple: (str, UsdShade.Material) - Path to the loaded texture and the created material
        """
        if not (abs_path or task_id):
            raise ValueError('Either abs_path or task_id must be provided')
        if task_id:
            texture_path = self.working_dir / task_id / 'textures' / 'material_0.png'
            material_id = task_id[-5:]
        else:
            texture_path = Path(abs_path)
            material_id = texture_path.stem[:5]
        if not texture_path.exists():
            raise FileNotFoundError(f'Texture file not found at: {texture_path}')
        material_path = f'/World/SimpleMaterial_{material_id}'
        material = UsdShade.Material.Define(self.stage, material_path)
        shader = UsdShade.Shader.Define(self.stage, f'{material_path}/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput('surface', Sdf.ValueTypeNames.Token))
        texture = UsdShade.Shader.Define(self.stage, f'{material_path}/Texture')
        texture.CreateIdAttr('UsdUVTexture')
        texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(texture_path))
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3))
        print(f'Created material with texture at {material_path}')
        self.material = material
        return (str(texture_path), material)

    def bind_texture_to_model(self, prim=None, material=None):
        """
        Bind a texture to a USD model.
        
        Args:
            prim (UsdGeom.Xformable, optional): The USD prim to bind the material to.
                                               If None, uses self.usd_prim
            material (UsdShade.Material, optional): The material to bind.
                                                  If None, uses self.material
        
        Returns:
            bool: True if binding succeeded, False otherwise
        """
        if prim is None:
            prim = self.usd_prim
        if material is None:
            material = self.material
        if prim is None or material is None:
            raise ValueError('Both prim and material must be provided or previously set')
        try:
            binding_api = UsdShade.MaterialBindingAPI(prim)
            binding_api.Bind(material)
            print(f'Successfully bound material to {prim.GetPath()}')
            return True
        except Exception as e:
            print(f'Failed to bind material: {str(e)}')
            return False

    def transform(self, prim=None, position=(0, 0, 50), scale=(10, 10, 10)):
        """
        Transform a USD model by applying position and scale.
        
        Args:
            prim (UsdGeom.Xformable, optional): The USD prim to transform.
                                               If None, uses self.usd_prim
            position (tuple, optional): The position to set (x, y, z)
            scale (tuple, optional): The scale to set (x, y, z)
            
        Returns:
            UsdGeom.Xformable: The transformed prim
        """
        if prim is None:
            prim = self.usd_prim
        if prim is None:
            raise ValueError('Prim must be provided or previously set')
        xformable = UsdGeom.Xformable(prim)
        xform_ops = xformable.GetOrderedXformOps()
        translate_op = None
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op:
            translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))
        else:
            xformable.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2]))
        print(f'Model positioned at {position}')
        scale_op = None
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        if scale_op:
            scale_op.Set(Gf.Vec3d(scale[0], scale[1], scale[2]))
        else:
            xformable.AddScaleOp().Set(Gf.Vec3d(scale[0], scale[1], scale[2]))
        print(f'Model scaled to {scale}')
        return xformable

    def _load_prim(self, url, path: str='/World/my_usd', size=1.0):
        """Helper method to load a USD prim from url to specific scene path."""
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(path)
            if prim:
                for p in stage.Traverse():
                    carb.log_info(f'Error in _load_prim: {p.GetPrimPath()}')
                count = len([p for p in stage.Traverse()])
                path = f'{path}_{count + 1}'
            carb.log_info(f'loading from {url} to {path}')
            prim = stage.DefinePrim(path)
            carb.log_info(f'prim loaded: {(path, prim)}')
            prim.GetReferences().AddReference(url)
            return prim
        except Exception as e:
            carb.log_info(f'Error in _load_prim: {str(e)}')
            return {'error': str(e)}

    def _set_transform(self, prim, location=None, rotation=None, scale=None):
        """Set transform operations on a USD prim."""
        if not prim.IsA(UsdGeom.Xformable):
            return
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        ops = []
        if scale is not None:
            scale_op = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionFloat)
            scale_op.Set(Gf.Vec3d(scale[0], scale[1], scale[2]))
            ops.append(scale_op)
        if rotation is not None:
            rot_op = xformable.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ, UsdGeom.XformOp.PrecisionDouble)
            rot_op.Set(Gf.Vec3d(rotation[0], rotation[1], rotation[2]))
            ops.append(rot_op)
        if location is not None:
            trans_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble)
            trans_op.Set(Gf.Vec3d(location[0], location[1], location[2]))
            ops.append(trans_op)
        xformable.SetXformOpOrder(ops)

    def _set_color(self, prim, color):
        """Set a simple color material on a USD prim."""
        stage = omni.usd.get_context().get_stage()
        if not UsdGeom.Gprim(prim):
            return
        material_path = str(prim.GetPath()) + '_material'
        material = UsdShade.Material.Define(stage, material_path)
        shader_path = material_path + '/PreviewSurface'
        pbrShader = UsdShade.Shader.Define(stage, shader_path)
        pbrShader.CreateIdAttr('UsdPreviewSurface')
        pbrShader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color[0], color[1], color[2]))
        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI(prim).Bind(material)

    def load_usd_from_url(self, url_path=None, target_path=None, location=None, rotation=None, scale=None, size=1.0, color=None):
        """
        Load a usd object from url and insert into current scene at target path.
        
        Args:
            url (str): url of the usd object to be loaded
            target_path (str, optional): target path for the loaded object
            location (list, optional): [x, y, z] position
            rotation (list, optional): [x, y, z] rotation
            scale (list, optional): [x, y, z] scale
            size (float, optional): Size parameter
            color (list, optional): [r, g, b] color
            
        Returns:
            dict: Information about the loaded object
        """
        try:
            prim = self._load_prim(url_path, path=target_path)
            path = str(prim.GetPath())
            if location or rotation or scale:
                self._set_transform(prim, location, rotation, scale)
            if color:
                self._set_color(prim, color)
            prim_info = {'target_path': path}
            if prim.IsA(UsdGeom.Xformable):
                xformable = UsdGeom.Xformable(prim)
                trans = xformable.GetLocalTransformation()
                translation = trans.ExtractTranslation()
                prim_info['transform'] = {'translation': [translation[0], translation[1], translation[2]]}
            details = json.dumps(prim_info, indent=2)
            print(f'Loaded USD model from {url_path} at {path}')
            return path
        except Exception as e:
            carb.log_info(f'Error in load_usd_from_url: {str(e)}')
            return {'error': str(e)}

