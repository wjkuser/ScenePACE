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
from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils, Sdf, UsdShade
import numpy as np
import trimesh
import sys

def AddTranslate(top, offset):
    top.AddTranslateOp().Set(value=offset)

def convert_mesh_to_usd(stage, usd_internal_path, verts, faces, collision_approximation, static, articulation, physics_iter=(16, 1), mass=None, apply_debug_torque=False, debug_torque_value=50.0, texture=None, usd_internal_art_reference_path='/World', add_damping=False):
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]
    points = verts
    vertex_counts = np.ones(n_faces).astype(np.int32) * 3
    mesh = UsdGeom.Mesh.Define(stage, usd_internal_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    mesh.CreateExtentAttr([(-100, -100, -100), (100, 100, 100)])
    prim = stage.GetPrimAtPath(usd_internal_path)
    if texture is not None:
        vts = texture['vts']
        fts = texture['fts']
        texture_map_path = texture['texture_map_path']
        tex_coords = vts[fts.reshape(-1)].reshape(-1, 2)
        texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        texCoords.Set(Vt.Vec2fArray.FromNumpy(tex_coords))
        usd_mat_path = usd_internal_path + '_mat'
        material = UsdShade.Material.Define(stage, usd_mat_path)
        stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
        stInput.Set('st')
        pbrShader = UsdShade.Shader.Define(stage, f'{usd_mat_path}/PBRShader')
        pbrShader.CreateIdAttr('UsdPreviewSurface')
        if 'pbr_parameters' in texture:
            pbr_parameters = texture['pbr_parameters']
            roughness = pbr_parameters.get('roughness', 1.0)
            metallic = pbr_parameters.get('metallic', 0.0)
        else:
            roughness = 1.0
            metallic = 0.0
        pbrShader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
        pbrShader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
        pbrShader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Bool).Set(True)
        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), 'surface')
        stReader = UsdShade.Shader.Define(stage, f'{usd_mat_path}/stReader')
        stReader.CreateIdAttr('UsdPrimvarReader_float2')
        stReader.CreateInput('varname', Sdf.ValueTypeNames.Token).ConnectToSource(stInput)
        diffuseTextureSampler = UsdShade.Shader.Define(stage, f'{usd_mat_path}/diffuseTexture')
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_map_path)
        diffuseTextureSampler.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')
        mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(mesh).Bind(material)
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    if not static:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        if mass is not None:
            mass_api.CreateMassAttr(mass)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        ps_rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rigid_body.CreateSolverPositionIterationCountAttr(physics_iter[0])
        physx_rigid_body.CreateSolverVelocityIterationCountAttr(physics_iter[1])
    if articulation is not None:
        articulation_api = UsdPhysics.ArticulationRootAPI.Apply(prim)
        (rotate_axis_point_lower, rotate_axis_point_upper) = articulation
        axis_vector = np.array(rotate_axis_point_upper) - np.array(rotate_axis_point_lower)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        joint_path = usd_internal_path + '_joint'
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateAxisAttr('Z')
        joint.CreateBody0Rel().SetTargets([usd_internal_path])
        joint.CreateBody1Rel().SetTargets([usd_internal_art_reference_path])
        joint_pos = (np.array(rotate_axis_point_lower) + np.array(rotate_axis_point_upper)) / 2
        joint_prim = stage.GetPrimAtPath(joint_path)
        joint_xform = UsdGeom.Xformable(joint_prim)
        joint.CreateLocalPos0Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        if not np.allclose(axis_vector, [0, 0, 1]):
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, axis_vector)
            if np.linalg.norm(rotation_axis) > 1e-06:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, axis_vector), -1.0, 1.0))
                angle_degrees = np.degrees(angle)
                sin_half = np.sin(angle / 2)
                cos_half = np.cos(angle / 2)
                quat = Gf.Quatf(cos_half, sin_half * rotation_axis[0], sin_half * rotation_axis[1], sin_half * rotation_axis[2])
                joint.CreateLocalRot0Attr(quat)
                joint.CreateLocalRot1Attr(quat)
        if apply_debug_torque:
            print(f'Applying debug torque: {debug_torque_value}')
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, 'angular')
            drive_api.CreateTypeAttr('force')
            drive_api.CreateTargetVelocityAttr(debug_torque_value)
            drive_api.CreateStiffnessAttr(0.0)
            drive_api.CreateDampingAttr(10000.0)
            drive_api.CreateMaxForceAttr(1000.0)
            print('Debug torque applied - joint should rotate')
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    ps_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    ps_collision_api.CreateContactOffsetAttr(0.005)
    ps_collision_api.CreateRestOffsetAttr(0.001)
    ps_collision_api.CreateTorsionalPatchRadiusAttr(0.01)
    physx_rigid_body.CreateLinearDampingAttr(10.0)
    physx_rigid_body.CreateAngularDampingAttr(10.0)
    physx_rigid_body.CreateMaxLinearVelocityAttr(0.5)
    physx_rigid_body.CreateMaxAngularVelocityAttr(0.5)
    physx_rigid_body.CreateMaxDepenetrationVelocityAttr(50.0)
    if collision_approximation == 'sdf':
        physx_sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        physx_sdf.CreateSdfResolutionAttr(256)
        collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
        collider.CreateApproximationAttr('sdf')
    elif collision_approximation == 'convexDecomposition':
        convexdecomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
        collider.CreateApproximationAttr('convexDecomposition')
    mat = UsdPhysics.MaterialAPI.Apply(prim)
    mat.CreateDynamicFrictionAttr(1e+20)
    mat.CreateStaticFrictionAttr(1e+20)
    return stage

def convert_mesh_to_usd_simple(stage, usd_internal_path, verts, faces, collision_approximation, static, articulation, physics_iter=(255, 255), apply_debug_torque=False, debug_torque_value=50.0, texture=None):
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]
    points = verts
    vertex_counts = np.ones(n_faces).astype(np.int32) * 3
    mesh = UsdGeom.Mesh.Define(stage, usd_internal_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    mesh.CreateExtentAttr([(-100, -100, -100), (100, 100, 100)])
    prim = stage.GetPrimAtPath(usd_internal_path)
    if texture is not None:
        vts = texture['vts']
        fts = texture['fts']
        texture_map_path = texture['texture_map_path']
        tex_coords = vts[fts.reshape(-1)].reshape(-1, 2)
        texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        texCoords.Set(Vt.Vec2fArray.FromNumpy(tex_coords))
        usd_mat_path = usd_internal_path + '_mat'
        material = UsdShade.Material.Define(stage, usd_mat_path)
        stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
        stInput.Set('st')
        pbrShader = UsdShade.Shader.Define(stage, f'{usd_mat_path}/PBRShader')
        pbrShader.CreateIdAttr('UsdPreviewSurface')
        pbrShader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.0)
        pbrShader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
        pbrShader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Bool).Set(True)
        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), 'surface')
        stReader = UsdShade.Shader.Define(stage, f'{usd_mat_path}/stReader')
        stReader.CreateIdAttr('UsdPrimvarReader_float2')
        stReader.CreateInput('varname', Sdf.ValueTypeNames.Token).ConnectToSource(stInput)
        diffuseTextureSampler = UsdShade.Shader.Define(stage, f'{usd_mat_path}/diffuseTexture')
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_map_path)
        diffuseTextureSampler.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')
        mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(mesh).Bind(material)
    if not static:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    if articulation is not None:
        (rotate_axis_point_lower, rotate_axis_point_upper) = articulation
        axis_vector = np.array(rotate_axis_point_upper) - np.array(rotate_axis_point_lower)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        joint_path = usd_internal_path + '_joint'
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateAxisAttr('Z')
        joint.CreateBody0Rel().SetTargets([usd_internal_path])
        joint_pos = (np.array(rotate_axis_point_lower) + np.array(rotate_axis_point_upper)) / 2
        joint_prim = stage.GetPrimAtPath(joint_path)
        joint_xform = UsdGeom.Xformable(joint_prim)
        joint.CreateLocalPos0Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        if not np.allclose(axis_vector, [0, 0, 1]):
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, axis_vector)
            if np.linalg.norm(rotation_axis) > 1e-06:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, axis_vector), -1.0, 1.0))
                angle_degrees = np.degrees(angle)
                sin_half = np.sin(angle / 2)
                cos_half = np.cos(angle / 2)
                quat = Gf.Quatf(cos_half, sin_half * rotation_axis[0], sin_half * rotation_axis[1], sin_half * rotation_axis[2])
                joint.CreateLocalRot0Attr(quat)
                joint.CreateLocalRot1Attr(quat)
    UsdPhysics.CollisionAPI.Apply(prim)
    mat = UsdPhysics.MaterialAPI.Apply(prim)
    mat.CreateDynamicFrictionAttr(2.0)
    mat.CreateStaticFrictionAttr(2.0)
    return stage

def door_frame_to_usd(stage, usd_internal_path_door, usd_internal_path_door_frame, mesh_obj_door, mesh_obj_door_frame, articulation_door, texture_door, texture_door_frame, apply_debug_torque=False, debug_torque_value=50.0):
    """
    Create door and door frame USD objects with a revolute joint between them.
    
    Args:
        stage: USD stage
        usd_internal_path_door: USD path for the door
        usd_internal_path_door_frame: USD path for the door frame
        mesh_obj_door: Trimesh object for the door
        mesh_obj_door_frame: Trimesh object for the door frame
        articulation_door: Tuple of (rotate_axis_point_lower, rotate_axis_point_upper) in world coordinates
        texture_door: Texture info for door (can be None)
        texture_door_frame: Texture info for door frame (can be None)
        apply_debug_torque: Whether to apply debug torque for testing joint functionality
        debug_torque_value: Target velocity for debug torque (degrees per second)
    """
    door_verts = np.array(mesh_obj_door.vertices)
    door_faces = np.array(mesh_obj_door.faces)
    frame_verts = np.array(mesh_obj_door_frame.vertices)
    frame_faces = np.array(mesh_obj_door_frame.faces)
    n_frame_verts = frame_verts.shape[0]
    n_frame_faces = frame_faces.shape[0]
    frame_vertex_counts = np.ones(n_frame_faces).astype(np.int32) * 3
    frame_mesh = UsdGeom.Mesh.Define(stage, usd_internal_path_door_frame)
    frame_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(frame_verts))
    frame_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(frame_vertex_counts))
    frame_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(frame_faces))
    frame_mesh.CreateExtentAttr([(-100, -100, -100), (100, 100, 100)])
    frame_prim = stage.GetPrimAtPath(usd_internal_path_door_frame)
    if texture_door_frame is not None:
        vts = texture_door_frame['vts']
        fts = texture_door_frame['fts']
        texture_map_path = texture_door_frame['texture_map_path']
        tex_coords = vts[fts.reshape(-1)].reshape(-1, 2)
        texCoords = UsdGeom.PrimvarsAPI(frame_mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        texCoords.Set(Vt.Vec2fArray.FromNumpy(tex_coords))
        usd_mat_path = usd_internal_path_door_frame + '_mat'
        material = UsdShade.Material.Define(stage, usd_mat_path)
        stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
        stInput.Set('st')
        pbrShader = UsdShade.Shader.Define(stage, f'{usd_mat_path}/PBRShader')
        pbrShader.CreateIdAttr('UsdPreviewSurface')
        pbrShader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.0)
        pbrShader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
        pbrShader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Bool).Set(True)
        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), 'surface')
        stReader = UsdShade.Shader.Define(stage, f'{usd_mat_path}/stReader')
        stReader.CreateIdAttr('UsdPrimvarReader_float2')
        stReader.CreateInput('varname', Sdf.ValueTypeNames.Token).ConnectToSource(stInput)
        diffuseTextureSampler = UsdShade.Shader.Define(stage, f'{usd_mat_path}/diffuseTexture')
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_map_path)
        diffuseTextureSampler.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')
        frame_mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(frame_mesh).Bind(material)
    frame_mat = UsdPhysics.MaterialAPI.Apply(frame_prim)
    frame_mat.CreateDynamicFrictionAttr(2.0)
    frame_mat.CreateStaticFrictionAttr(2.0)
    n_door_verts = door_verts.shape[0]
    n_door_faces = door_faces.shape[0]
    door_vertex_counts = np.ones(n_door_faces).astype(np.int32) * 3
    door_mesh = UsdGeom.Mesh.Define(stage, usd_internal_path_door)
    door_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(door_verts))
    door_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(door_vertex_counts))
    door_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(door_faces))
    door_mesh.CreateExtentAttr([(-100, -100, -100), (100, 100, 100)])
    door_prim = stage.GetPrimAtPath(usd_internal_path_door)
    if texture_door is not None:
        vts = texture_door['vts']
        fts = texture_door['fts']
        texture_map_path = texture_door['texture_map_path']
        tex_coords = vts[fts.reshape(-1)].reshape(-1, 2)
        texCoords = UsdGeom.PrimvarsAPI(door_mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        texCoords.Set(Vt.Vec2fArray.FromNumpy(tex_coords))
        usd_mat_path = usd_internal_path_door + '_mat'
        material = UsdShade.Material.Define(stage, usd_mat_path)
        stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
        stInput.Set('st')
        pbrShader = UsdShade.Shader.Define(stage, f'{usd_mat_path}/PBRShader')
        pbrShader.CreateIdAttr('UsdPreviewSurface')
        pbrShader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.0)
        pbrShader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
        pbrShader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Bool).Set(True)
        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), 'surface')
        stReader = UsdShade.Shader.Define(stage, f'{usd_mat_path}/stReader')
        stReader.CreateIdAttr('UsdPrimvarReader_float2')
        stReader.CreateInput('varname', Sdf.ValueTypeNames.Token).ConnectToSource(stInput)
        diffuseTextureSampler = UsdShade.Shader.Define(stage, f'{usd_mat_path}/diffuseTexture')
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_map_path)
        diffuseTextureSampler.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')
        door_mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(door_mesh).Bind(material)
    UsdPhysics.CollisionAPI.Apply(door_prim)
    mass_api = UsdPhysics.MassAPI.Apply(door_prim)
    mass_api.CreateMassAttr(10.0)
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(door_prim)
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(door_prim)
    physx_rigid_body.CreateSolverPositionIterationCountAttr(255)
    physx_rigid_body.CreateSolverVelocityIterationCountAttr(255)
    physx_rigid_body.CreateLinearDampingAttr(10.0)
    physx_rigid_body.CreateAngularDampingAttr(10.0)
    physx_rigid_body.CreateMaxLinearVelocityAttr(0.5)
    physx_rigid_body.CreateMaxAngularVelocityAttr(0.5)
    physx_rigid_body.CreateMaxDepenetrationVelocityAttr(50.0)
    physx_sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(door_prim)
    physx_sdf.CreateSdfResolutionAttr(256)
    collider = UsdPhysics.MeshCollisionAPI.Apply(door_prim)
    collider.CreateApproximationAttr('sdf')
    door_mat = UsdPhysics.MaterialAPI.Apply(door_prim)
    door_mat.CreateDynamicFrictionAttr(2.0)
    door_mat.CreateStaticFrictionAttr(2.0)
    if articulation_door is not None:
        (rotate_axis_point_lower, rotate_axis_point_upper) = articulation_door
        axis_vector = np.array(rotate_axis_point_upper) - np.array(rotate_axis_point_lower)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        joint_path = usd_internal_path_door + '_hinge_joint'
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateAxisAttr('Z')
        joint.CreateBody0Rel().SetTargets([usd_internal_path_door])
        joint.CreateBody1Rel().SetTargets([usd_internal_path_door_frame])
        joint_pos = (np.array(rotate_axis_point_lower) + np.array(rotate_axis_point_upper)) / 2
        joint.CreateLocalPos0Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        if not np.allclose(axis_vector, [0, 0, 1]):
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, axis_vector)
            if np.linalg.norm(rotation_axis) > 1e-06:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, axis_vector), -1.0, 1.0))
                sin_half = np.sin(angle / 2)
                cos_half = np.cos(angle / 2)
                quat = Gf.Quatf(cos_half, sin_half * rotation_axis[0], sin_half * rotation_axis[1], sin_half * rotation_axis[2])
                joint.CreateLocalRot0Attr(quat)
                joint.CreateLocalRot1Attr(quat)
        joint.CreateLowerLimitAttr(-120.0)
        joint.CreateUpperLimitAttr(120.0)
        joint_prim = stage.GetPrimAtPath(joint_path)
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        if apply_debug_torque:
            print(f'Applying debug torque to door joint: {debug_torque_value}')
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, 'angular')
            drive_api.CreateTypeAttr('force')
            drive_api.CreateTargetVelocityAttr(debug_torque_value)
            drive_api.CreateStiffnessAttr(0.0)
            drive_api.CreateDampingAttr(100)
            drive_api.CreateMaxForceAttr(1000.0)
            print('Debug torque applied - door should rotate')
        else:
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, 'angular')
            drive_api.CreateTypeAttr('force')
            drive_api.CreateStiffnessAttr(0.0)
            drive_api.CreateDampingAttr(100.0)
            drive_api.CreateMaxForceAttr(1000.0)
    return stage
