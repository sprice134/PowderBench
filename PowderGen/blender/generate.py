import bpy
import glob
import os
import random
import math
import json
import tempfile
import numpy as np
import time
import sys
import argparse
import mathutils
from bpy_extras.object_utils import world_to_camera_view



# Defaults (can be overridden by command line)
simulation_frames = 250
camera_ortho_scale = 17.0
min_scene_visible_particles = 2
max_scene_generation_attempts = 2
max_camera_sampling_attempts = 8
scene_base_size_range = (2.0, 4.0)
base_camera_ortho_scale = 40.0
camera_height = 20.0
camera_center_jitter_fraction = 0.25
camera_sampling_modes = {
    "close": {
        "weight": 0.3,
        "zoom_factor_range": (0.15, 0.35),
        "min_visible_particles": 2,
    },
    "mid": {
        "weight": 0.4,
        "zoom_factor_range": (0.35, 0.7),
        "min_visible_particles": 6,
    },
    "wide": {
        "weight": 0.3,
        "zoom_factor_range": (0.7, 1.0),
        "min_visible_particles": 12,
    },
}

def parse_args():
    """Parses arguments passed after the '--' separator in Blender's command line."""
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SandAtlasHamburg', 
                        help='Powder folder name under --mesh-root')
    parser.add_argument('--mesh-root', type=str, required=True,
                        help='Directory of externally acquired, processed STL powder folders')
    parser.add_argument('--output-root', type=str, required=True,
                        help='Directory for generated powder scene folders')
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='CPU',
                        help='Cycles device; CPU works for a small smoke run')
    parser.add_argument('--preflight-only', action='store_true',
                        help='Import one source STL and validate geometry without simulation, rendering, or output files')
    parser.add_argument('--num_scenes', type=int, default=100, 
                        help='Number of scenes to generate')
    parser.add_argument('--start_id', type=int, default=1, 
                        help='ID of the first scene generated')
    parser.add_argument('--min_faces', type=int, default=50, 
                        help='Minimum number of faces for an STL to be used')
    parser.add_argument(
        '--dynamic_camera_zooming',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable per-scene dynamic camera crops after physics settling'
    )
    parser.add_argument(
        '--voxel_smoothing',
        type=float,
        default=0.0,
        help='Light smoothing factor applied once after import to reduce voxel stair-stepping'
    )
    parser.add_argument(
        '--voxel_smoothing_iterations',
        type=int,
        default=8,
        help='Number of smoothing iterations for light surface refinement'
    )
    parser.add_argument(
        '--voxel_remesh_size',
        type=float,
        default=0.0,
        help='Optional light voxel remesh size applied before smoothing to reduce stair-stepping'
    )
    return parser.parse_args(argv)

class MetadataTracker:
    def __init__(self):
        # Explicit initialization for linting/clarity
        self.steps_time = {}
        self.storage = {}
        self.start_times = {}
        self.reset()

    def reset(self):
        self.steps_time = {}
        self.storage = {}
        self.start_times = {}

    def start_step(self, step_name):
        self.start_times[step_name] = time.time()

    def end_step(self, step_name):
        if step_name in self.start_times:
            elapsed = time.time() - self.start_times[step_name]
            self.steps_time[step_name] = self.steps_time.get(step_name, 0) + elapsed

    def track_storage(self, name, path):
        if os.path.exists(path):
            self.storage[name] = os.path.getsize(path)

    def get_metadata(self):
        return {
            "time_per_step_seconds": self.steps_time,
            "storage_required_bytes": self.storage,
            "total_storage_bytes": sum(self.storage.values()),
            "total_time_seconds": sum(self.steps_time.values())
        }

    def save_to_json(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.get_metadata(), f, indent=4)


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clear_scene_and_data():
    """Aggressively clear all objects and datablocks to prevent memory leaks in loop."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Remove orphaned datablocks avoiding leaks over 100 loops
    for block in [bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.lights, bpy.data.cameras]:
        for item in block:
            block.remove(item)

def setup_scene():
    """Clears the scene and sets up a rigid body world and ground."""
    clear_scene_and_data()

    # Set up Rigid Body World
    if bpy.context.scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    bpy.context.scene.rigidbody_world.substeps_per_frame = 20
    bpy.context.scene.rigidbody_world.solver_iterations = 20
    bpy.context.scene.rigidbody_world.time_scale = 1.2
    
    # Set world background to black for SEM look
    world = bpy.context.scene.world
    if world and world.use_nodes:
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs[0].default_value = (0, 0, 0, 1)

    # Add Ground Plane
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "Ground"
    
    # Give the ground a black material so the background appears black
    black_mat = bpy.data.materials.new(name="BlackBackground")
    black_mat.use_nodes = True
    bg_bsdf = black_mat.node_tree.nodes.get('Principled BSDF')
    if bg_bsdf:
        bg_bsdf.inputs['Base Color'].default_value = (0, 0, 0, 1)
        bg_bsdf.inputs['Roughness'].default_value = 1.0 # purely diffuse black
    ground.data.materials.append(black_mat)
    
    # Add Passive Rigid Body to ground
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = 'PASSIVE'
    ground.rigid_body.collision_shape = 'BOX'
    ground.rigid_body.friction = 1.0
    ground.rigid_body.restitution = 0.0
    ground.rigid_body.use_margin = True
    ground.rigid_body.collision_margin = 0.001

def load_and_drop_particles(
    stl_files,
    num,
    base_size,
    min_faces,
    voxel_remesh_size,
    voxel_smoothing,
    voxel_smoothing_iterations
):
    """Loads random particles, filters by face count, and assigns rigid body physics."""
    valid_particles_count = 0
    attempts = 0
    max_attempts = num * 10  # Prevent infinite loops if many STLs are bad
    
    while valid_particles_count < num and attempts < max_attempts:
        attempts += 1
        stl_file = random.choice(stl_files)
        
        # Import STL
        bpy.ops.wm.stl_import(filepath=stl_file)
        obj = bpy.context.object

        # Ignore empty or invalid objects
        if max(obj.dimensions) == 0:
            bpy.ops.object.delete()
            continue
            
        # Check face count threshold
        face_count = len(obj.data.polygons)
        if face_count < min_faces:
            print(f"Skipping {os.path.basename(stl_file)}: too few faces ({face_count} < {min_faces})")
            bpy.ops.object.delete()
            continue

        valid_particles_count += 1
        instance_id = valid_particles_count

        # Stable traceable IDs / names
        obj.name = f"Particle_{instance_id:04d}"
        obj["instance_id"] = instance_id
        obj["source_stl"] = os.path.basename(stl_file)
        obj.pass_index = instance_id
        
        # Scale down so max dimension is relative to base_size with noise
        individual_size = base_size + random.gauss(0, base_size * 0.1)
        individual_size = max(0.5, individual_size)
        
        scale_factor = individual_size / max(obj.dimensions)
        obj.scale = (scale_factor, scale_factor, scale_factor)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        if voxel_remesh_size > 0.0:
            remesh_mod = obj.modifiers.new(name="VoxelRemesh", type='REMESH')
            remesh_mod.mode = 'VOXEL'
            remesh_mod.voxel_size = voxel_remesh_size
            remesh_mod.adaptivity = 0.0
            remesh_mod.use_smooth_shade = True
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=remesh_mod.name)

        if voxel_smoothing > 0.0:
            smooth_mod = obj.modifiers.new(name="VoxelSmooth", type='LAPLACIANSMOOTH')
            smooth_mod.lambda_factor = min(max(voxel_smoothing, 0.0), 1.0)
            smooth_mod.iterations = max(1, voxel_smoothing_iterations)
            if hasattr(smooth_mod, "use_volume_preserve"):
                smooth_mod.use_volume_preserve = True
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=smooth_mod.name)

        # Arrange in a random grid
        grid_size = math.ceil(math.sqrt(num))
        spacing = 4.0
        offset = (grid_size * spacing) / 2.0
        
        # Use (valid_particles_count - 1) for grid placement
        idx = valid_particles_count - 1
        row = idx // grid_size
        col = idx % grid_size
        
        x = (col * spacing) - offset + random.uniform(-1.5, 1.5)
        y = (row * spacing) - offset + random.uniform(-1.5, 1.5)
        z = random.uniform(25.0, 60.0)
        
        obj.location = (x, y, z)
        
        # Random initial rotation
        obj.rotation_euler = (random.uniform(0, 2*math.pi), 
                              random.uniform(0, 2*math.pi), 
                              random.uniform(0, 2*math.pi))
        
        # Create and assign a grayscale / SEM-like material
        mat = bpy.data.materials.new(name=f"SEM_{instance_id}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            # Simple gray material
            bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1)
            bsdf.inputs['Roughness'].default_value = 0.8
            if 'Specular' in bsdf.inputs:
                bsdf.inputs['Specular'].default_value = 0.1
            elif 'Specular IOR Level' in bsdf.inputs:
                bsdf.inputs['Specular IOR Level'].default_value = 0.1

        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        # Apply Active Rigid Body
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = 'ACTIVE'
        # CONVEX_HULL is faster and stable for sand particles
        obj.rigid_body.collision_shape = 'CONVEX_HULL'
        obj.rigid_body.mass = 1000.0
        obj.rigid_body.friction = 1.0
        obj.rigid_body.restitution = 0.0
        obj.rigid_body.linear_damping = 0.1
        obj.rigid_body.angular_damping = 0.2
        obj.rigid_body.use_margin = True
        # Tiny margin is necessary to prevent them from "floating" due to internal Blender padding
        obj.rigid_body.collision_margin = 0.001
        
        # Deactivation prevents endless micro-rocking on the ground
        obj.rigid_body.use_deactivation = True
        obj.rigid_body.use_start_deactivated = False
        obj.rigid_body.deactivate_linear_velocity = 0.2
        obj.rigid_body.deactivate_angular_velocity = 0.25
        
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        # Calculate the true Center of Mass based on the rigid body's actual volume/shape,
        # so heavy/larger sides properly offset the center of gravity.
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')

def setup_camera_and_lighting(camera_location, camera_ortho_scale):
    """Sets up an orthographic camera and sun lighting with a configurable top-down camera."""
    # Camera
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    # Default camera points down -Z, so rotation 0,0,0 looks straight down at the XY plane
    camera.rotation_euler = (0, 0, 0)
    bpy.context.scene.camera = camera
    
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = camera_ortho_scale
    camera.data.clip_start = 0.1
    camera.data.clip_end = 1000.0
    
    # Sun Light setup (Single light for SEM feel, usually illuminating top-down with slight angle)
    bpy.ops.object.light_add(type='SUN', location=(camera_location[0], camera_location[1], camera_height))
    light = bpy.context.object
    light.rotation_euler = (math.radians(15), math.radians(15), 0)
    light.data.energy = 5.0

def apply_camera_and_lighting(camera_location, camera_ortho_scale):
    """Apply a sampled camera crop while keeping lighting aligned with the camera region."""
    camera = bpy.context.scene.camera
    if camera is None:
        raise RuntimeError("Scene camera is not configured.")

    camera.location = camera_location
    camera.data.ortho_scale = camera_ortho_scale

    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            obj.location = (camera_location[0], camera_location[1], camera_height)

def configure_rendering(device='CPU'):
    """Configure Cycles, with CPU as the portable default."""
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = device
    if device == 'CPU':
        return
    
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.get_devices()
    
    # Try OptiX first for NVIDIA GPUs like H100, fallback to CUDA
    if hasattr(prefs, "compute_device_type"):
        for compute_type in ('OPTIX', 'CUDA'):
            try:
                prefs.compute_device_type = compute_type
                break
            except TypeError:
                pass
                
    for device in prefs.devices:
        device.use = True
        
def create_pass_materials():
    """Creates materials for Depth, Normals, and AO to be used as View Layer overrides."""
    # NORMALS
    mat_normal = bpy.data.materials.new(name="Override_Normal")
    mat_normal.use_nodes = True
    nodes = mat_normal.node_tree.nodes
    links = mat_normal.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    geom = nodes.new('ShaderNodeNewGeometry')
    add = nodes.new('ShaderNodeVectorMath')
    add.operation = 'ADD'
    add.inputs[1].default_value = (1.0, 1.0, 1.0)
    mul = nodes.new('ShaderNodeVectorMath')
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = (0.5, 0.5, 0.5)
    links.new(geom.outputs['True Normal'], add.inputs[0])
    links.new(add.outputs[0], mul.inputs[0])
    links.new(mul.outputs[0], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    # DEPTH
    mat_depth = bpy.data.materials.new(name="Override_Depth")
    mat_depth.use_nodes = True
    nodes = mat_depth.node_tree.nodes
    links = mat_depth.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    cam = nodes.new('ShaderNodeCameraData')
    links.new(cam.outputs['View Z Depth'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    # AO
    mat_ao = bpy.data.materials.new(name="Override_AO")
    mat_ao.use_nodes = True
    nodes = mat_ao.node_tree.nodes
    links = mat_ao.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    ao = nodes.new('ShaderNodeAmbientOcclusion')
    links.new(ao.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    return mat_normal, mat_depth, mat_ao

def get_particle_objects():
    """Returns all particle mesh objects sorted by instance ID."""
    particles = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name != "Ground" and "instance_id" in obj:
            particles.append(obj)
    particles.sort(key=lambda o: int(o["instance_id"]))
    return particles

def encode_instance_id_to_rgba(instance_id):
    """Encode instance ID into RGB so it can be decoded deterministically from an 8-bit PNG."""
    r = (instance_id & 255) / 255.0
    g = ((instance_id >> 8) & 255) / 255.0
    b = ((instance_id >> 16) & 255) / 255.0
    return (r, g, b, 1.0)

def prepare_instance_render_materials(single_object=None, binary_only=False, bit_plane=None):
    """
    Replace particle materials with flat emission materials for deterministic mask rendering.
    If single_object is provided, only that object is shown for amodal mask export.
    """
    scene = bpy.context.scene
    bpy.context.view_layer.material_override = None

    world = scene.world
    if world and getattr(world, 'node_tree', None):
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs['Strength'].default_value = 0.0

    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            obj.hide_render = True

    ground = bpy.data.objects.get("Ground")
    if ground is not None:
        ground.hide_render = True

    scene.cycles.filter_width = 0.01
    try:
        scene.view_settings.view_transform = 'Raw'
    except Exception:
        pass

    particle_objects = get_particle_objects()

    for obj in particle_objects:
        obj.hide_render = (single_object is not None and obj != single_object)

        mat = bpy.data.materials.new(name=f"Mask_{obj.name}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        for node in list(nodes):
            nodes.remove(node)

        emission = nodes.new(type='ShaderNodeEmission')
        output = nodes.new(type='ShaderNodeOutputMaterial')

        if bit_plane is not None:
            bit = (int(obj["instance_id"]) >> bit_plane) & 1
            v = 1.0 if bit else 0.0
            color = (v, v, v, 1.0)
        elif binary_only:
            color = (1.0, 1.0, 1.0, 1.0)
        else:
            color = encode_instance_id_to_rgba(int(obj["instance_id"]))

        emission.inputs['Color'].default_value = color
        emission.inputs['Strength'].default_value = 1.0
        links.new(emission.outputs['Emission'], output.inputs['Surface'])

        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

def get_rendered_image_rgba_from_file(tmp_png_path):
    """Load a rendered PNG from disk into an HxWx4 float32 numpy array."""
    img = bpy.data.images.load(tmp_png_path)
    try:
        width, height = img.size
        pixels = np.array(img.pixels[:], dtype=np.float32)
        rgba = pixels.reshape((height, width, 4))
        rgba = np.flipud(rgba)
        return rgba
    finally:
        bpy.data.images.remove(img)

def render_visible_instance_map(output_path):
    """
    Render a deterministic visible instance map and save as compressed uint8 .npz.
    Pixel value = instance_id, background = 0.
    """
    scene = bpy.context.scene
    world = scene.world
    ground = bpy.data.objects.get("Ground")

    old_view_transform = scene.view_settings.view_transform
    old_filter_width = scene.cycles.filter_width
    old_dither = scene.render.dither_intensity
    old_samples = scene.cycles.samples
    old_use_denoising = getattr(scene.cycles, "use_denoising", None)
    old_use_adaptive_sampling = getattr(scene.cycles, "use_adaptive_sampling", None)
    old_bg_strength = None
    bg_node = None

    if world and getattr(world, 'node_tree', None):
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            old_bg_strength = bg_node.inputs['Strength'].default_value

    old_light_states = {}
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            old_light_states[obj.name] = obj.hide_render

    particle_materials = {}
    particle_hide_render_states = {}
    for obj in get_particle_objects():
        particle_materials[obj.name] = list(obj.data.materials)
        particle_hide_render_states[obj.name] = obj.hide_render

    old_ground_hide_render = None
    if ground is not None:
        old_ground_hide_render = ground.hide_render

    fd, tmp_png_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)

    try:
        scene.render.filepath = tmp_png_path
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '8'
        scene.render.dither_intensity = 0.0
        scene.cycles.samples = 1
        if hasattr(scene.cycles, "use_denoising"):
            scene.cycles.use_denoising = False
        if hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = False

        max_instance_id = max((int(obj["instance_id"]) for obj in get_particle_objects()), default=0)
        if max_instance_id > 255:
            raise ValueError(f"visible instance map requires <=255 instance IDs, got {max_instance_id}")

        # Render 8 binary bit-planes to avoid RGB decode artifacts from color blending.
        visible_ids = np.zeros(
            (scene.render.resolution_y, scene.render.resolution_x),
            dtype=np.uint8
        )

        for bit in range(8):
            prepare_instance_render_materials(single_object=None, bit_plane=bit)
            bpy.ops.render.render(write_still=True)

            rgba = get_rendered_image_rgba_from_file(tmp_png_path)
            bitmask = (rgba[:, :, 0] >= 0.5).astype(np.uint8)
            visible_ids |= (bitmask << bit)

        # Suppress any phantom labels that can arise from edge blending.
        visible_ids[visible_ids > max_instance_id] = 0

        np.savez_compressed(output_path, visible_ids=visible_ids)
    finally:
        if os.path.exists(tmp_png_path):
            os.remove(tmp_png_path)

        scene.view_settings.view_transform = old_view_transform
        scene.cycles.filter_width = old_filter_width
        scene.render.dither_intensity = old_dither
        scene.cycles.samples = old_samples
        if old_use_denoising is not None:
            scene.cycles.use_denoising = old_use_denoising
        if old_use_adaptive_sampling is not None:
            scene.cycles.use_adaptive_sampling = old_use_adaptive_sampling

        if bg_node is not None and old_bg_strength is not None:
            bg_node.inputs['Strength'].default_value = old_bg_strength

        for obj in bpy.data.objects:
            if obj.type == 'LIGHT' and obj.name in old_light_states:
                obj.hide_render = old_light_states[obj.name]

        for obj in get_particle_objects():
            if obj.name in particle_hide_render_states:
                obj.hide_render = particle_hide_render_states[obj.name]

            original_materials = particle_materials.get(obj.name)
            if original_materials is not None:
                obj.data.materials.clear()
                for mat in original_materials:
                    obj.data.materials.append(mat)

        if ground is not None and old_ground_hide_render is not None:
            ground.hide_render = old_ground_hide_render

def count_visible_particles(visible_instance_map_path):
    """Count how many unique particle IDs are visible in the rendered instance map."""
    visible_ids = np.load(visible_instance_map_path)["visible_ids"]
    unique_ids = np.unique(visible_ids)
    return int(np.count_nonzero(unique_ids))

def get_object_xy_bounds(obj):
    """Return world-space XY bounds for an object's bounding box."""
    world_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    xs = [corner.x for corner in world_corners]
    ys = [corner.y for corner in world_corners]
    return min(xs), max(xs), min(ys), max(ys)

def get_central_particle(particle_objects):
    """Pick the particle whose XY location is closest to the scene's XY centroid."""
    if not particle_objects:
        return None

    mean_x = sum(obj.location.x for obj in particle_objects) / len(particle_objects)
    mean_y = sum(obj.location.y for obj in particle_objects) / len(particle_objects)
    return min(
        particle_objects,
        key=lambda obj: (obj.location.x - mean_x) ** 2 + (obj.location.y - mean_y) ** 2
    )

def choose_camera_sampling_mode():
    """Randomly select one of the explicit SEM camera regimes."""
    mode_names = list(camera_sampling_modes.keys())
    weights = [camera_sampling_modes[name]["weight"] for name in mode_names]
    return random.choices(mode_names, weights=weights, k=1)[0]

def sample_zoom_camera_for_scene(
    particle_objects,
    base_ortho_scale,
    resolution_x,
    resolution_y,
    mode_name,
):
    """
    Sample a top-down camera centered on a local particle cluster instead of scaling particles up.
    Returns a camera location and an orthographic scale that remain fully traceable.
    """
    if not particle_objects:
        return (0.0, 0.0, camera_height), base_ortho_scale

    mode_config = camera_sampling_modes[mode_name]

    aspect = resolution_x / resolution_y
    central_particle = get_central_particle(particle_objects)
    min_x, max_x, min_y, max_y = get_object_xy_bounds(central_particle)

    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    span_x = max(max_x - min_x, 0.5)
    span_y = max(max_y - min_y, 0.5)

    min_object_scale = max(4.0, span_x * 1.5, span_y * aspect * 1.5)
    random_zoom_scale = base_ortho_scale * random.uniform(*mode_config["zoom_factor_range"])
    ortho_scale = min(base_ortho_scale, max(min_object_scale, random_zoom_scale))

    max_jitter_x = ortho_scale * camera_center_jitter_fraction
    max_jitter_y = (ortho_scale / aspect) * camera_center_jitter_fraction
    center_x += random.uniform(-max_jitter_x, max_jitter_x)
    center_y += random.uniform(-max_jitter_y, max_jitter_y)

    return (center_x, center_y, camera_height), ortho_scale

def render_amodal_masks(output_path):
    """
    Generate one binary amodal mask per particle directly from projected geometry
    and save as packed-bit compressed .npz.
    Ordering matches scene_meta.json and instance_id ordering.
    """
    particle_objects = get_particle_objects()
    scene = bpy.context.scene
    camera = scene.camera

    width = scene.render.resolution_x
    height = scene.render.resolution_y
    amodal_stack = np.zeros((len(particle_objects), height, width), dtype=np.uint8)

    for idx, obj in enumerate(particle_objects):
        print(f"Amodal projection {idx + 1}/{len(particle_objects)}: {obj.name}")
        amodal_stack[idx] = project_object_to_amodal_mask(
            obj=obj,
            scene=scene,
            camera=camera,
            width=width,
            height=height
        )

    amodal_packed = np.packbits(amodal_stack, axis=-1)

    np.savez_compressed(
        output_path,
        masks=amodal_packed,
        original_shape=np.array(amodal_stack.shape, dtype=np.int32)
    )

def save_scene_meta(output_path, visible_instance_map_path, amodal_masks_path, obj_path):
    """Save per-scene metadata tying instance IDs, object names, OBJ names, and mask indices together."""
    particle_objects = get_particle_objects()

    particles_meta = []
    for stack_index, obj in enumerate(particle_objects):
        instance_id = int(obj["instance_id"])
        particles_meta.append({
            "instance_id": instance_id,
            "object_name": obj.name,
            "obj_name": obj.name,
            "source_stl": obj.get("source_stl", ""),
            "visible_instance_value": instance_id,
            "amodal_stack_index": stack_index
        })

    meta = {
        "visible_instance_map": os.path.basename(visible_instance_map_path),
        "amodal_masks": os.path.basename(amodal_masks_path),
        "scene_obj": os.path.basename(obj_path),
        "num_particles": len(particles_meta),
        "particles": particles_meta
    }

    with open(output_path, 'w') as f:
        json.dump(meta, f, indent=4)

def export_3d_scene(output_path):
    """Exports the fully assembled and settled 3D scene as an OBJ file, decimating first to save space."""
    # Ensure all objects are selected and their transformations are applied from the physics
    bpy.ops.object.select_all(action='SELECT')
    
    # Apply a decimate modifier to all particle meshes to reduce file size
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH' and obj.name != "Ground":
            # Add decimate modifier
            mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
            mod.ratio = 0.005  # Reduce polygon count to 5%
            
            # Apply the modifier immediately so it's baked into the export
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier="Decimate")
            
    # Reselect all in case active object changes messed with selection
    bpy.ops.object.select_all(action='SELECT')
    
    bpy.ops.wm.obj_export(filepath=output_path, export_selected_objects=True, export_materials=False)

def ndc_to_pixel(ndc_x, ndc_y, width, height):
    """
    Convert normalized device coordinates from world_to_camera_view()
    into image pixel coordinates.
    """
    px = ndc_x * (width - 1)
    py = (1.0 - ndc_y) * (height - 1)
    return px, py

def rasterize_triangle(mask, p0, p1, p2):
    """
    Rasterize a single 2D triangle into a binary mask using pixel-center sampling.
    mask: HxW uint8 array
    p0, p1, p2: (x, y) float pixel coordinates
    """
    height, width = mask.shape

    min_x = max(0, int(math.floor(min(p0[0], p1[0], p2[0]))))
    max_x = min(width - 1, int(math.ceil(max(p0[0], p1[0], p2[0]))))
    min_y = max(0, int(math.floor(min(p0[1], p1[1], p2[1]))))
    max_y = min(height - 1, int(math.ceil(max(p0[1], p1[1], p2[1]))))

    if min_x > max_x or min_y > max_y:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(xs, ys)

    def edge(a, b, x, y):
        return (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])

    area = edge(p0, p1, p2[0], p2[1])
    if abs(area) < 1e-8:
        return

    w0 = edge(p1, p2, xx, yy)
    w1 = edge(p2, p0, xx, yy)
    w2 = edge(p0, p1, xx, yy)

    if area > 0:
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    else:
        inside = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)

    mask[min_y:max_y + 1, min_x:max_x + 1] |= inside.astype(np.uint8)

def project_object_to_amodal_mask(obj, scene, camera, width, height):
    """
    Build an amodal mask directly from the evaluated mesh geometry by:
    1) getting the evaluated mesh from the depsgraph
    2) projecting vertices into image space
    3) rasterizing projected triangles into a binary mask
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    mask = np.zeros((height, width), dtype=np.uint8)

    try:
        mesh.calc_loop_triangles()

        world_matrix = eval_obj.matrix_world
        projected_verts = []

        for v in mesh.vertices:
            world_co = world_matrix @ v.co
            ndc = world_to_camera_view(scene, camera, world_co)
            px, py = ndc_to_pixel(ndc.x, ndc.y, width, height)
            projected_verts.append((px, py))

        for tri in mesh.loop_triangles:
            p0 = projected_verts[tri.vertices[0]]
            p1 = projected_verts[tri.vertices[1]]
            p2 = projected_verts[tri.vertices[2]]
            rasterize_triangle(mask, p0, p1, p2)

        return mask
    finally:
        eval_obj.to_mesh_clear()

def main():
    args = parse_args()
    
    # Dynamic paths based on dataset argument
    stl_dir = os.path.join(args.mesh_root, args.dataset)
    base_output_dir = os.path.join(args.output_root, args.dataset)
    num_scenes = args.num_scenes
    start_id = args.start_id

    tracker = MetadataTracker()
    
    stl_files = glob.glob(os.path.join(stl_dir, "*.stl"))
    if not stl_files:
        print(f"No STLs found in {stl_dir}")
        return 1
    if args.preflight_only:
        chosen = sorted(stl_files)[0]
        bpy.ops.wm.stl_import(filepath=chosen)
        obj = bpy.context.object
        face_count = len(obj.data.polygons) if obj is not None else 0
        dimensions = [float(value) for value in obj.dimensions] if obj is not None else []
        valid = face_count >= args.min_faces and dimensions and max(dimensions) > 0
        print(json.dumps({"stage": "mesh_import_preflight", "powder": args.dataset,
                          "input_stl": os.path.basename(chosen), "faces": face_count,
                          "dimensions": dimensions, "valid": bool(valid)}))
        return 0 if valid else 1

    configure_rendering(args.device)
        
    create_directory_if_not_exists(base_output_dir)
    
    # Save camera sampling configuration to the dataset root folder
    config = {
        "camera_type": "ORTHO",
        "camera_location_mode": "per_scene_central_particle_zoom_regimes",
        "camera_base_location": [0.0, 0.0, camera_height],
        "camera_rotation_euler": [0.0, 0.0, 0.0],
        "camera_base_ortho_scale": base_camera_ortho_scale,
        "camera_sampling_modes": {
            mode_name: {
                "weight": mode_config["weight"],
                "zoom_factor_range": list(mode_config["zoom_factor_range"]),
                "min_visible_particles": mode_config["min_visible_particles"],
            }
            for mode_name, mode_config in camera_sampling_modes.items()
        },
        "camera_center_jitter_fraction": camera_center_jitter_fraction,
        "image_resolution": [1024, 768],
        "min_scene_visible_particles": min_scene_visible_particles,
        "max_camera_sampling_attempts": max_camera_sampling_attempts,
        "dynamic_camera_zooming": args.dynamic_camera_zooming,
        "voxel_remesh_size": args.voxel_remesh_size,
        "voxel_smoothing": args.voxel_smoothing,
        "voxel_smoothing_iterations": args.voxel_smoothing_iterations
    }
    
    config_path = os.path.join(base_output_dir, "camera_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Dataset: {args.dataset} | Scenes: {num_scenes} | Starting ID: {start_id}")
    print(f"Saved global camera configuration to {config_path} (Base scale: {base_camera_ortho_scale})")
    
    for scene_idx in range(start_id, start_id + num_scenes):
        print(f"\n==========================================")
        print(f"Generating Scene {scene_idx} of {num_scenes}...")
        
        # Create subfolder for this scene
        scene_dir = os.path.join(base_output_dir, f"scene_{scene_idx:03d}")
        create_directory_if_not_exists(scene_dir)
        
        # Formulate save paths
        output_path = os.path.join(scene_dir, "snapshot.png")
        normals_output_path = os.path.join(scene_dir, "normals.png")
        depth_output_path = os.path.join(scene_dir, "depth.exr")
        ao_output_path = os.path.join(scene_dir, "ao.png")
        visible_instance_output_path = os.path.join(scene_dir, "visible_instances.npz")
        amodal_output_path = os.path.join(scene_dir, "amodal_masks.npz")
        obj_output_path = os.path.join(scene_dir, "assembled_scene.obj")
        scene_camera_path = os.path.join(scene_dir, "camera_data.json")
        scene_meta_path = os.path.join(scene_dir, "scene_meta.json")
        
        # Determine a random number of particles for this scene
        current_particles = random.randint(20, 100)

        # Determine a random base size for all particles in this scene.
        # Wider scales help create more zoomed-in compositions without changing the camera.
        scene_base_size = random.uniform(*scene_base_size_range)

        visible_particle_count = 0
        scene_generated = False
        selected_camera_location = (0.0, 0.0, camera_height)
        selected_camera_scale = base_camera_ortho_scale
        selected_camera_mode = "fixed"

        for attempt_idx in range(1, max_scene_generation_attempts + 1):
            tracker.reset()

            setup_scene()
            mat_normal, mat_depth, mat_ao = create_pass_materials()

            print(
                f"Loading and dropping {current_particles} particles with base size "
                f"{scene_base_size:.2f} (attempt {attempt_idx}/{max_scene_generation_attempts}, "
                f"Min faces: {args.min_faces}, Remesh size: {args.voxel_remesh_size:.4f}, "
                f"Voxel smoothing: {args.voxel_smoothing:.3f}, "
                f"Iterations: {args.voxel_smoothing_iterations})..."
            )
            tracker.start_step("mesh_loading_and_physics_setup")
            load_and_drop_particles(
                stl_files,
                current_particles,
                scene_base_size,
                args.min_faces,
                args.voxel_remesh_size,
                args.voxel_smoothing,
                args.voxel_smoothing_iterations
            )
            tracker.end_step("mesh_loading_and_physics_setup")

            print("Setting up camera and lighting...")
            setup_camera_and_lighting((0.0, 0.0, camera_height), base_camera_ortho_scale)

            bpy.context.scene.render.resolution_x = 1024
            bpy.context.scene.render.resolution_y = 768
            # Get samples for GFLOPs estimation
            render_samples = bpy.context.scene.cycles.samples

            print(f"Simulating physics for {simulation_frames} frames...")
            tracker.start_step("physics_simulation")
            # Go to frame 1 to start simulation cleanly
            bpy.context.scene.frame_set(1)
            for frame in range(1, simulation_frames + 1):
                bpy.context.scene.frame_set(frame)
            tracker.end_step("physics_simulation")

            scene = bpy.context.scene
            scene.view_settings.view_transform = 'Filmic'
            scene.cycles.filter_width = 1.5

            world = scene.world
            if world and getattr(world, 'node_tree', None):
                bg_node = world.node_tree.nodes.get('Background')
                if bg_node:
                    bg_node.inputs['Strength'].default_value = 1.0

            particle_objects = get_particle_objects()
            if args.dynamic_camera_zooming:
                scene_camera_mode = choose_camera_sampling_mode()
                mode_config = camera_sampling_modes[scene_camera_mode]
                print("Sampling zoomed camera views for scene validation...")
                for camera_attempt_idx in range(1, max_camera_sampling_attempts + 1):
                    camera_location, sampled_camera_scale = sample_zoom_camera_for_scene(
                        particle_objects=particle_objects,
                        base_ortho_scale=base_camera_ortho_scale,
                        resolution_x=bpy.context.scene.render.resolution_x,
                        resolution_y=bpy.context.scene.render.resolution_y,
                        mode_name=scene_camera_mode,
                    )
                    apply_camera_and_lighting(camera_location, sampled_camera_scale)

                    tracker.start_step("render_visible_masks")
                    render_visible_instance_map(visible_instance_output_path)
                    tracker.end_step("render_visible_masks")
                    tracker.track_storage("visible_instance_map", visible_instance_output_path)

                    visible_particle_count = count_visible_particles(visible_instance_output_path)
                    print(
                        f"Camera sample {camera_attempt_idx}/{max_camera_sampling_attempts} "
                        f"({scene_camera_mode}) found "
                        f"{visible_particle_count} visible particle(s) at location "
                        f"({camera_location[0]:.2f}, {camera_location[1]:.2f}, {camera_location[2]:.2f}) "
                        f"with ortho scale {sampled_camera_scale:.2f}."
                    )

                    if visible_particle_count >= mode_config["min_visible_particles"]:
                        selected_camera_location = camera_location
                        selected_camera_scale = sampled_camera_scale
                        selected_camera_mode = scene_camera_mode
                        scene_generated = True
                        break
            else:
                print("Using fixed base camera for scene validation...")
                selected_camera_location = (0.0, 0.0, camera_height)
                selected_camera_scale = base_camera_ortho_scale
                selected_camera_mode = "fixed"
                apply_camera_and_lighting(selected_camera_location, selected_camera_scale)

                tracker.start_step("render_visible_masks")
                render_visible_instance_map(visible_instance_output_path)
                tracker.end_step("render_visible_masks")
                tracker.track_storage("visible_instance_map", visible_instance_output_path)

                visible_particle_count = count_visible_particles(visible_instance_output_path)
                print(
                    f"Fixed camera validation found {visible_particle_count} visible particle(s) "
                    f"at ortho scale {selected_camera_scale:.2f}."
                )

                if visible_particle_count >= min_scene_visible_particles:
                    scene_generated = True

            if scene_generated:
                break

            if args.dynamic_camera_zooming:
                print(
                    f"Scene {scene_idx} did not yield a sufficiently populated {scene_camera_mode} crop after "
                    f"{max_camera_sampling_attempts} camera samples. Re-simulating with the same "
                    f"particle scale distribution."
                )
            else:
                print(
                    f"Scene {scene_idx} did not yield enough visible particles with the fixed camera. "
                    f"Re-simulating with the same particle scale distribution."
                )

        if not scene_generated:
            print(
                f"Skipping scene {scene_idx}: could not find a zoomed view with at least "
                f"{min_scene_visible_particles} visible particles after {max_scene_generation_attempts} "
                f"scene attempts."
            )
            tracker.reset()
            print(f"==========================================\n")
            continue

        apply_camera_and_lighting(selected_camera_location, selected_camera_scale)

        print("Simulation complete. Rendering snapshot, Depth, Normals, and AO...")

        # 1. Snapshot
        tracker.start_step("render_snapshot")
        bpy.context.view_layer.material_override = None
        bpy.context.scene.render.filepath = output_path
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_depth = '8'
        bpy.ops.render.render(write_still=True)
        tracker.end_step("render_snapshot")
        tracker.track_storage("snapshot", output_path)

        # 2. Normals
        tracker.start_step("render_normals")
        bpy.context.view_layer.material_override = mat_normal
        bpy.context.scene.render.filepath = normals_output_path
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_depth = '16'
        bpy.ops.render.render(write_still=True)
        tracker.end_step("render_normals")
        tracker.track_storage("normals", normals_output_path)

        # 3. Depth
        tracker.start_step("render_depth")
        bpy.context.view_layer.material_override = mat_depth
        bpy.context.scene.render.filepath = depth_output_path
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        bpy.context.scene.render.image_settings.color_depth = '32'
        bpy.ops.render.render(write_still=True)
        tracker.end_step("render_depth")
        tracker.track_storage("depth", depth_output_path)

        # 4. AO
        tracker.start_step("render_ao")
        bpy.context.view_layer.material_override = mat_ao
        bpy.context.scene.render.filepath = ao_output_path
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_depth = '8'
        bpy.ops.render.render(write_still=True)
        tracker.end_step("render_ao")
        tracker.track_storage("ao", ao_output_path)

        # Clear override
        bpy.context.view_layer.material_override = None

        print(f"Render passes successfully saved to: {scene_dir}")
        print(f"Visible instance map successfully saved to: {visible_instance_output_path}")

        print("Rendering amodal masks...")
        tracker.start_step("render_amodal_masks")
        render_amodal_masks(amodal_output_path)
        tracker.end_step("render_amodal_masks")
        tracker.track_storage("amodal_masks", amodal_output_path)
        print(f"Amodal masks successfully saved to: {amodal_output_path}")

        print("Exporting fully assembled 3D scene data (.obj)...")
        tracker.start_step("export_3d_scene")
        export_3d_scene(obj_output_path)
        tracker.end_step("export_3d_scene")
        tracker.track_storage("scene_obj", obj_output_path)
        print(f"3D Scene successfully exported to: {obj_output_path}")

        print("Saving scene metadata...")
        save_scene_meta(scene_meta_path, visible_instance_output_path, amodal_output_path, obj_output_path)
        print(f"Scene metadata successfully saved to: {scene_meta_path}")

        # Save generation metadata
        metadata_json_path = os.path.join(scene_dir, "generation_metadata.json")
        tracker.save_to_json(metadata_json_path)
        print(f"Generation metadata successfully saved to: {metadata_json_path}")
        tracker.reset() # Reset for next scene
        
        # Extract and save Camera Intrinsics and Extrinsics for SAM3D / NeRF / SFM pipelines
        cam = bpy.context.scene.camera
        
        # The 4x4 transformation matrix (Rotation/Translation) of the camera in world space
        extrinsics_world = cam.matrix_world
        extrinsics_list = [list(row) for row in extrinsics_world]
        
        # Intrinsics are trickier for Orthographic cameras vs Perspective.
        # For Orthographic, the most important parameters are the Ortho Scale and the Render Resolution.
        scene_cam_data = {
            "camera_type": "ORTHO",
            "camera_mode": selected_camera_mode,
            "resolution": [1024, 768],
            "location": list(cam.location),
            "ortho_scale": cam.data.ortho_scale,
            "clip_start": cam.data.clip_start,
            "clip_end": cam.data.clip_end,
            "matrix_world_extrinsics": extrinsics_list
        }
        
        with open(scene_camera_path, 'w') as f:
            json.dump(scene_cam_data, f, indent=4)
            
        print(f"Camera SAM3D matrix successfully saved to: {scene_camera_path}")
        
        print(f"==========================================\n")

if __name__ == "__main__":
    raise SystemExit(main())
