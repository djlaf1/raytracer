"""
Rasterizer
Copyright (c) 2025 Leighton Ferguson
Licensed under the MIT License
"""

import pygame
import moderngl
import numpy as np
from pywavefront import Wavefront
import os

# ---------------------------
# Helper functions
# ---------------------------
def normalize(v):
    """Normalize a 3D vector safely."""
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def perspective(fovy, aspect, near, far):
    """Create a perspective projection matrix."""
    f = 1.0 / np.tan(fovy / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype='f4')

def look_at(eye, target, up):
    """Create a view matrix for a camera looking at a target."""
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    
    mat = np.eye(4, dtype='f4')
    mat[0, :3] = s
    mat[1, :3] = u
    mat[2, :3] = -f
    mat[0, 3] = -np.dot(s, eye)
    mat[1, 3] = -np.dot(u, eye)
    mat[2, 3] = np.dot(f, eye)
    return mat

# ---------------------------
# OBJ Loader
# ---------------------------
def load_obj(obj_path):
    """Load vertices and faces from an OBJ file using pywavefront."""
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")
    
    scene = Wavefront(obj_path, collect_faces=True, parse=True)
    
    vertices = np.array(scene.vertices, dtype='f4')
    
    faces = []
    for name, mesh in scene.meshes.items():
        for f in mesh.faces:
            faces.extend(f)
    faces = np.array(faces, dtype='i4')
    
    return vertices, faces

# ---------------------------
# Menu system
# ---------------------------
def obj_menu(screen, clock, obj_folder="OBJ_files"):
    """Display a simple menu to select OBJ files from a folder."""
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 30)
    
    obj_files = [f for f in os.listdir(obj_folder) if f.lower().endswith(".obj")]
    if not obj_files:
        raise FileNotFoundError(f"No OBJ files found in {obj_folder}")
    
    selected_index = 0
    menu_active = True
    
    while menu_active:
        screen.fill((20, 20, 20))
        title = font.render("Select a room to load:", True, (255, 255, 255))
        screen.blit(title, (50, 50))
        
        for i, f in enumerate(obj_files):
            color = (255, 255, 0) if i == selected_index else (200, 200, 200)
            text_surface = font.render(f"{i+1}. {f}", True, color)
            screen.blit(text_surface, (100, 150 + i * 40))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_index = (selected_index - 1) % len(obj_files)
                elif event.key == pygame.K_DOWN:
                    selected_index = (selected_index + 1) % len(obj_files)
                elif event.key == pygame.K_RETURN:
                    menu_active = False
        
        pygame.display.flip()
        clock.tick(30)
    
    return os.path.join(obj_folder, obj_files[selected_index])

# ---------------------------
# Main program
# ---------------------------
def main():
    pygame.init()
    WIDTH, HEIGHT = 1280, 720
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("ModernGL GPU Raycaster")
    clock = pygame.time.Clock()
    
    # ---------------------------
    # Menu: select OBJ
    # ---------------------------
    obj_path = obj_menu(screen, clock)
    
    try:
        vertices, faces = load_obj(obj_path)
    except Exception as e:
        print(f"Error loading OBJ: {e}")
        return
    
    # ---------------------------
    # ModernGL setup
    # ---------------------------
    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    
    vbo = ctx.buffer(vertices.tobytes())
    ibo = ctx.buffer(faces.tobytes())
    
    prog = ctx.program(
        vertex_shader='''
        #version 330
        in vec3 in_vert;
        uniform mat4 MVP;
        void main() {
            gl_Position = MVP * vec4(in_vert,1.0);
        }
        ''',
        fragment_shader='''
        #version 330
        out vec4 f_color;
        void main() {
            f_color = vec4(0.8,0.7,0.6,1.0);
        }
        '''
    )
    
    vao = ctx.vertex_array(prog, [(vbo, '3f', 'in_vert')], ibo)
    
    # ---------------------------
    # Camera
    # ---------------------------
    cam_pos = np.array([0.0, 1.0, 4.0], dtype='f4')
    yaw, pitch = np.pi, 0.0
    MOVE_SPEED = 0.1
    LOOK_SPEED = 0.003
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.MOUSEMOTION:
                mx, my = event.rel
                yaw -= mx * LOOK_SPEED
                pitch += my * LOOK_SPEED
                pitch = np.clip(pitch, -np.pi/2, np.pi/2)
        
        # Movement
        keys = pygame.key.get_pressed()
        forward = np.array([np.sin(yaw), 0, np.cos(yaw)], dtype='f4')
        right = normalize(np.cross(forward, [0,1,0]))
        if keys[pygame.K_w]: cam_pos += forward * MOVE_SPEED
        if keys[pygame.K_s]: cam_pos -= forward * MOVE_SPEED
        if keys[pygame.K_a]: cam_pos -= right * MOVE_SPEED
        if keys[pygame.K_d]: cam_pos += right * MOVE_SPEED
        if keys[pygame.K_q]: cam_pos[1] += MOVE_SPEED
        if keys[pygame.K_e]: cam_pos[1] -= MOVE_SPEED
        
        # Render
        ctx.clear(0.2, 0.25, 0.35)
        aspect = WIDTH / HEIGHT
        proj = perspective(np.radians(60), aspect, 0.1, 100.0)
        target = cam_pos + np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
        view = look_at(cam_pos, target, np.array([0,1,0], dtype='f4'))
        MVP = proj @ view
        prog['MVP'].write(MVP.astype('f4').tobytes())
        vao.render()
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()
