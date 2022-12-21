import glfw
import OpenGL
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import numpy as np
from ctypes import c_void_p
from PIL import Image
from math import sin, cos
import glm
import os

vertex_shader_source = """
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 in_tex_coords;

out vec2 out_tex_coords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0f);
    out_tex_coords = in_tex_coords;
}
"""

fragment_shader_source = """
#version 330 core

in vec2 out_tex_coords;

out vec4 FragColor;
uniform sampler2D texture1;
uniform sampler2D texture2;
uniform float mix_ratio;

void main() {
    FragColor = mix(texture(texture1, out_tex_coords), texture(texture2, out_tex_coords), mix_ratio);
}
"""

def main():
    if not glfw.init():
        return -1

    print(f"pyopengl version: {OpenGL.__version__}")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    window = glfw.create_window(800, 600, "learn OpenGL", None, None)
    if window is None:
        print("Failed to create glfw window")
        glfw.terminate()
        return -1

    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, on_resize)

    # z-buffer
    gl.glEnable(gl.GL_DEPTH_TEST)

    vertices = np.array([
     # positions      tex_coords
    -0.5, -0.5, -0.5,  0.0, 0.0,
     0.5, -0.5, -0.5,  1.0, 0.0,
     0.5,  0.5, -0.5,  1.0, 1.0,
     0.5,  0.5, -0.5,  1.0, 1.0,
    -0.5,  0.5, -0.5,  0.0, 1.0,
    -0.5, -0.5, -0.5,  0.0, 0.0,

    -0.5, -0.5,  0.5,  0.0, 0.0,
     0.5, -0.5,  0.5,  1.0, 0.0,
     0.5,  0.5,  0.5,  1.0, 1.0,
     0.5,  0.5,  0.5,  1.0, 1.0,
    -0.5,  0.5,  0.5,  0.0, 1.0,
    -0.5, -0.5,  0.5,  0.0, 0.0,

    -0.5,  0.5,  0.5,  1.0, 0.0,
    -0.5,  0.5, -0.5,  1.0, 1.0,
    -0.5, -0.5, -0.5,  0.0, 1.0,
    -0.5, -0.5, -0.5,  0.0, 1.0,
    -0.5, -0.5,  0.5,  0.0, 0.0,
    -0.5,  0.5,  0.5,  1.0, 0.0,

     0.5,  0.5,  0.5,  1.0, 0.0,
     0.5,  0.5, -0.5,  1.0, 1.0,
     0.5, -0.5, -0.5,  0.0, 1.0,
     0.5, -0.5, -0.5,  0.0, 1.0,
     0.5, -0.5,  0.5,  0.0, 0.0,
     0.5,  0.5,  0.5,  1.0, 0.0,

    -0.5, -0.5, -0.5,  0.0, 1.0,
     0.5, -0.5, -0.5,  1.0, 1.0,
     0.5, -0.5,  0.5,  1.0, 0.0,
     0.5, -0.5,  0.5,  1.0, 0.0,
    -0.5, -0.5,  0.5,  0.0, 0.0,
    -0.5, -0.5, -0.5,  0.0, 1.0,

    -0.5,  0.5, -0.5,  0.0, 1.0,
     0.5,  0.5, -0.5,  1.0, 1.0,
     0.5,  0.5,  0.5,  1.0, 0.0,
     0.5,  0.5,  0.5,  1.0, 0.0,
    -0.5,  0.5,  0.5,  0.0, 0.0,
    -0.5,  0.5, -0.5,  0.0, 1.0
    ], dtype=np.float32)
    vertices_size = vertices.size * vertices.itemsize

    shader = shaders.compileProgram(shaders.compileShader(vertex_shader_source, gl.GL_VERTEX_SHADER),
                                    shaders.compileShader(fragment_shader_source, gl.GL_FRAGMENT_SHADER))
    gl.glUseProgram(shader)


    # vao
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    # vbo
    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_size, vertices, gl.GL_STATIC_DRAW)

    # position
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 5 * vertices.itemsize, c_void_p(0))
    gl.glEnableVertexAttribArray(0)

    # texture
    gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 5 * vertices.itemsize, c_void_p(3 * vertices.itemsize))
    gl.glEnableVertexAttribArray(1)

    # texture binding - 1
    texture_1 = gl.glGenTextures(1)
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_1)
    # texture wrapping
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    # texture filtering
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    # texture img load
    img = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "resources", "fox.png"))
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.width, img.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img.tobytes())
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

    # texture binding - 2
    texture_2 = gl.glGenTextures(1)
    gl.glActiveTexture(gl.GL_TEXTURE1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_2)
    # texture wrapping
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    # texture filtering
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    # texture img load
    img = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "resources", "logo_bg_removed.png"))
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.width, img.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img.tobytes())
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

    # wireframe mode
    # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

    # fill mode
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
    
    gl.glClearColor(0.2, 0.3, 0.3, 1.0)

    # texture shader
    gl.glUniform1i(gl.glGetUniformLocation(shader, "texture1"), 0)
    gl.glUniform1i(gl.glGetUniformLocation(shader, "texture2"), 1)

    processunit = ProcessInputUnit(shader)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_cursor_pos_callback(window, processunit.mouse_callback)
    glfw.set_scroll_callback(window, processunit.scroll_callback)

    # model
    model = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, 0.0))

    # many cubes
    cube_positions = [
        glm.vec3(0.0, 0.0, 0.0    ),
        glm.vec3(2.0, 5.0, -15.0  ),
        glm.vec3(-1.5, -2.2, -2.5 ),
        glm.vec3(-3.8, -2.0, -12.3),
        glm.vec3(2.4, -0.4, -3.5  ),
        glm.vec3(-1.7, 3.0, -7.5  ),
        glm.vec3(1.3, -2.0, -2.5  ),
        glm.vec3(1.3, 2.0, -1.5   ),
        glm.vec3(-1.3, 0.2, -1.5  ),
    ]

    while not glfw.window_should_close(window):
        processunit.process_input(window)

        gl.glUniform1f(gl.glGetUniformLocation(shader, "mix_ratio"), abs(sin(glfw.get_time())))
        glfw.poll_events()
        view = processunit.camera.get_lookAt()
        projection = processunit.camera.get_projection()

        gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "view"), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "projection"), 1, gl.GL_FALSE, glm.value_ptr(projection))

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glBindVertexArray(vao)
        for i, cube_position in enumerate(cube_positions):
            # model
            model = glm.translate(glm.mat4(1.0), cube_position)
            model = glm.rotate(model, float((i+100.0) * glfw.get_time() / 200.0), glm.vec3(i%3, (i+1)%3, (i+2)%3))
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "model"), 1, gl.GL_FALSE, glm.value_ptr(model))
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(vertices.size / 5))

        glfw.swap_buffers(window)
    glfw.terminate()


def on_resize(window, w, h):
    gl.glViewport(0, 0, w, h)

class ProcessInputUnit():
    def __init__(self, shader) -> None:
        self.mix_val = 0.5
        self.shader = shader
        gl.glUniform1f(gl.glGetUniformLocation(self.shader, "mix_ratio"), self.mix_val)

        self.camera = Camera(None, None, None)
        self.delta_time = 0.0
        self.last_frame = 0.0

        self.first_mouse = True
        self.sensitivity = 0.1
        self.last_x = 800 / 2
        self.last_y = 800 / 2
        self.yaw = -90.0
        self.pitch = 0.0
        self.zoom_flag = False

    def process_input(self, window) -> None:
        current_frame = glfw.get_time()
        self.delta_time = current_frame - self.last_frame
        self.last_frame = current_frame
        self.camera_speed = 2.5 * self.delta_time

        if (glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS):
            glfw.set_window_should_close(window, True)

        if (glfw.get_key(window, glfw.KEY_W) == glfw.PRESS):
            self.camera.camera_pos += self.camera_speed * self.camera.camera_front
            self.camera.cal_lookat()
        if (glfw.get_key(window, glfw.KEY_S) == glfw.PRESS):
            self.camera.camera_pos -= self.camera_speed * self.camera.camera_front
            self.camera.cal_lookat()
        if (glfw.get_key(window, glfw.KEY_A) == glfw.PRESS):
            self.camera.camera_pos -= self.camera.camera_right * self.camera_speed
            self.camera.cal_lookat()
        if (glfw.get_key(window, glfw.KEY_D) == glfw.PRESS):
            self.camera.camera_pos += self.camera.camera_right * self.camera_speed
            self.camera.cal_lookat()
            
        return None
    
    def mouse_callback(self, window, xpos, ypos):
        # xpos, ypos: current cursor location
        if self.first_mouse:
            self.last_x, self.last_y = xpos, ypos
            self.first_mouse = False
        
        x_offset = xpos - self.last_x
        y_offset = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos
        
        x_offset *= self.sensitivity
        y_offset *= self.sensitivity

        self.yaw += x_offset
        self.pitch += y_offset

        # to avoid problems
        if self.pitch > 89.0:
            self.pitch = 89.0
        elif self.pitch < -89.0:
            self.pitch = -89.0
        
        front = glm.vec3()
        front.x = cos(glm.radians(self.yaw)) * cos(glm.radians(self.pitch))
        front.y = sin(glm.radians(self.pitch))
        front.z = sin(glm.radians(self.yaw)) * cos(glm.radians(self.pitch))
        self.camera.camera_front = glm.normalize(front)
        self.camera.cal_lookat()
        return None
    
    def scroll_callback(self, window, x_offset, y_offset):
        # change field of view
        # y offset:  wheel
        fov = self.camera.fov
        if (fov >= 10.0 and fov <= 60.0):
            fov -= y_offset

        if (fov <= 10.0):
            fov = 10.0
        if (fov >= 60.0):
            fov = 60.0
        self.camera.set_fov(fov)
        return None
        
        
class Camera:
    def __init__(self, camera_pos, camera_front, world_up) -> None:
        if camera_pos is None:
            camera_pos = glm.vec3(0.0, 0.0, 3.0)
        if camera_front is None:
            camera_front = glm.vec3(0.0, 0.0, -1.0)
        if world_up is None:
            world_up = glm.vec3(0.0, 1.0, 0.0)
            
        self.camera_pos = camera_pos
        self.camera_front = camera_front
        self.world_up = world_up
        self.fov = 20.0
        self.camera_right = glm.normalize(glm.cross(world_up, camera_pos - (self.camera_pos + self.camera_front)))
        self.lookat = glm.lookAt(self.camera_pos, self.camera_pos + self.camera_front, self.world_up)

        # projection
        self.projection = glm.perspective(glm.radians(self.fov), 800.0/800.0, 0.1, 100.0)

    def cal_lookat(self, ):
        self.camera_right = glm.normalize(glm.cross(self.world_up, self.camera_pos - (self.camera_pos + self.camera_front)))
        self.lookat = glm.lookAt(self.camera_pos, self.camera_pos + self.camera_front, self.world_up)
    
    def change_camera_pos(self, camera_pos):
        self.camera_pos = camera_pos
        self.lookat = glm.lookAt(self.camera_pos, self.camera_pos + self.camera_front, self.world_up)
        return None

    def change_camera_pos(self, camera_pos):
        self.camera_pos = camera_pos
        self.lookat = glm.lookAt(self.camera_pos, self.camera_pos + self.camera_front, self.world_up)
        return None

    def change_camera_pos(self, camera_pos):
        self.camera_pos = camera_pos
        self.lookat = glm.lookAt(self.camera_pos, self.camera_pos + self.camera_front, self.world_up)
        return None
    
    def get_lookAt(self,):
        return self.lookat
    
    def set_fov(self, fov: float):
        self.fov = fov
        self.cal_projection()
        return None
    
    def cal_projection(self, ):
        self.projection = glm.perspective(glm.radians(self.fov), 800.0/800.0, 0.1, 100.0)
        return None
    
    def get_projection(self, ):
        return self.projection
    


if __name__ == "__main__":
    main()