import glfw
import OpenGL
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import numpy as np
from ctypes import c_void_p
from PIL import Image
from scipy.spatial.transform import Rotation
from math import radians, tan
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

    # model
    model = np.identity(4, dtype=np.float32)
    r = Rotation.from_rotvec([0.0, 0.0, 0.0])
    model[:3, :3] = r.as_matrix()

    # view
    view = np.identity(4, dtype=np.float32)
    view[0:3, 3] = np.array([0, 0, -3], dtype=np.float32).transpose()

    # projection matrix
    projection = perspective(45.0, 800.0/600.0, 0.1, 100)

    while not glfw.window_should_close(window):
        processunit.process_input(window)
        # model
        model = np.identity(4, dtype=np.float32)
        r = Rotation.from_rotvec(glfw.get_time() * np.array([0.3, 0.6, 0.5]))
        model[:3, :3] = r.as_matrix()

        gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "model"), 1, gl.GL_TRUE, model)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "view"), 1, gl.GL_TRUE, view)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "projection"), 1, gl.GL_TRUE, projection)

        glfw.poll_events()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glBindVertexArray(vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(vertices.size / vertices.itemsize))

        glfw.swap_buffers(window)
    glfw.terminate()


def on_resize(window, w, h):
    gl.glViewport(0, 0, w, h)

class ProcessInputUnit():
    def __init__(self, shader) -> None:
        self.mix_val = 0.5
        self.shader = shader
        gl.glUniform1f(gl.glGetUniformLocation(self.shader, "mix_ratio"), self.mix_val)

    def process_input(self, window) -> None:
        # get_key returns glfw.PRESS if key is pressed. If not, glfw.RELEASE is returned.
        if (glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS):
            glfw.set_window_should_close(window, True)

        if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            self.mix_val += 0.001
            if self.mix_val > 1.0:
                self.mix_val = 1.0
            gl.glUniform1f(gl.glGetUniformLocation(self.shader, "mix_ratio"), self.mix_val)

        if (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            self.mix_val -= 0.001
            if self.mix_val < 0.0:
                self.mix_val = 0.0
            gl.glUniform1f(gl.glGetUniformLocation(self.shader, "mix_ratio"), self.mix_val)
        return

def perspective(field_of_view_y: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
    '''
    opengl perspective for python
    field of view y : degree
    aspect: width / height
    z near: near z plane
    z far: far z plane
    '''
    t = tan(radians(field_of_view_y/2))
    r = t * aspect
    perspective_matrix = np.zeros((4, 4), dtype=np.float32)
    perspective_matrix[0][0] = 1 / r
    perspective_matrix[1][1] = 1 / t
    perspective_matrix[2][2] = -(z_far + z_near)/(z_far - z_near)
    perspective_matrix[2][3] = (-2.0 * z_far * z_near) / (z_far - z_near)
    perspective_matrix[3][2] = -1.0

    return perspective_matrix

if __name__ == "__main__":
    main()