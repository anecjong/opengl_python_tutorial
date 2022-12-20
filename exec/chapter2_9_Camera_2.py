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

    # model
    model = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, 0.0))

    # projection matrix
    projection = glm.perspective(glm.radians(45.0), 800.0/800.0, 0.1, 100.0)

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


        glfw.poll_events()
        # camera - view
        radius = 10.0
        cam_x = radius * cos(glfw.get_time())
        cam_z = radius * sin(glfw.get_time())
        view = glm.lookAt(glm.vec3(cam_x, 0.0, cam_z),
                        glm.vec3(0.0, 0.0, 0.0),
                        glm.vec3(0.0, 1.0, 0.0))

        gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "view"), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "projection"), 1, gl.GL_FALSE, glm.value_ptr(projection))

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glBindVertexArray(vao)
        for i, cube_position in enumerate(cube_positions):
            # model
            model = glm.translate(glm.mat4(1.0), cube_position)
            model = glm.rotate(model, float((i+100.0) * glfw.get_time() / 200.0), glm.vec3(i%3, (i+1)%3, (i+2)%3))
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "model"), 1, gl.GL_FALSE, glm.value_ptr(model))
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

if __name__ == "__main__":
    main()