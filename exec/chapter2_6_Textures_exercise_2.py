import glfw
import OpenGL
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import numpy as np
from ctypes import c_void_p
from PIL import Image
import os

vertex_shader_source = """
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 in_tex_coords;

out vec3 new_color;
out vec2 out_tex_coords;

void main() {
    gl_Position = vec4(position, 1.0f);
    new_color = color;
    out_tex_coords = in_tex_coords;
}
"""

fragment_shader_source = """
#version 330 core

in vec3 new_color;
in vec2 out_tex_coords;

out vec4 FragColor;
uniform sampler2D texture1;
uniform sampler2D texture2;

void main() {
    FragColor = mix(texture(texture1, out_tex_coords), texture(texture2, out_tex_coords), 0.4);
    // first 0.8, second 0.
}
"""

def main():
    if not glfw.init():
        return -1

    print(f"pyopengl version: {OpenGL.__version__}")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    window = glfw.create_window(800, 800, "learn OpenGL", None, None)
    if window is None:
        print("Failed to create glfw window")
        glfw.terminate()
        return -1

    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, on_resize)

    vertices = np.array([0.9, 0.9, 0.0,     1.0, 0.0, 0.0,  2.0, 0.0,       # top right
                         0.9, -0.9, 0.0,    0.0, 1.0, 0.0,  2.0, 2.0,       # bottom right
                         -0.9, -0.9, 0.0,   0.0, 0.0, 1.0,  0.0, 2.0,       # bottom left
                         -0.9, 0.9, 0.0,    1.0, 1.0, 1.0,  0.0, 0.0,       # top left
        ], dtype=np.float32)

    # locate image at center
    # vertices = np.array([0.5, 0.5, 0.0,     1.0, 0.0, 0.0,  1.5, -0.5,      # top right
    #                      0.5, -0.5, 0.0,    0.0, 1.0, 0.0,  1.5, 1.5,       # bottom right
    #                      -0.5, -0.5, 0.0,   0.0, 0.0, 1.0,  -0.5, 1.5,      # bottom left
    #                      -0.5, 0.5, 0.0,    1.0, 1.0, 1.0,  -0.5, -0.5,     # top left
    #     ], dtype=np.float32)

    vertices_size = vertices.size * vertices.itemsize

    # indices
    indices = np.array([
        0, 1, 3, 
        1, 2, 3,
        ], dtype=np.uint32)
    
    indices_size = indices.size * indices.itemsize

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

    # ebo buffer binding
    ebo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices_size, indices, gl.GL_STATIC_DRAW)

    # position
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 8 * vertices.itemsize, c_void_p(0))
    gl.glEnableVertexAttribArray(0)

    # color
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 8 * vertices.itemsize, c_void_p(3 * vertices.itemsize))
    gl.glEnableVertexAttribArray(1)

    # texture
    gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, 8 * vertices.itemsize, c_void_p(6 * vertices.itemsize))
    gl.glEnableVertexAttribArray(2)

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

    # shader
    gl.glUniform1i(gl.glGetUniformLocation(shader, "texture1"), 0)
    gl.glUniform1i(gl.glGetUniformLocation(shader, "texture2"), 1)

    while not glfw.window_should_close(window):
        process_input(window)
        glfw.poll_events()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glBindVertexArray(vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, c_void_p(0))

        glfw.swap_buffers(window)
    glfw.terminate()


def on_resize(window, w, h):
    gl.glViewport(0, 0, w, h)

def process_input(window):
    '''
    if esc key is pressed, change window_should_close attribute True.
    '''
    # get_key returns glfw.PRESS if key is pressed. If not, glfw.RELEASE is returned.
    if (glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS):
        glfw.set_window_should_close(window, True)

if __name__ == "__main__":
    main()