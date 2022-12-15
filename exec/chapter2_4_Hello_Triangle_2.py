import glfw
import OpenGL
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import numpy as np
from ctypes import c_uint, c_float, sizeof, c_void_p

vertex_shader_source = """
#version 330 core

in vec3 position;

void main() {
    gl_Position = vec4(position, 1.0f);
}
"""

fragment_shader_source = """
#version 330 core

out vec4 FragColor;

void main() {
    FragColor = vec4(0.2f, 0.7f, 0.2f, 1.0f);
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

    vertices = np.array([
        0.5, 0.5,  0.0,     # top right
        0.5, -0.5,  0.0,    # bottom right
        -0.5, -0.5, 0.0,    # bottom left
        -0.5, 0.5, 0.0,     # top left
        ], dtype=np.float32)
    
    vertices_size = vertices.size * vertices.itemsize

    # indices
    indices = np.array([
        0, 1, 3, 
        1, 2, 3,
        ], dtype=np.uint32)
    
    indices_size = indices.size * indices.itemsize

    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_size, vertices, gl.GL_STATIC_DRAW)

    vertex_shader = shaders.compileShader(vertex_shader_source, gl.GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(fragment_shader_source, gl.GL_FRAGMENT_SHADER)

    shader = shaders.compileProgram(vertex_shader, fragment_shader)
    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)
    gl.glUseProgram(shader)

    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    # ebo buffer binding
    ebo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices_size, indices, gl.GL_STATIC_DRAW)

    position = gl.glGetAttribLocation(shader, "position")
    gl.glVertexAttribPointer(position, 3, gl.GL_FLOAT, gl.GL_FALSE, vertices.itemsize * 3, c_void_p(0))
    gl.glEnableVertexAttribArray(position)

    gl.glClearColor(0.2, 0.3, 0.3, 1.0)

    # wireframe mode
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

    # fill mode
    # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
    
    while not glfw.window_should_close(window):
        process_input(window)
        glfw.poll_events()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # glDrawArrays -> glDrawElements
        # glDrawElements(primitive type, number of elements, type of indicies, offset)
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