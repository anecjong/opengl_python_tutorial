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

    # pyopengl version : 3.1
    print(f"pyopengl version: {OpenGL.__version__}")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    window = glfw.create_window(800, 600, "learn OpenGL", None, None)
    if window is None:
        print("Failed to create glfw window")
        glfw.terminate()
        return -1

    # make_context_current function make viewport with window size one time
    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, on_resize)

    # triangle on plane (z = 0)
    vertices = np.array([
        -0.5, -0.5,  0.0,
        0.5, -0.5,  0.0,
        0.0,  0.5,  0.0], dtype=np.float32)
    
    # vertices_size ~ size of vertices (bytes)
    vertices_size = vertices.size * vertices.itemsize

    # glGenBuffers(count, buffers: specifies an array in which the generated buffer object names are stored.)
    # type of vertex buffer object: GL_ARRAY_BUFFER
    vbo = gl.glGenBuffers(1)

    # OpenGL allows us to bind several buffers at once as long as they have a different buffer type
    # Any buffer calls we make on the GL_ARRAY_BUFFER target will be used to configure the currently bound buffer.
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)

    # copy previously dfined vertex data into the buffer's memory
    # glBufferData function is a specifically targeted to copy user-defined data into the currently bounded buffer.
    # glBufferData(type of the buffer, size of the data, actual data, graphic cards managing policy)
    # 4th parameter: GL_STREAM_DRAW, GL_STATIC_DRAW, GL_DYNAMIC_DRAW
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_size, vertices, gl.GL_STATIC_DRAW)

    # compile vertex shader
    vertex_shader = shaders.compileShader(vertex_shader_source, gl.GL_VERTEX_SHADER)

    # compile fragment shader
    fragment_shader = shaders.compileShader(fragment_shader_source, gl.GL_FRAGMENT_SHADER)

    # linking shaders to program and delete shader
    shader = shaders.compileProgram(vertex_shader, fragment_shader)
    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)
    gl.glUseProgram(shader)

    # vertex array object
    # save vertex attributes pointers
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    position = gl.glGetAttribLocation(shader, "position")

    # glVertexAttribPointer(location of position, size of the vertex attributes, type of data, normalize,
    # stride, offset of where the position data begins in buffer)
    gl.glVertexAttribPointer(position, 3, gl.GL_FLOAT, gl.GL_FALSE, vertices.itemsize * 3, c_void_p(0))
    gl.glEnableVertexAttribArray(position)

    gl.glClearColor(0.2, 0.3, 0.3, 1.0)

    while not glfw.window_should_close(window):
        process_input(window)
        glfw.poll_events()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # glDrawyArrays(primitive type, starting index of the vertex array, how many vertices)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

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