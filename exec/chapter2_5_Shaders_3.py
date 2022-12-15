import glfw
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import numpy as np
from ctypes import c_void_p

vertex_shader_source = """
#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aColor;

out vec3 ourColor;

void main(){
    gl_Position = vec4(aPos, 1.0f);
    ourColor = aColor;
}
"""

fragment_shader_source = """
#version 330 core
out vec4 FragColor;
in vec3 ourColor;

void main(){
    FragColor = vec4(ourColor, 1.0f);
}
"""

def main():
    if not glfw.init():
        return -1
    
    # pyopengl version : 3.1
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
    
    vertices = np.array([
        # position        # color
        -0.5, -0.5,  0.0, 1.0, 0.0, 0.0,
        0.5, -0.5,  0.0,  0.0, 1.0, 0.0,
        0.0,  0.5,  0.0,  0.0, 0.0, 1.0
        ], dtype=np.float32)
    vertices_size = vertices.size * vertices.itemsize

    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_size, vertices, gl.GL_STATIC_DRAW)

    shader = shaders.compileProgram(
        shaders.compileShader(vertex_shader_source, gl.GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader_source, gl.GL_FRAGMENT_SHADER),
                                    )
    
    gl.glUseProgram(shader)

    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    # aPos
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 6 * vertices.itemsize, c_void_p(0))
    gl.glEnableVertexAttribArray(0)

    # aColor
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 6 * vertices.itemsize, c_void_p(3 * vertices.itemsize))
    gl.glEnableVertexAttribArray(1)

    gl.glClearColor(0.2, 0.3, 0.3, 1.0)

    while not glfw.window_should_close(window):
        process_input(window)
        glfw.poll_events()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glBindVertexArray(vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        glfw.swap_buffers(window)
    glfw.terminate
    

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