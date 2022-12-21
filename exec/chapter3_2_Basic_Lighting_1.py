import glfw
import OpenGL
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import numpy as np
from ctypes import c_void_p
from PIL import Image
import glm
import os
import sys
from utils import ProcessUnit, Camera, Shader
from math import sin, cos

vertex_source = """
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec3 normal_out;
out vec3 frag_pos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection; void main() {
    gl_Position = projection * view * model * vec4(position, 1.0f);

    frag_pos = vec3(model * vec4(position, 1.0));
    normal_out = mat3(transpose(inverse(model))) * normal;
}
"""

lamp_fragment_source = """
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0f);
}
"""

container_fragment_source = """
#version 330 core
out vec4 frag_color;

in vec3 normal_out;
in vec3 frag_pos;

uniform vec3 object_color;
uniform vec3 light_color;
uniform vec3 light_pos;

void main() {
    // ambient
    float ambient_strength = 0.1;
    vec3 ambient = ambient_strength * light_color;

    // diffuse
    vec3 norm = normalize(normal_out);
    vec3 light_dir = normalize(light_pos - frag_pos);
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diff * light_color;

    frag_color = vec4((ambient+diffuse) * object_color, 1.0f);
}
"""

def main():
    if not glfw.init():
        return -1

    print(f"pyopengl version: {OpenGL.__version__}")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)

    width, height = 800, 800
    window = glfw.create_window(width, height, "learn OpenGL", None, None)
    if window is None:
        print("Failed to create glfw window")
        glfw.terminate()
        return -1

    glfw.make_context_current(window)
    process_unit = ProcessUnit(width, height)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_window_size_callback(window, process_unit.resize_callback)
    glfw.set_cursor_pos_callback(window, process_unit.mouse_callback)
    glfw.set_scroll_callback(window, process_unit.scroll_callback)

    # z-buffer
    gl.glEnable(gl.GL_DEPTH_TEST)

    vertices = np.array([
        # positions        normals         
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
         0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
        -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
        
        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
         0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
        -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
        
        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,
        -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
        -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,
        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,
        
         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,
         0.5,  0.5, -0.5,  1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
         0.5, -0.5,  0.5,  1.0,  0.0,  0.0,
         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,
        
        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
         0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
        -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
        
        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
         0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
        -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
    ], dtype=np.float32)

    vertices_size = vertices.size * vertices.itemsize

    container_shader = shaders.compileProgram(shaders.compileShader(vertex_source, gl.GL_VERTEX_SHADER),
                                    shaders.compileShader(container_fragment_source, gl.GL_FRAGMENT_SHADER))
    container_shader = Shader(container_shader)

    lamp_shader = shaders.compileProgram(shaders.compileShader(vertex_source, gl.GL_VERTEX_SHADER),
                                    shaders.compileShader(lamp_fragment_source, gl.GL_FRAGMENT_SHADER))
    lamp_shader = Shader(lamp_shader)

    # container vao
    container_vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(container_vao)

    # vbo
    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)

    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_size, vertices, gl.GL_STATIC_DRAW)

    # position
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 6 * vertices.itemsize, c_void_p(0))
    gl.glEnableVertexAttribArray(0)

    # normal
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 6 * vertices.itemsize, c_void_p(3 * vertices.itemsize))
    gl.glEnableVertexAttribArray(1)

    # lamp vao
    lamp_vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(lamp_vao)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 6 * vertices.itemsize, c_void_p(0))
    gl.glEnableVertexAttribArray(0)

    # wireframe mode
    # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

    # fill mode
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
    
    last_frame = 0.0

    while not glfw.window_should_close(window):
        process_unit.keyboard_input(window)

        # light position
        theta = glfw.get_time()
        radius = 2
        light_pos = radius * glm.vec3(cos(theta), sin(theta), 0)

        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # container
        container_shader.use()
        container_shader.uniform_vec3("object_color", glm.vec3(1.0, 0.5, 0.31))
        container_shader.uniform_vec3("light_color", glm.vec3(1.0, 0.5, 0.31))

        view = process_unit.camera.get_lookAt()
        projection = process_unit.camera.get_projection()
        container_shader.uniform_mat4("projection", projection)
        container_shader.uniform_mat4("view", view)
        container_shader.uniform_vec3("light_pos", light_pos)

        model = glm.mat4(1.0)
        model = glm.rotate(model, 0.1 * theta, glm.vec3(1, 1, 1))
        model = glm.scale(model, glm.vec3(1.0, 1.5, 2.0))
        container_shader.uniform_mat4("model", model)
        
        gl.glBindVertexArray(container_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(vertices.size / 3))

        # lamp
        lamp_shader.use()
        view = process_unit.camera.get_lookAt()
        projection = process_unit.camera.get_projection()
        lamp_shader.uniform_mat4("projection", projection)
        lamp_shader.uniform_mat4("view", view)

        model = glm.translate(glm.mat4(1), light_pos)
        # model = glm.rotate(model, current_frame, glm.vec3(0, 0, -1))
        model = glm.scale(model, glm.vec3(0.2, 0.2, 0.2))
        lamp_shader.uniform_mat4("model", model)
        
        gl.glBindVertexArray(lamp_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(vertices.size / 3))
        
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()

            


if __name__ == "__main__":
    main()