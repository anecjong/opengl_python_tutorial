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
from math import sin, cos
from utils import ProcessUnit, Camera, Shader

vertex_source = """
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 in_tex_coord;

out vec3 normal_out;
out vec3 frag_pos;
out vec3 light_pos_out;
out vec2 tex_coord;

uniform mat4 model;
uniform mat4 view;
uniform vec3 light_pos;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(position, 1.0f);

    frag_pos = vec3(view * model * vec4(position, 1.0f));
    normal_out = mat3(transpose(inverse(view * model))) * normal;
    light_pos_out = vec3(view * vec4(light_pos, 1.0f));
    tex_coord = in_tex_coord;
}
"""

lamp_fragment_source = """
#version 330 core
out vec4 FragColor;

uniform vec3 lamp_color;

void main()
{
    FragColor = vec4(lamp_color, 1.0f);
}
"""

container_fragment_source = """
#version 330 core
out vec4 frag_color;

in vec3 normal_out;
in vec3 frag_pos;
in vec3 light_pos_out;
in vec2 tex_coord;

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float     shininess;
}; 

struct Light {
    vec3 position;
  
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform Material material;
uniform Light light;  

void main() {
    // ambient
    vec3 ambient = light.ambient * vec3(texture(material.diffuse, tex_coord));

    // diffuse
    vec3 norm = normalize(normal_out);
    vec3 light_dir = normalize(light_pos_out - frag_pos);
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = light.diffuse * (diff * vec3(texture(material.diffuse, tex_coord)));

    // specular
    vec3 view_dir = normalize(-frag_pos);
    vec3 reflect_dir = reflect(-light_dir, norm);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material.shininess);
    vec3 specular = light.specular * (spec * vec3(texture(material.specular, tex_coord)));

    vec3 result = (ambient + diffuse + specular);
    frag_color = vec4(result, 1.0f);
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
        # positions        normals           texture coords
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  1.0,  0.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0,  1.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0,  1.0,
        -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  0.0,  1.0,
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0,  0.0,

        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0,  0.0,
         0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0,  0.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0,  1.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0,  1.0,
        -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  0.0,  1.0,
        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0,  0.0,

        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0,  0.0,
        -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  1.0,  1.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0,  1.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0,  1.0,
        -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  0.0,  0.0,
        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0,  0.0,

         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0,  0.0,
         0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  1.0,  1.0,
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  0.0,  1.0,
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  0.0,  1.0,
         0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  0.0,  0.0,
         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0,  0.0,

        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0,  1.0,
         0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0,  1.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0,  0.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0,  0.0,
        -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0,  0.0,
        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0,  1.0,

        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0,  1.0,
         0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  1.0,  1.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0,  0.0,
        -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  0.0,  0.0,
        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0,  1.0,
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
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 8 * vertices.itemsize, c_void_p(0))
    gl.glEnableVertexAttribArray(0)

    # normal
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 8 * vertices.itemsize, c_void_p(3 * vertices.itemsize))
    gl.glEnableVertexAttribArray(1)

    # texture coord
    gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, 8 * vertices.itemsize, c_void_p(6 * vertices.itemsize))
    gl.glEnableVertexAttribArray(2)

    # texture binding - diffuse
    texture_diffuse = gl.glGenTextures(1)
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_diffuse)

    # texture wrapping
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)

    # texture filtering
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

    # texture img load
    img = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "resources", "logo.png")).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.width, img.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img.tobytes())
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

    # texture binding - specular
    texture_specular = gl.glGenTextures(1)
    gl.glActiveTexture(gl.GL_TEXTURE1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_specular)

    # texture wrapping
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)

    # texture filtering
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

    # texture img load
    img = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "resources", "logo_size_black.png")).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.width, img.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img.tobytes())
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

    container_shader.use()
    container_shader.uniform_int("material.diffuse", 0)
    container_shader.uniform_int("material.specular", 1)

    # lamp vao
    lamp_vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(lamp_vao)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 8 * vertices.itemsize, c_void_p(0))
    gl.glEnableVertexAttribArray(0)

    # wireframe mode
    # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

    # fill mode
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
    
    # camera location change
    process_unit.camera.change_camera_pos(glm.vec3(0.0, -0.3, 5.0))

    while not glfw.window_should_close(window):
        process_unit.keyboard_input(window)

        # light position
        theta = glfw.get_time()
        radius = 2
        light_pos = radius * glm.vec3(cos(theta/5), 0.0, sin(theta/5))

        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # container
        container_shader.use()

        view = process_unit.camera.get_lookAt()
        projection = process_unit.camera.get_projection()
        container_shader.uniform_mat4("projection", projection)
        container_shader.uniform_mat4("view", view)

        container_shader.uniform_vec3("light_pos", light_pos)

        model = glm.mat4(1.0)
        container_shader.uniform_mat4("model", model)

        # material
        container_shader.uniform_float("material.shininess", 8.0)

        # light
        lamp_color = glm.vec3(1.0, 1.0, 1.0)
        container_shader.uniform_vec3("light.ambient",  glm.vec3(0.2))
        container_shader.uniform_vec3("light.diffuse",  glm.vec3(0.5))
        container_shader.uniform_vec3("light.specular", glm.vec3(1.0))
        
        gl.glBindVertexArray(container_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(vertices.size / 3))

        # lamp
        lamp_shader.use()
        view = process_unit.camera.get_lookAt()
        projection = process_unit.camera.get_projection()
        lamp_shader.uniform_mat4("projection", projection)
        lamp_shader.uniform_mat4("view", view)

        model = glm.translate(glm.mat4(1), light_pos)
        model = glm.scale(model, glm.vec3(0.2, 0.2, 0.2))
        lamp_shader.uniform_mat4("model", model)
        lamp_shader.uniform_vec3("lamp_color", lamp_color)
        
        gl.glBindVertexArray(lamp_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(vertices.size / 3))
        
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    main()