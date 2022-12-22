import glfw
import OpenGL
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import numpy as np
from ctypes import c_void_p
from PIL import Image
from math import sin, cos
import glm
from utils import ProcessUnit, Camera, Shader
import os

vertex_source = """
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 in_tex_coord;

out vec3 normal_out;
out vec3 frag_pos;
out vec2 tex_coord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0f);

    frag_pos = vec3(view * model * vec4(position, 1.0f));
    normal_out = mat3(transpose(inverse(view * model))) * normal;
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

fragment_source = """
#version 330 core
out vec4 frag_color;

in vec3 normal_out;
in vec3 frag_pos;
in vec2 tex_coord;

struct Material {
    sampler2D diffuse_1;
    sampler2D diffuse_2;
    float     mix_ratio;
    sampler2D specular;
    float     shininess;
}; 

struct PointLight {
    vec3 position;
  
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float constant;
    float linear;
    float quadratic;
};

uniform Material material;
uniform PointLight light;  

void main() {
    float distance = length(light.position - frag_pos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    
    // ambient
    vec3 material_diffuse = mix(texture(material.diffuse_1, tex_coord), texture(material.diffuse_2, tex_coord), material.mix_ratio).rgb;
    vec3 ambient = light.ambient * material_diffuse;

    // diffuse
    vec3 norm = normalize(normal_out);
    vec3 light_dir = normalize(light.position - frag_pos);
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = light.diffuse * (diff * material_diffuse);

    // specular
    vec3 view_dir = normalize(-frag_pos);
    vec3 reflect_dir = reflect(-light_dir, norm);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material.shininess);
    vec3 specular = light.specular * (spec * vec3(texture(material.specular, tex_coord)));

    vec3 result = attenuation * (ambient + diffuse + specular);
    frag_color = vec4(result, 1.0f);
}
"""
def main():
    if not glfw.init():
        return -1

    print(f"pyopengl version: {OpenGL.__version__}")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    width, height = 800, 800
    window = glfw.create_window(width, height, "learn OpenGL", None, None)
    if window is None:
        print("Failed to create glfw window")
        glfw.terminate()
        return -1

    process_unit = ProcessUnit(width, height)
    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, process_unit.resize_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
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

    shader = shaders.compileProgram(shaders.compileShader(vertex_source, gl.GL_VERTEX_SHADER),
                                    shaders.compileShader(fragment_source, gl.GL_FRAGMENT_SHADER))
    shader = Shader(shader)

    shader_lamp = shaders.compileProgram(shaders.compileShader(vertex_source, gl.GL_VERTEX_SHADER),
                                    shaders.compileShader(lamp_fragment_source, gl.GL_FRAGMENT_SHADER))
    shader_lamp = Shader(shader_lamp)

    # vao
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

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

    # texture
    gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, 8 * vertices.itemsize, c_void_p(6 * vertices.itemsize))
    gl.glEnableVertexAttribArray(2)

    # texture binding - diffuse
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
    img = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "resources", "logo.png"))
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.width, img.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img.tobytes())
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

    # texture binding - specular
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
    img = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "resources", "logo_differ_color.png"))
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.width, img.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img.tobytes())
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

    # wireframe mode
    # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

    # fill mode
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
    
    gl.glClearColor(0.2, 0.3, 0.3, 1.0)

    # many cubes
    cube_positions = [
        glm.vec3(0.0,   0.0,   0.0),
        glm.vec3(2.0,   5.0,   -15.0),
        glm.vec3(-1.5, -2.2,   -2.5),
        glm.vec3(-3.8, -2.0,   -12.3),
        glm.vec3(2.4,  -0.4,   -3.5),
        glm.vec3(-1.7,  3.0,   -7.5),
        glm.vec3(1.3,  -2.0,   -2.5),
        glm.vec3(1.3,   2.0,   -1.5),
        glm.vec3(-1.3,  0.2,   -1.5),
    ]

    process_unit.camera.change_camera_pos(glm.vec3(0, 0.0, 8.0))
    process_unit.camera.change_fov(45)

    while not glfw.window_should_close(window):
        # keyboard_input()
        process_unit.keyboard_input(window)

        # view, projection info
        view = process_unit.camera.get_lookAt()
        projection = process_unit.camera.get_projection()

        # lamp position
        light_pos = glm.vec3(1.0, 1.0, 3.0)

        # texture shader
        shader.use()
        shader.uniform_int("material.diffuse_1", 0)
        shader.uniform_int("material.specular", 1)
        shader.uniform_int("material.diffuse_2", 0)

        # set projection / view
        shader.uniform_mat4("projection", projection)
        shader.uniform_mat4("view", view)
        
        # light
        shader.uniform_vec3("light.position", view * light_pos)
        shader.uniform_vec3("light.ambient",  glm.vec3(0.2))
        shader.uniform_vec3("light.diffuse",  glm.vec3(0.8))
        shader.uniform_vec3("light.specular", glm.vec3(1.0))

        shader.uniform_float("light.constant", 1.0)
        shader.uniform_float("light.linear", 0.09)
        shader.uniform_float("light.quadratic", 0.032)

        # material shiness, mix ratio
        shader.uniform_float("material.shininess", 32)
        shader.uniform_float("material.mix_ratio", 0.5)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glBindVertexArray(vao)

        for i, cube_position in enumerate(cube_positions):
            # model
            model = glm.translate(glm.mat4(1), cube_position)
            model = glm.rotate(model,(i+100.0) * glfw.get_time() / 200.0, glm.vec3(i%3, (i+1)%3, (i+2)%3))

            shader.uniform_mat4("model", model)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(vertices.size / 5))

        # lamp
        shader_lamp.use()

        # projection / view / model
        shader_lamp.uniform_mat4("projection", projection)
        shader_lamp.uniform_mat4("view", view)
        shader_lamp.uniform_vec3("position", light_pos)
        shader_lamp.uniform_vec3("lamp_color", glm.vec3(1.0, 1.0, 1.0))

        model = glm.translate(glm.mat4(1), light_pos)
        model = glm.scale(model, glm.vec3(0.1, 0.1, 0.1))
        shader_lamp.uniform_mat4("model", model)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(vertices.size / 5))

        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    main()