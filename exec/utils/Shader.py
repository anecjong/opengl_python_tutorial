import OpenGL.GL as gl
import glm

class Shader:
    '''
    shader wrapper
    '''
    def __init__(self, program) -> None:
        self.program = program
    
    def use(self, ):
        gl.glUseProgram(self.program)
    
    def uniform_bool(self, uniform_str, bool_val):
        gl.glUniform1i(gl.glGetUniformLocation(self.program, uniform_str), bool_val)
    
    def uniform_int(self, uniform_str, int_val):
        gl.glUniform1i(gl.glGetUniformLocation(self.program, uniform_str), int_val)

    def uniform_float(self, uniform_str, float_val):
        gl.glUniform1f(gl.glGetUniformLocation(self.program, uniform_str), float_val)
    
    def uniform_vec2(self, uniform_str, vec2):
        gl.glUniform2fv(gl.glGetUniformLocation(self.program, uniform_str), 1, glm.value_ptr(vec2))

    def uniform_vec3(self, uniform_str, vec3):
        gl.glUniform3fv(gl.glGetUniformLocation(self.program, uniform_str), 1, glm.value_ptr(vec3))

    def uniform_vec4(self, uniform_str, vec4):
        gl.glUniform4fv(gl.glGetUniformLocation(self.program, uniform_str), 1, glm.value_ptr(vec4))
    
    def uniform_mat4(self, uniform_str:str, mat3):
        gl.glUniformMatrix3fv(gl.glGetUniformLocation(self.program, uniform_str), 1, gl.GL_FALSE, glm.value_ptr(mat3))

    def uniform_mat4(self, uniform_str:str, mat4):
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.program, uniform_str), 1, gl.GL_FALSE, glm.value_ptr(mat4))
