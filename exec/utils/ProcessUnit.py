import glfw
import glm
import OpenGL.GL as gl
from math import sin, cos
from utils.Camera import Camera

class ProcessUnit():
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height

        self.camera = Camera()
        self.last_frame = 0.0

        # mouse
        self.first_mouse = True
        self.sensitivity = 0.1
        self.last_x = 800 / 2
        self.last_y = 800 / 2
        self.yaw = -90.0
        self.pitch = 0.0

    def resize_callback(self, window, width, height):
        '''
        change viewport corresponding to window
        '''
        gl.glViewport(0, 0, width, height)
        self.width = width
        self.height = height

    def keyboard_input(self, window) -> None:
        current_frame = glfw.get_time()
        self.delta_time = current_frame - self.last_frame
        self.last_frame = current_frame
        self.camera_speed = 2.5 * self.delta_time

        if (glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS):
            glfw.set_window_should_close(window, True)

        if (glfw.get_key(window, glfw.KEY_W) == glfw.PRESS):
            self.camera.camera_pos += self.camera_speed * self.camera.camera_front
            self.camera.cal_lookat()
        if (glfw.get_key(window, glfw.KEY_S) == glfw.PRESS):
            self.camera.camera_pos -= self.camera_speed * self.camera.camera_front
            self.camera.cal_lookat()
        if (glfw.get_key(window, glfw.KEY_A) == glfw.PRESS):
            self.camera.camera_pos -= self.camera.camera_right * self.camera_speed
            self.camera.cal_lookat()
        if (glfw.get_key(window, glfw.KEY_D) == glfw.PRESS):
            self.camera.camera_pos += self.camera.camera_right * self.camera_speed
            self.camera.cal_lookat()

    def mouse_callback(self, window, xpos, ypos):
        # xpos, ypos: current cursor location
        if self.first_mouse:
            self.last_x, self.last_y = xpos, ypos
            self.first_mouse = False
        
        x_offset = xpos - self.last_x
        y_offset = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos
        
        x_offset *= self.sensitivity
        y_offset *= self.sensitivity

        self.yaw += x_offset
        self.pitch += y_offset

        # to avoid problems
        if self.pitch > 89.0:
            self.pitch = 89.0
        elif self.pitch < -89.0:
            self.pitch = -89.0
        
        front = glm.vec3()
        front.x = cos(glm.radians(self.yaw)) * cos(glm.radians(self.pitch))
        front.y = sin(glm.radians(self.pitch))
        front.z = sin(glm.radians(self.yaw)) * cos(glm.radians(self.pitch))
        self.camera.camera_front = glm.normalize(front)
        self.camera.cal_lookat()
        return None
    
    def scroll_callback(self, window, x_offset, y_offset):
        # change field of view
        # y offset:  wheel
        fov = self.camera.fov
        if (fov >= 10.0 and fov <= 60.0):
            fov -= y_offset

        if (fov <= 10.0):
            fov = 10.0
        if (fov >= 60.0):
            fov = 60.0
        self.camera.set_fov(fov)
        return None