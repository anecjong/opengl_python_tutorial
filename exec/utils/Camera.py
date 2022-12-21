import glm

class Camera:
    '''
    camera projection and loo
    '''
    def __init__(self, camera_pos=None, camera_front=None, world_up=None) -> None:
        if camera_pos is None:
            camera_pos = glm.vec3(0.0, 0.0, 3.0)
        if camera_front is None:
            camera_front = glm.vec3(0.0, 0.0, -1.0)
        if world_up is None:
            world_up = glm.vec3(0.0, 1.0, 0.0)
            
        self.camera_pos = camera_pos
        self.camera_front = camera_front
        self.world_up = world_up
        self.fov = 45.0
        self.camera_right = glm.normalize(glm.cross(world_up, camera_pos - (self.camera_pos + self.camera_front)))
        self.lookat = glm.lookAt(self.camera_pos, self.camera_pos + self.camera_front, self.world_up)

        # projection
        self.projection = glm.perspective(glm.radians(self.fov), 800.0/800.0, 0.1, 100.0)

    def cal_lookat(self, ):
        self.camera_right = glm.normalize(glm.cross(self.world_up, self.camera_pos - (self.camera_pos + self.camera_front)))
        self.lookat = glm.lookAt(self.camera_pos, self.camera_pos + self.camera_front, self.world_up)
    
    def change_camera_pos(self, camera_pos):
        self.camera_pos = camera_pos
        self.lookat = glm.lookAt(self.camera_pos, self.camera_pos + self.camera_front, self.world_up)
        return None

    def change_camera_front(self, camera_front):
        self.camera_front = camera_front
        self.lookat = glm.lookAt(self.camera_pos, self.camera_pos + self.camera_front, self.world_up)
        return None

    def change_camera_right(self, camera_right):
        self.camera_right = camera_right
        self.lookat = glm.lookAt(self.camera_pos, self.camera_pos + self.camera_front, self.world_up)
        return None
    
    def set_fov(self, fov: float):
        self.fov = fov
        self.cal_projection()
        return None
    
    def cal_projection(self, ):
        self.projection = glm.perspective(glm.radians(self.fov), 800.0/800.0, 0.1, 100.0)
        return None
    
    def get_lookAt(self,):
        return self.lookat
    
    def get_projection(self, ):
        return self.projection
    
    def get_camera_pos(self, ):
        return self.camera_pos

    def get_camera_front(self, ):
        return self.camera_front

    def get_camera_right(self, ):
        return self.camera_right