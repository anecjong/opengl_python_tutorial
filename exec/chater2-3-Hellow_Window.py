import glfw
import OpenGL.GL as gl

def main():
    # glfw initialize
    glfw.init()

    # glfw window hint - version 3.3 for learnopengl document
    # use core profile only
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # glfw.create_window(width, height, window name, ...) -> opengl object
    window = glfw.create_window(800, 600, "LearnOpenGL", None, None)
    if window is None:
        print("Failed to create glfw window")
        glfw.terminate()
        return -1

    # make the context of our window the main context on the current thread
    glfw.make_context_current(window)

    # rendering window size for opengl
    # first 2 parameter: left down
    # 3rd, 4th parameter: width, height
    # width, height maps to (-1, 1)
    gl.glViewport(0, 0, 800, 600)

    # if window size is changed, frame buffer size callback function is called.
    glfw.set_framebuffer_size_callback(window, frame_buffer_size_callback)
    
    # At this point, window is immediatley quit after draw a single image.
    # To keep drawing images and handling user input until the program has been explicitly told to stop,
    # render loop - It keeps on running until we tell GLFW to stop - should be created.
    # window_should_close function checks at the start of each loop iteration if GLFW has been instructed to close.
    # poll_events function checks if any events are triggered.
    # swap_buffers function will swap the color buffer
    while (not glfw.window_should_close(window)):
        # check input
        process_input(window)

        # rendering
        # glClearColor: state-setting function, here clear color is red, green, blue, opacity
        # glClear: clear buffer
        # GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_STENCIL_BUFFER_BIT
        gl.glClearColor(0.2, 0.6, 0.3, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        # event check and swap buffer - dual buffer
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    # after render loop, all resources should be freed.
    glfw.terminate()

    return 0

def frame_buffer_size_callback(window, width: int, height: int) -> None:
    '''
    if window size change, change opengl Viewport(mapping).
    '''
    gl.glViewport(0, 0, width, height)

def process_input(window):
    '''
    if esc key is pressed, change window_should_close attribute True.
    '''
    # get_key returns glfw.PRESS if key is pressed. If not, glfw.RELEASE is returned.
    if (glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS):
        glfw.set_window_should_close(window, True)

if __name__ == "__main__":
    main()