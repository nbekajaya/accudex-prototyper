import screen_util
import numpy as np
import time
import style
import cv2 as cv
from drawer import EasyDrawer
from camera_stream import Stream
from models import ModelIndices, LandmarkContainer

screen_width, screen_height = screen_util.get_screen_info()
camera = Stream()

screen_middle = [screen_dim//2 for screen_dim in (screen_width,screen_height)]
camera_aspect_ratio = camera.shape[0]/camera.shape[1]

camera_window_ratio = (0.6, 0.6*camera_aspect_ratio)
control_window_ratio = (1-camera_window_ratio[0], camera_window_ratio[1])
skeleton_window_ratio = (0.4, 1-camera_window_ratio[1])

main_dimension = int(camera_window_ratio[0]*screen_width), int(camera_window_ratio[1]*screen_width)
control_dimension = int(control_window_ratio[0]*screen_width), int(control_window_ratio[1]*screen_width)
skeleton_dimension = int(skeleton_window_ratio[0]*screen_width), int(skeleton_window_ratio[1]*screen_width)

main_window_fill = np.zeros((*main_dimension[::-1],3),np.uint8)
control_window_fill = np.zeros((*control_dimension[::-1],3),np.uint8)
skeleton_window_fill = np.zeros((*skeleton_dimension[::-1],3),np.uint8)

main_window = cv.namedWindow('Main', cv.WINDOW_NORMAL)
cv.imshow('Main', main_window_fill)
cv.resizeWindow('Main', *main_dimension)
cv.moveWindow('Main',0, -100)

control_window = cv.namedWindow('Control')
cv.imshow('Control', control_window_fill)
cv.resizeWindow('Control', *control_dimension)
cv.moveWindow('Control', main_dimension[0], -100)

feed = camera.stream()

drawer = EasyDrawer(EasyDrawer.CV)
hand0 = LandmarkContainer(model=ModelIndices.HAND_MODEL,
                        options={'num_hands':2},
                        renderer=EasyDrawer.CV)
pose0 = LandmarkContainer(model=ModelIndices.POSE_MODEL,
                        options={'num_poses':1},
                        renderer=EasyDrawer.CV)

running = True

while running:
    if cv.pollKey() == ord('q'):
        running = False
        
    use_image = next(feed)
    current_time = int(time.time()*1000)

    # hand0.detect_async(use_image, current_time)
    pose0.detect_async(use_image, current_time)

    use_image=cv.flip(use_image,1)
    
    #hand0.flip_axes('x')
    pose0.flip_axes('x')

    pose0.reorder_landmarks(
        {13:'11,12 0.5 mid shoulder',
         16:'11,'}
    )

    drawer.set_image(use_image)
    # hand0.set_display(use_image)
    pose0.set_display(use_image)

    current_time = int(time.time()*1000)
    # print(pose0.landmark_connections)

    # hand0.draw(current_time, connector='bone')
    pose0.draw(current_time,connector='line')

    drawer.render_text(pose0.landmark_connections)

    # hand0.measure('distance world \'xy\' 0 4,20 -')
    # drawer.render_text(hand0.measured, (0,400))

    cv.imshow('Main',cv.cvtColor(use_image, cv.COLOR_RGB2BGR))
    

cv.destroyAllWindows()
cv.waitKey(1)


    
