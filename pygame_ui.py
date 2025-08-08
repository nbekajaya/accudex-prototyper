import pygame
import style
import time

import matplotlib.pyplot as plt
import measurement_protocols as mprot
import numpy as np
import pandas as pd

from camera_stream import Stream
from drawer import EasyDrawer
from matplotlib.backends.backend_agg import FigureCanvasAgg
from models import ModelIndices, LandmarkContainer, AlternateLandmarks
from toolbox import Toolbox
from process_excel import recording_info_to_pd_dataframe, write_and_fix_excel

# INITIALISING PYGAME
pygame.init()
window_width, window_height = 0,0
screen = pygame.display.set_mode((window_width,window_height), 
                                 flags=pygame.RESIZABLE, vsync=1)
clock = pygame.time.Clock()

def convert_cv_to_pygame(cv_image):
    return pygame.image.frombuffer(cv_image.tobytes(), 
                                   cv_image.shape[-2::-1], 
                                   "RGB")

# INITIALISING MODELS AND CAMERA FEED
camera = Stream(1)
feed = camera.stream()
pose = LandmarkContainer(ModelIndices.POSE_MODEL, 
                         options={'num_poses':1}, 
                         renderer = EasyDrawer.CV)
hand = LandmarkContainer(ModelIndices.HAND_MODEL, 
                         options={'num_hands':2}, 
                         renderer = EasyDrawer.CV)

# LOOP VALUES
do_calibrate = do_multi_calibrate = do_record = False
recording = []

start_time = int(time.time()*1000)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            
            if event.key == pygame.K_v:
                do_calibrate = True
            
            if event.key == pygame.K_b:
                do_multi_calibrate = True

            if event.key == pygame.K_r:
                do_record = not(do_record)
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 3:
                pass

    current_time = int(1000*time.time()) - start_time

    # RECEIVING AND PROCESSING IMAGE
    feed_image = next(feed)
    pose.detect_async(feed_image, current_time)
    hand.detect_async(feed_image, current_time)

    # REORDERING AND TRANSFORMING LANDMARKS
    pose.reorder_landmarks(AlternateLandmarks.DRESIO
                           +AlternateLandmarks.NAT_CUSTOM
                           +AlternateLandmarks.TORSO_CENTER
                           +AlternateLandmarks.SPINE
                           +AlternateLandmarks.CENTER_OF_MASS)
    pose.flip_axes('x')
    hand.flip_axes('x')

    try:
        pose.landmark_connections += [[30,400], 
                                      [400,401], 
                                      [401,402], 
                                      [402,13]]
    except IndexError: pass

    pose.measure(*mprot.DEMMI_STAND)
        
    if do_calibrate:
        pose.calibrate(current_time)
    
    if do_multi_calibrate:
        pose.multi_calibrate(current_time)

    real_time = int(1000*time.time()) - start_time

    pose.set_display(feed_image, flip=True)    
    pose.renderer.fill_image((50,40,130))
    
    if do_record:
        pose.renderer.render_text("RECORDING", 
                                  (10, 30), 
                                  color=style.FontColorRed, 
                                  scale=1.2, 
                                  font_thickness=2)
        recording.append([pose.current_processed_timestamp, 
                          pose.landmark_list, 
                          pose.measured])

    # FINAL DRAWING
    use_image = pose.draw(real_time, 
                          draw_measurements=True, 
                          flipped=True)
    
    hand.set_display(use_image, flip=True)
    use_image = hand.draw(real_time, 
                          connector='bone', 
                          flipped=True)
    
    # CONVERTING TO CV AND DISPLAYING IN PYGAME WINDOW
    current_window_size = pygame.display.get_window_size()
    use_image = convert_cv_to_pygame(use_image)
    use_image = pygame.transform.scale(use_image, current_window_size)
    use_image = pygame.transform.flip(use_image, 1, 0)
    screen.blit(use_image, (0,0))

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60
    do_calibrate = do_multi_calibrate = False

# CLOSE STUFF
camera.stop_stream()
pose.close()
hand.close()
pygame.quit()

# WRITES RECORDING TO EXCEL
write_and_fix_excel(dataframe=recording_info_to_pd_dataframe(recording),
                    file_name='TEST_EXPORTS', 
                    directory='./exports')


