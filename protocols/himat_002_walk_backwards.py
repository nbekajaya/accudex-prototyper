import sys
import os

# HANDLING IMPORTS
SCRIPT_DIR = os.path.realpath(__file__)
USE_SCRIPT_DIR = SCRIPT_DIR
while not (os.path.split(USE_SCRIPT_DIR)[-1].lower() == 'accudex-prototyper'):
    if USE_SCRIPT_DIR == '/':
        break
    USE_SCRIPT_DIR = os.path.dirname(USE_SCRIPT_DIR)
print(USE_SCRIPT_DIR)
sys.path.append(USE_SCRIPT_DIR)
# quit()

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
from models import LandmarkContainer
from models_utils import ModelIndices, AlternateLandmarks
from toolbox import Toolbox
from data_processing import recording_info_to_pd_dataframe, write_to_excel

# PYGAME INITIALISATION
pygame.init()
window_width, window_height = 0,0
screen = pygame.display.set_mode((window_width,window_height), 
                                 flags=pygame.RESIZABLE, vsync=1)
clock = pygame.time.Clock()

def convert_cv_to_pygame(cv_image):
    return pygame.image.frombuffer(cv_image.tobytes(), 
                                   cv_image.shape[-2::-1], 
                                   "RGB")

# CAMERA SYSTEM
camera = Stream(1)
feed = camera.stream()

# MODEL INFO
pose = LandmarkContainer(ModelIndices.POSE_MODEL,
                         options={'num_poses':1},
                         renderer=EasyDrawer.CV)
pose.set_landmarks(
    AlternateLandmarks.DRESIO
    + AlternateLandmarks.NAT_CUSTOM
    + AlternateLandmarks.CENTER_OF_MASS
)
pose.flip_axes('x')
pose.set_assessment(mprot.HIMAT_WALK_BACKWARDSD)

# LOOP VARIABLES
do_calibrate = do_multi_calibrate = do_record = False

recording_prompt = 'START'
recording_color = style.FontColorOrange

calibration_prompt= 'SET'

multi_calibration_prompt = 'ADD'

recording_msg_addition = calibration_msg_addition = multi_calibration_msg_addition = ''
recording = []
check_time = 0
position_check = []

# PROGRAM LOOP
running = True
start_time = int(time.time()*1000)
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
    # hand.detect_async(feed_image, current_time)

    if not hasattr(pose, 'landmark_list'):
        continue

    real_time = int(1000*time.time()) - start_time

    pose.set_display(feed_image, flip=True)    
    # pose.renderer.fill_image((50,40,130))
    
    if current_time > check_time:
        try:
            test = [
                0.9<x<1.1 
                for x 
                in Toolbox.compare_ratio(pose.measured[0][2]['result'], position_check)
            ]
            comparison = all(test)
        except Exception as e:
            print(f'Assessment checking returned error: {e}')
            test = None
            comparison = None
        print(f'assessment_checking: {current_time}, {comparison}')

        if comparison:
            assessment_measure = getattr(pose, 'multi_calibrated_groups', [])
            print(f"TEST: assessment_measure length {len(assessment_measure)}")

            # if len(assessment_measure) < 2:
            #     do_multi_calibrate = True
            
            if len(assessment_measure) == 2:
                param_0 = pose.measured[0][-2]
                param_1 = pose.measured[0][-1]
                param_0_test = 0.9<param_0['result'][param_0['params']['check_index']]<1.1
                param_1_test = 0.9<param_1['result'][param_1['params']['check_index']]<1.1

                if param_0_test:
                    print('TEST: RECORDING START')
                    do_record = True

                if param_1_test:
                    print('TEST: RECORDING END')
                    do_record = False
                else:
                    pass
        else:
            try:
                position_check = pose.measured[0][2]['result']
            except IndexError:
                pass
        check_time = current_time+1250
        

    if do_calibrate:
        pose.calibrate(current_time)
        calibration_prompt = 'RESET'
        calibration_msg_addition = f"; CALIBRATED AT {current_time/1000:0.2f}"
    
    if do_multi_calibrate:
        pose.multi_calibrate(current_time, 4)
        multi_calibration_msg_addition = f"; {len(pose.multi_calibrated_groups)} CALIBRATIONS SET"
    
    if do_record:
        pose.record_data()
        recording_prompt = 'STOP'
        recording_msg_addition = f"; {(current_time)/1000:08.2f}s" 
        recording_color = (80,240,140)

    # DEBUGGING TIMER CHECKER
    # pose.renderer.render_text(
    #     f'{test}', 
    #     (1400, 65), 
    #     displacer=(0,35),
    #     color=recording_color, 
    #     scale=1.2, 
    #     font_thickness=2
    # )
    # pose.renderer.render_text(
    #     f'{current_time} {check_time}', 
    #     (1700, 30), 
    #     displacer=(0,35),
    #     color=recording_color, 
    #     scale=1.2, 
    #     font_thickness=2
    # )

    pose.renderer.render_text(
        f"'R' TO {recording_prompt} RECORDING"+recording_msg_addition, 
        (10, 30), 
        displacer=(0,35),
        color=recording_color, 
        scale=1.2, 
        font_thickness=2
    )
    pose.renderer.render_text(
        f"'V' TO {calibration_prompt} CALIBRATION"+calibration_msg_addition, 
        (10, 30), 
        displacer=(0,35),
        color=style.FontColorOrange, 
        scale=1.2, 
        font_thickness=2
    )
    pose.renderer.render_text(
        f"'B' TO ADD MULTI CALIBRATION POINT"+multi_calibration_msg_addition, 
        (10, 30), 
        displacer=(0,35),
        color=style.FontColorOrange, 
        scale=1.2, 
        font_thickness=2
    )
        

    # FINAL DRAWING
    use_image = pose.draw(real_time, 
                          draw_measurements=True, 
                          attributes='',
                          flipped=True)
    
    # hand.set_display(use_image, flip=True)
    # use_image = hand.draw(real_time, 
    #                       connector='bone', 
    #                       flipped=True)
    
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
    recording_prompt, calibration_prompt = 'START', 'SET'
    recording_color = style.FontColorOrange

pose.close()
pygame.quit()
camera.stop_stream()