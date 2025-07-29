import pygame
import time
import numpy as np
import style
from camera_stream import Stream
from drawer import EasyDrawer
from models import ModelIndices, LandmarkContainer, AlternateLandmarks
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


pygame.init()
window_width, window_height = 0,0
screen = pygame.display.set_mode((window_width,window_height), flags=pygame.RESIZABLE, vsync=1)
clock = pygame.time.Clock()
running = True

def convert_cv_to_pygame(cv_image):
    return pygame.image.frombuffer(cv_image.tobytes(), 
                                   cv_image.shape[-2::-1], 
                                   "RGB")

camera = Stream(1)
drawer = EasyDrawer(EasyDrawer.PYGAME)
feed = camera.stream()
pose = LandmarkContainer(ModelIndices.POSE_MODEL, options={'num_poses':1}, renderer = EasyDrawer.CV)
hand = LandmarkContainer(ModelIndices.HAND_MODEL, options={'num_hands':2}, renderer = EasyDrawer.CV)
do_calibrate = False

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            
            if event.key == pygame.K_v:
                do_calibrate = True
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 3:
                pass

        

    use_image = next(feed)

    current_time = int(800*time.time())

    pose.detect_async(use_image, current_time)
    hand.detect_async(use_image, current_time)

    pose.reorder_landmarks(AlternateLandmarks.DRESIO
                           |AlternateLandmarks.NAT_CUSTOM
                           |AlternateLandmarks.SPINE
                           |{99:'11,23 0.5 mid left',100:'12,24 0.5 mid right'})

    pose.flip_axes('x')
    hand.flip_axes('x')

    # try:
    #     print(len(pose.landmark_list[0]))
    # except IndexError:
    #     pass

    # pose.localise_vectors(('index',28,29),('index',30,13))
    # pose.relative_displace(39)
    try:
        pose.landmark_connections += [[30,45], [45,44], [44,43], [43,13]]
    except IndexError: pass
    pose.measure('displacement screen - 11,12 -')

    if do_calibrate:
        pose.calibrate()

    # Nose Plotting stuff
    fig, ax = plt.subplots()
    ax.set_ylim(0,1)
    nose_positions = []
    for stored in pose.data_storage:
        # pass
        for group in stored:
            nose_positions += [group[0].screen[1]]
            # pass
    # ax.plot(nose_positions)
    # canvas = FigureCanvasAgg(fig)
    # canvas.draw()
    # image_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    # plt.close()
    # image_plot = image_plot.reshape(fig.canvas.get_width_height()[::-1]+(4,))[:,:,:3]
    # use_image[:960,:1280,:]=image_plot

    current_time = int(800*time.time())

    pose.set_display(use_image, flip=True)    
    try:
     for measure in pose.measured[0]:
        pose.renderer.render_text(measure, color=style.FontColorBlack, position= pose.renderer.image_center_left,
                                  font_thickness=2)
    except IndexError:
        pass

    try:
        assert pose.calibrated_groups
        pose.renderer.render_text("Calibrated", (10,200), color=style.FontColorRed, font_thickness=2)
    except AttributeError:
        pass

    use_image = pose.draw(current_time, flipped=True)
    
    hand.set_display(use_image, flip=True)
    use_image = hand.draw(current_time, connector='bone', flipped=True)

    use_image = convert_cv_to_pygame(use_image)
    current_window_size = pygame.display.get_window_size()
    use_image = pygame.transform.scale(use_image, current_window_size)
    use_image = pygame.transform.flip(use_image, 1, 0)

    # fill the screen with a color to wipe away anything from last frame
    screen.blit(use_image, (0,0))
    
    # RENDER YOUR GAME HERE
    drawer.set_image(screen)

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(30)  # limits FPS to 60
    do_calibrate = False

camera.stop_stream()
pose.close()
hand.close()

# with open(f'Storage.txt','w') as text:
#     for timestamp, data in zip(hand0.timestamp_storage, hand0.data_storage):
#         text.write(f'[{timestamp},{data}]\n')
pygame.quit()
