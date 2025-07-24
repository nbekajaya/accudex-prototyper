import pygame
import time
import style
from camera_stream import Stream
from drawer import EasyDrawer
from models import ModelIndices, LandmarkContainer, AlternateLandmarks


pygame.init()
window_width, window_height = 0,0
screen = pygame.display.set_mode((window_width,window_height), flags=pygame.RESIZABLE, vsync=1)
clock = pygame.time.Clock()
running = True

def convert_cv_to_pygame(cv_image):
    return pygame.image.frombuffer(cv_image.tostring(), 
                                   cv_image.shape[-2::-1], 
                                   "RGB")

camera = Stream(1)
drawer = EasyDrawer(EasyDrawer.PYGAME)
feed = camera.stream()
pose = LandmarkContainer(ModelIndices.POSE_MODEL, options={'num_poses':1}, renderer = EasyDrawer.CV)
hand = LandmarkContainer(ModelIndices.HAND_MODEL, options={'num_hands':2}, renderer = EasyDrawer.CV)

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    use_image = next(feed)
    # print(use_image.shape)

    current_time = int(800*time.time())

    pose.detect_async(use_image, current_time)
    hand.detect_async(use_image, current_time)

    pose.reorder_landmarks(AlternateLandmarks.DRESIO|AlternateLandmarks.NAT_CUSTOM)

    pose.flip_axes('x')
    hand.flip_axes('x')

    pose.localise_vectors(('index',28,29),('index',30,13))
    pose.relative_displace(39)

    current_time = int(800*time.time())

    pose.set_display(use_image, flip=True)    
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

camera.stop_stream()
pose.close()
hand.close()

# with open(f'Storage.txt','w') as text:
#     for timestamp, data in zip(hand0.timestamp_storage, hand0.data_storage):
#         text.write(f'[{timestamp},{data}]\n')
pygame.quit()
