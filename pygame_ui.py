import pygame
import time
import style
from camera_stream import Stream
from drawer import EasyDrawer
from models import HandLandmarkContainer, PoseLandmarkContainer


pygame.init()
window_width, window_height = 0,0
screen = pygame.display.set_mode((window_width,window_height), vsync=1)
clock = pygame.time.Clock()
running = True

def convert_cv_to_pygame(cv_image):
    return pygame.image.frombuffer(cv_image.tostring(), 
                                   cv_image.shape[-2::-1], 
                                   "RGB")

camera = Stream()
drawer = EasyDrawer(EasyDrawer.PYGAME)
feed = camera.stream()
hand0 = HandLandmarkContainer()

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    feed_image = next(feed)
    hand0.detect_async(feed_image, q:=int(1000*time.time()))
    current_window_size = pygame.display.get_window_size()

    use_image = convert_cv_to_pygame(feed_image)
    use_image = pygame.transform.scale(use_image, current_window_size)
    use_image = pygame.transform.flip(use_image, 1, 0)

    # fill the screen with a color to wipe away anything from last frame
    screen.blit(use_image, (0,0))
    
    # RENDER YOUR GAME HERE
    drawer.set_image(screen)
    hand0.set_display(screen)
    hand0.flip_axes('x')
    hand0.draw(int(time.time()*1000))

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

camera.stop_stream()
hand0.close()
with open(f'Storage.txt','w') as text:
    for timestamp, data in zip(hand0.timestamp_storage, hand0.data_storage):
        text.write(f'[{timestamp},{data}]\n')
pygame.quit()
