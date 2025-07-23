import cv2 as cv
import mediapipe as mp
import time

class Stream:
    def __init__(self, camera_index=1):
        self.camera = cv.VideoCapture(camera_index)

        camera = cv.VideoCapture(camera_index)
        while camera.isOpened():
            status, frame = camera.read()
            self.shape = frame.shape
            break
        camera.release()
    
    def stream(self):
        while self.camera.isOpened():
            status, frame = self.camera.read()
            if not status:
                raise RuntimeError("Camera fails to capture frame")
            yield cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
    def stop_stream(self):
        self.camera.release()



if __name__=='__main__':
    my_camera = Stream(1)
    streamer = my_camera.stream()
    x = 0
    while x<10:
        next(streamer)
        print(my_camera.shape)
        x+=1
    my_camera.stop_stream()