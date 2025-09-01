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

pygame.init()
window_width, window_height = 0,0
screen = pygame.display.set_mode((window_width,window_height), 
                                 flags=pygame.RESIZABLE, vsync=1)
clock = pygame.time.Clock()

def convert_cv_to_pygame(cv_image):
    return pygame.image.frombuffer(cv_image.tobytes(), 
                                   cv_image.shape[-2::-1], 
                                   "RGB")