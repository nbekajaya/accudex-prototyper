import tkinter as tk
import os

prototype_current_path = current_path = os.path.abspath(os.path.dirname(__file__))

def get_screen_info():
    root = tk.Tk('CLOSE ME!')
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    with open(os.path.join(current_path,'screen_info.txt'),'w') as save_file:
        save_file.write(f'{screen_width} {screen_height}')
        
    return screen_width, screen_height

def load_screen_info():
    with open(os.path.join(current_path,'screen_info.txt')) as save_file:
        screen_width, screen_height = [int(value) for value in save_file.read().split()]
    return screen_width, screen_height



