from toolbox import Toolbox
from style import *

class EasyDrawer:
    CV = 0
    PYGAME = 1

    def __init__(self, renderer:int):
        '''
        Picks renderer:
        EasyDrawer.PYGAMEDRAW or EasyDrawer.CV
        '''
        if renderer == EasyDrawer.CV :
            import cv2 as cv
            self.renderer = cv 

        if renderer == EasyDrawer.PYGAME:
            import pygame
            self.renderer = pygame

        self.renderer_index = renderer
        
    def set_image(self, image, flip = False):
        self.image = image
        self.positions, self.counters = [],[]

        if self.renderer_index==EasyDrawer.PYGAME:
            self.image_info = image.get_size()
        if self.renderer_index==EasyDrawer.CV:
            self.image_info = image.shape[-2::-1]

        if flip:
            self.flip_render()

        self.image_center = [el//2 for el in self.image_info]
        self.image_top_left = [0,0]
        self.image_top_right = [self.image_info[0],0]
        self.image_top_center = [self.image_center[0],0]
        self.image_center_left = [0, self.image_center[1]]
        self.image_center_right = [self.image_info[0], self.image_center[1]]
        self.image_bottom_left = [0,self.image_info[1]]
        self.image_bottom_center = [self.image_center[0], self.image_info[1]]
        self.image_bottom_right = [self.image_info[0], self.image_info[1]]

        self.base_render_scale = min(self.image_info) * 0.0008
        self.base_thickness_scale = int(min(self.image_info) * 0.002)
        self.font_thickness_scale = self.base_thickness_scale//2

    def render_text(self, 
            string, 
            position:tuple = None, 
            displacer:tuple = (0,25),
            color:tuple = FontColorOrange,
            scale:float|int = 1, 
            font_face = FontFace,
            font_thickness = 1):
        renderer = self.renderer
        scale = self.base_render_scale*scale
        font_thickness = self.font_thickness_scale*font_thickness
        
        if position is None:
            position = self.image_center

        if position not in self.positions:
            self.positions.append(position)
            self.counters.append(0)
        pos_index = self.positions.index(position)
        pos_displace = self.counters[pos_index]
        position = tuple([int(p+d*pos_displace) for p,d in zip(position, displacer)])

        if self.renderer_index == EasyDrawer.PYGAME:
            this_font = renderer.font.match_font('arialunicode')
            this_font = renderer.font.Font(this_font, 12)
            this_text = this_font.render(f'{string}', True, color[::-1])
            self.image.blit(this_text, position)
            
        if self.renderer_index == EasyDrawer.CV:
            renderer.putText(self.image, f'{string}', position, 
                    font_face, scale, color, font_thickness)
            
        self.counters[pos_index] = self.counters[pos_index]+1

    def render_circle(self,
               origin:tuple, 
               color:tuple,
               radius:int = 10, 
               thickness = 1):
        renderer = self.renderer
        radius = self.base_render_scale * radius
        thickness = self.base_thickness_scale * thickness


        if self.renderer_index == EasyDrawer.PYGAME:
            renderer.draw.circle(self.image, color[::-1], origin, int(radius), 
                                      int(Toolbox.clamp(thickness,0,100)))
        if self.renderer_index == EasyDrawer.CV:
            renderer.circle(self.image, origin, radius, thickness, color)

    def render_line(self,
             p1:tuple,
             p2:tuple,
             color:tuple = FontColorWhite,
             thickness:int = 1):
        renderer = self.renderer
        thickness = self.base_thickness_scale * thickness
        
        if self.renderer_index==EasyDrawer.PYGAME:
            renderer.draw.line(self.image, color[::-1], p1, p2, int(thickness))
        if self.renderer_index==EasyDrawer.CV:
            renderer.line(self.image, p1, p2, color, int(thickness))

    def render_landmark(self,
                        position:tuple,
                        color1:tuple=FontColorWhite,
                        radius:int=1,
                        thickness:int=1,
                        scale:float=1.0,
                        color2:tuple=None):
        renderer = self.renderer 
        scale = self.base_render_scale * scale 
        thickness =  self.base_thickness_scale * thickness 

        if color2 is None:
            color2=color1

        if self.renderer_index == EasyDrawer.PYGAME:
            renderer.draw.circle(self.image, color2[::-1], position, int(radius*scale), 0)
            renderer.draw.circle(self.image, color1[::-1], position, int(radius*scale*1.8), int(2*scale))
            
        if self.renderer_index == EasyDrawer.CV:
            renderer.circle(self.image, position, int(radius*scale*0.75), color2, -1)
            renderer.circle(self.image, position, int(radius*scale*2.5), color1, thickness)

    def render_bone(self, 
             p1:tuple, 
             p2:tuple, 
             color:tuple=FontColorWhite, 
             thickness:int=1,
             kite_ratio:float=0.35,
             kite_length_ratio:float=0.15):

        vec_dir = Toolbox.make_vector(p1,p2)
        kite_point = [p_el + vec_el * kite_ratio for p_el, vec_el in zip(p1,vec_dir)]
        tangent1 = [-vec_dir[1], vec_dir[0]]
        tangent2 = Toolbox.invert_vector(tangent1)
        p3 = [int(kp + vec_el*kite_length_ratio) for kp, vec_el in zip(kite_point, tangent1)]
        p4 = [int(kp + vec_el*kite_length_ratio) for kp, vec_el in zip(kite_point, tangent2)]
        
        self.render_line(p1, p3, color, thickness)
        self.render_line(p1, p4, color, thickness)
        self.render_line(p3, p2, color, thickness)
        self.render_line(p4, p2, color, thickness)
    
    def flip_render(self):
        renderer = self.renderer
        if self.renderer_index == EasyDrawer.PYGAME:
            self.image = renderer.transform.flip(self.image, 1, 0)
        if self.renderer_index == EasyDrawer.CV:
            self.image = renderer.flip(self.image, 1)


