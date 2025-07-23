import numpy as np
import enum

class Toolbox:
    dot_product = lambda v1, v2: sum([e1*e2 for e1,e2 in zip(v1,v2)])
    vector_magnitude = lambda v: sum([e*e for e in v])**0.5
    normalise_vector = lambda v, magnitude: [e/magnitude for e in v]
    normalise_vector2 = lambda v: [e/Toolbox.vector_magnitude(v) for e in v]
    clamp = lambda v,min_out,max_out: min_out if v<=min_out else max_out if v>=max_out else v
    lerp = lambda v,min_in,max_in,min_out,max_out: ((v-min_in)/(max_in-min_in))*(max_out/min_out) + min_out
    make_vector = lambda p1, p2: [e2-e1 for e1,e2 in zip(p1,p2)]
    invert_vector = lambda v:[-el for el in v]
    middle_point = lambda v1,v2:[(e1+e2)//2 for e1,e2 in zip(v1,v2)]

    def mask_factor(v1:tuple|list, v2:tuple|list, factors):
        new_vec = []
        if isinstance(factors, (int, float)):
            factors = [factors for i in range(len(v1))]
        for e1, e2, factor in zip(v1,v2,factors):
            new_vec += [Toolbox.lerp(factor, 0, 1, e1, e2)]
        return new_vec

    def cross_product(v1,v2):
        for v in (v1,v2):
            if len(v)<3:
                v+=[0]
        return [v1[1]*v2[2]-v1[2]*v2[1], 
                v1[2]*v2[0]-v1[0]*v2[2], 
                v1[0]*v2[1]-v1[1]*v2[0]]
    
    def angle_vector(v1, v2) -> list:
        '''
        Returns angle in the following order:
        around x-axis, y-axis, z-axis

        params:
         - v1: vector 1
         - v2: vector 2
        '''
        normalise_vector = Toolbox.normalise_vector2
        dot_product = Toolbox.dot_product
        nv1, nv2 = normalise_vector(v1), normalise_vector(v2)
        angles = [np.arccos(dot_product(*[
            [0 if idx==i else el for i,el in enumerate(vec)] 
            for vec in (nv1,nv2)])) * 180/np.pi
            for idx in [0,1,2]]
        return [float(f'{angle:0.2f}') for angle in angles]
        
    def angle_point(p1, p2, p3) -> list:
        '''
        Returns angle in the following order:
        around x-axis, y-axis, z-axis

        params:
         - p1: point 1
         - p2: point 2
         - p3: point 3
        '''
        make_vector = Toolbox.make_vector
        angle_vector = Toolbox.angle_vector
        v1, v2 = make_vector(p2,p1), make_vector(p2,p3)
        return angle_vector(v1, v2)

    def displacement(p1, p2) -> list:
        make_vector = Toolbox.make_vector
        vectors = [make_vector(*[[0 if idx!=i else el for i, el in enumerate(p)] 
                    for p in (p1,p2)])
                    for idx in [0,1,2]]
        return [float(f'{vector:0.2f}' for vector in vectors)]

    def distance(p1, p2) -> list:
        vector_magnitude = Toolbox.vector_magnitude
        magnitudes = [vector_magnitude(*[[0 if idx==i else el for i,el in enumerate(p)]
                   for p in (p1,p2)]) 
                  for idx in [2,0,1,3]]
        return [float(f'{magnitude:0.2f}') for magnitude in magnitudes]

    def compare(val1, val2):
        return [float(f'{e1/e2:0.2f}') for e1,e2 in zip(val1, val2)]

    def rotator(point, rotation_value, origin = [0,0,0]):
        cos = np.cos
        sin = np.sin

        x_rot, y_rot, z_rot = [float(f'{r_val*np.pi/180}') 
                               for r_val 
                               in rotation_value]
        
        point = [el_p-el_o for el_p, el_o in zip(point, origin)]
        
        # x rotation
        point = [point[0], 
                 cos(x_rot)*point[1] - sin(x_rot)*point[2],  
                 sin(x_rot)*point[1] + cos(x_rot)*point[2]]

        # y rotation
        point = [cos(y_rot)*point[0] - sin(y_rot)*point[2],
                 point[1],
                 sin(y_rot)*point[0] + cos(y_rot)*point[2]]

        # z rotation
        point = [cos(z_rot)*point[0] - sin(z_rot)*point[1],
                 sin(z_rot)*point[0] + cos(z_rot)*point[1],
                 point[2]]
        
        point = [el_p+el_o for el_p, el_o in zip(point, origin)]
        
        return point

    def bounding_box(*points):
        a = list()
        return
    
    def bounding_box_size(*points):
        return 
    
if __name__=='__main__':
    pass