import numpy as np
import cv2 as cv
import enum

class Measurement:
    def __init__(self, **kwargs):
        for k,v in kwargs.items:
            setattr(self,k,v)

class Toolbox:
    dot_product = lambda v1, v2: sum([e1*e2 for e1,e2 in zip(v1,v2)])
    vector_magnitude = lambda v: sum([e*e for e in v])**0.5
    normalise_vector = lambda v, magnitude: [e/magnitude for e in v]
    normalise_vector2 = lambda v: [Toolbox.vector_magnitude(v) and e/Toolbox.vector_magnitude(v) or 0 
                                   for e in v] 
    clamp = lambda v,min_out,max_out: min_out if v<=min_out else max_out if v>=max_out else v
    lerp = lambda v,min_in,max_in,min_out,max_out: ((v-min_in)/(max_in-min_in))*(max_out-min_out) + min_out
    make_vector = lambda p1, p2: [e2-e1 for e1,e2 in zip(p1,p2)]
    invert_vector = lambda v:[-el for el in v]
    color_convert = lambda x, space:[int(x) for x in cv.cvtColor(np.array([[x]],np.uint8), space)[0][0]]
    middle_point = lambda v1,v2:[(e1+e2)//2 for e1,e2 in zip(v1,v2)]

    def mask_factor(v1:tuple|list, v2:tuple|list, factors):
        '''
        Does 3D Cross product for 2 vectors

        Params:
         - v1(tuple|list): an interable acting as a vector
         - v2(tuple|list): an interable acting as a vector

        Returns:
         A list acting as the cross product vector
        '''
        new_vec = []
        if isinstance(factors, (int, float)):
            factors = [factors for i in range(len(v1))]
        for e1, e2, factor in zip(v1,v2,factors):
            new_vec += [Toolbox.lerp(factor, 0, 1, e1, e2)]
        return new_vec

    def cross_product(v1:tuple|list, v2:tuple|list) -> list:
        '''
        Does 3D Cross product for 2 vectors

        Params:
         - v1(tuple|list): an interable acting as a vector
         - v2(tuple|list): an interable acting as a vector

        Returns:
         A list acting as the cross product vector
        '''
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
            normalise_vector([0 if idx==i else el 
                                       for i,el 
                                       in enumerate(vec)]) 
            for vec in (nv1,nv2)])) * 180/np.pi
            for idx in [0,1,2,3]]
        return angles
        
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

    def displacement(p1:tuple|list, p2:tuple|list) -> list:
        '''
        Finds displacement between two points

        Axes listed in order of:
         - x
         - y
         - z

        Params:
         - p1(tuple|list): an iterable acting as a coordinate
         - p2(tuple|list): an iterable acting as a coordinate
        
        Returns:
         A list describing the displacement of the two points in order of the axess
        '''
        make_vector = Toolbox.make_vector
        vector = make_vector(p1,p2)
        return vector

    def distance(p1:tuple|list, p2:tuple|list) -> list:
        '''
        Finds the distance between two points
        in specified planes

        Planes listed in order of:
         - xy
         - yz
         - xz
         - xyz

        Params:
         - p1(tuple|list): an iterable acting as a coordinate
         - p2(tuple|list): an iterable acting as a coordinate
        
        Returns:
         A list describing the displacement of the two points in order of the planes
        '''
        vector_magnitude = Toolbox.vector_magnitude
        make_vector = Toolbox.make_vector
        magnitudes =[vector_magnitude(
                        make_vector(*[[0 if idx==i else el 
                                       for i,el in enumerate(p)]
                                    for p in (p1,p2)])) 
                    for idx in [2,0,1,3]]
        return magnitudes
    
    def absolute_displacement(p1,p2):
        displacement = Toolbox.displacement(p1,p2)
        return [abs(element) for element in displacement]

    def compare(val1:tuple|list, val2:tuple|list) -> list:
        '''
        Compares elements between two values as a ratio

        Params:
         - val1(any): A list containing values
         - val2(any): A list containing values 
        
        Returns:
         A list which are ratios of elements of val1 to elements of val 2
        '''
        if len(val1) != len(val2):
            raise ValueError('Incompatible comparison between lists of unequal lengths')
        return [e1/e2 for e1,e2 in zip(val1, val2)]

    def rotator(point:tuple|list, rotation_value:tuple|list, origin:tuple|list = [0,0,0]) -> list:
        '''
        Rotates a point around origin using rotation_value
        as elemental rotations

        rotation_value order:
         - x
         - y
         - z
        
        Params:
         - point(tuple|list): An iterable acting as a coordinate; The point to be rotated
         - rotation_value(tuple|list): An iterable that describes the elemental rotation axes
         - origin(tuple|list): An iterable acting as a coordinate, defaults to (0,0,0); The origin of rotation

        Returns:
         A list describing the new position of the point
        '''
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

    def bounding_box(points:list[list]):
        '''
        Returns the bounding box of points 

        Params:
         - points: a list of lists acting as coordinates
        
        Returns:
         Top left and bottom right coordinates
        '''
        return [bounding_point for bounding_point in zip(*[(min(element), max(element)) for element in zip(*points)][:2])]
    
    def bounding_box_size(*points):
        return 
    
    def color_lerp(value, value_min, value_max, color_min, color_max):
        color_convert = Toolbox.color_convert
        clamp = Toolbox.clamp
        lerp = Toolbox.lerp
        c1, c2 = [color_convert(color, cv.COLOR_BGR2HSV) for color in (color_min, color_max)]
        t = clamp(lerp(value, value_min, value_max, 0, 1), 0, 1)
        return tuple(color_convert([int(lerp(t, 0, 1, el1, el2)) for el1, el2 in zip(c1,c2)],
                                cv.COLOR_HSV2BGR))
    
    def coordinate_weighter(coordinates, weights):
        return [sum(positions) 
                for positions 
                in zip(*[[el*weight for el in coordinate] 
                         for coordinate, weight in zip(coordinates,weights)])]
    
    def set_range(value, value_min, value_max):
        if value_min > value_max:
            return int(value_min > value > value_max)
        return int(value_min < value < value_max)
    
    def value_discriminator(value:int|float, param:dict) -> int:
        """Discriminates a value between two boundaries

        Args:
            value (int | float): The value
            param (dict): A dictionary containing either 'lower_bound' key, 'upper_bound' key, both or neither

        Returns:
            int: 0 for No values, 1 for Maybe Values, 2 for Yes values
        """
        if all([x not in param for x in ('lower_bound','upper_bound')]):
            return 2
        
        if 'lower_bound' not in param:
            return value < param['upper_bound'] and 2 or 0
        
        if 'upper_bound' not in param:
            return value > param['lower_bound'] and 2 or 0
        
        lower, upper = param['lower_bound'], param['upper_bound']

        if lower < upper:
            return lower < value < upper and 1 or upper < value and 2 or 0
        if lower > upper:
            return int(not(lower < value < upper))


if __name__=='__main__':
    pass