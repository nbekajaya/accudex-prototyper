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
    
    def rot_matrix_to_elemental(R:np.ndarray) -> list:
        """Turns a rotation matrix represented as an ndarray to 
        angular elemental rotation in a list

        Args:
            R (np.ndarray): Rotation matrix

        Returns:
            list: Rotation around x,y,z axes respectively
        """
        to_deg = 180/np.pi
        theta_x = np.arctan2(R[2][1],R[2][2]) * to_deg
        theta_y = np.arctan2(
            -R[2][0], np.sqrt( np.power(R[2][1],2) + np.power(R[2][2],2) ) 
        ) * to_deg
        theta_z = np.arctan2(R[1][0], R[0][0]) * to_deg

        return [abs(theta_x), abs(theta_y), abs(theta_z)]

    def find_rot_matrix(v1:list|tuple, v2:list|tuple) -> np.ndarray:
        """Calculates the rotation matrix of two vectors
        Adapted from: 
        https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311 

        Args:
            v1 (list | tuple): A 3-dimensional vector, with elements x,y,z.
            v2 (list | tuple): A 3-dimensional vector, with elements x,y,z.

        Returns:
            np.ndarray: A 3x3 matrix represented by an ndarray as the rotation matrix
        """
        normalise_vector = Toolbox.normalise_vector2
        vector_magnitude = Toolbox.vector_magnitude
        dot_product = np.dot
        cross_product = np.cross

        for vector in v1, v2:
            if len(vector) < 3:
                vector.append(0)
            if len(vector) > 3:
                raise ValueError(f"Too many elements in vector {vector}; expected 3 elements at most!")
        
        v = cross_product(normalise_vector(v1),normalise_vector(v2))
        s = vector_magnitude(v)

        if np.isclose([s], [0], 1e-6):
            return np.identity(3)
        
        c = dot_product(v1,v2)
        vx = np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]]
        )
        vx2 = dot_product(vx,vx)
        i = np.identity(3)
        R = i + vx + vx2 * (1-c)/(s*s)
        
        return R
    
    def angle_vector(v1:list, v2:list) -> list:
        '''
        Returns angle in the following order:
        around x-axis, y-axis, z-axis

        params:
         - v1: vector 1
         - v2: vector 2
        '''
        normalise_vector = Toolbox.normalise_vector2
        dot_product = np.dot

        normalised_vectors = []
        for vector in (v1,v2):
            try:
                normalised_vectors.append(normalise_vector(vector))
            except Exception as e:
                raise Exception(f'Failed vector normalisation of {vector} ', e)
        nv1, nv2 = normalised_vectors

        yaw_pitch_roll = Toolbox.rot_matrix_to_elemental(Toolbox.find_rot_matrix(nv1,nv2))
        general = np.arccos(dot_product(nv1,nv2)) * 180/np.pi
        angles = yaw_pitch_roll + [general]

        return angles
        
    def angle_point(p1:list|tuple, p2:list|tuple, p3:list|tuple) -> list:
        '''Calculates angle of lines p1-p2 and p2-p3, with p2 as the corner
        Returns angle in the following order:
        around x-axis, y-axis, z-axis, and general angle

        Args:
            p1 (list | tuple): x,y,z coordinates of point 1.
            p2 (list | tuple): x,y,z coordinates of point 2.
            p3 (list | tuple): x,y,z coordinates of point 3.
        
        Returns:
            list: A list of angular values about each axis and general angle
        '''
        make_vector = Toolbox.make_vector
        angle_vector = Toolbox.angle_vector

        vectors = []
        for endpoints in ((p2,p1),(p2,p3)):
            try:
                vectors.append(make_vector(*endpoints))
            except Exception as e:
                raise Exception(f'Failed vector making of {endpoints} ', e)
        v1, v2 = vectors

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

        try:
            vector = make_vector(p1,p2)
        except Exception as e:
            raise Exception(f'Failed vector making of {(p1,p2)} ', e)
        
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

        magnitudes = []
        for idx in (2,0,1,3):
            tmp_point1 = [0 if idx==i else el for i,el in enumerate(p1)]
            tmp_point2 = [0 if idx==i else el for i,el in enumerate(p2)]

            try:
                vector = make_vector(tmp_point1, tmp_point2)
            except Exception as e:
                raise Exception(f'Failed vector making of points {(tmp_point1,tmp_point2)} ', e)

            magnitudes.append(vector_magnitude(vector))
        
        return magnitudes
    
    def absolute_displacement(p1,p2):
        displacement = Toolbox.displacement(p1,p2)
        return [abs(element) for element in displacement]

    def compare_ratio(val1:tuple|list, val2:tuple|list) -> list:
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
        
        result = []

        for e1, e2 in zip(val1,val2):
            if float(e2) == 0.0:
                result.append(0)
            result.append(e1/e2)

        return result
    
    def compare_substract(val1:tuple|list, val2:tuple|list) -> list:
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
        return [e2-e1 for e1,e2 in zip(val1,val2)]

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
        return [bounding_point 
                for bounding_point in zip(*[(min(element), max(element)) for element in zip(*points)][:2])]
    
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