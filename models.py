import mediapipe as mp
import numpy as np
from drawer import EasyDrawer
from style import *
from toolbox import Toolbox as toolbox
from copy import deepcopy
from pygame import Surface


# Template landmarking stuff
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

PoseModelPath = './pose_landmarker_lite.task'
HandModelPath = './hand_landmarker.task'

class CustomLandmark:
    '''
    Custom landmark structure for usage
    '''
    
    def __init__(self,
                 screen:list[float], 
                 world:list[float],  
                 idx:int,
                 name:str, 
                 side:str=''):
        self.name = name 
        self.idx = idx
        self.side = side

        try:
            self.screen = [getattr(screen,el) for el in 'xyz']
            self.world = [getattr(world,el)*100 for el in 'xyz']
            self.visibility = (screen.visibility*100,world.visibility*100)
            self.presence = (screen.presence*100,world.presence*100)
        except AttributeError:
            self.screen = screen
            self.world = world
            self.visibility = 0
            self.presence = 0
        
    def flip_axes(self, axes):
        for axis in axes:
            if axis in '0x':
                self.screen[0]=1-self.screen[0]
                self.world[0]=-self.world[0]
            if axis in '1y':
                self.screen[1]=1-self.screen[1]
                self.world[1]=-self.world[1]
            if axis in '2z':
                self.screen[2]=1-self.screen[2]
                self.world[2]=-self.world[2]

    def __repr__(self):
        return f'{self.side}-{self.idx}-{self.name}|screen: {self.screen}|world: {self.world}'
    
class ModelIndices:
    HAND_MODEL = 0
    POSE_MODEL = 1

class AlternateLandmarks:
    DRESIO = {
        13:'11,12 0.5 mid shoulder',
        16:'11,13 0.5 left upper arm',
        17:'12,14 0.5 right upper arm',
        20:'13,15 0.5 left forearm',
        21:'14,16 0.5 right forearm',
        30:'23,24 0.5 mid hip',
        33:'23,25 0.5 left thigh',
        34:'24,26 0.5 right thigh',
        37:'25,27 0.5 left calf',
        38:'26,28 0.5 right calf',
    }

    NAT_CUSTOM = {
        39:'29,30,31,32 0.25,0.25,0.25 mid feet',
        40:'11,12,23,24 0.15,0.15,0.35 navel'
    }

    TORSO_CENTER = {
        41:'11,12,23,24 0.25,0.25,0.25 average torso',
        42:'11,12,23,24 0.3,0.3,0.2 weighted average'
    }

    SPINE = {
        43:'11,12,23,24 0.4,0.4,0.1 spine 1',
        44:'11,12,23,24 0.2,0.2,0.3 spine 2',
        45:'11,12,23,24 0.1,0.1,0.4 spine 3'
    }


class LandmarkConnections:
    HAND_LANDMARK_CONNECTIONS = [
        [0, 1], [0, 5], [0, 17], 
        [1, 2], [2, 3], [3, 4], 
        [5, 6], [5, 9], [6, 7], 
        [7, 8], [9, 10], [9, 13], 
        [10, 11], [11, 12], [13, 14], 
        [13, 17], [14, 15], [15, 16], 
        [17, 18], [18, 19], [19, 20]
    ]

    POSE_LANDMARK_CONNECTIONS = [
        [0, 2], [0, 5], [2, 7], [5, 8], [9, 10], 
        [11, 12], [11, 23], [12, 24], [11, 13], [13, 15], 
        [15, 17], [17, 19], [19, 21], [15, 21], [12, 14], 
        [14, 16], [16, 18], [18, 20], [20, 22], [16, 22], 
        [23, 24], [23, 25], [25, 27], [27, 29], [29, 31], 
        [27, 31], [24, 26], [26, 28], [28, 30], [30, 32], [28, 32]
    ]

class LandmarkNames:
    HAND_LANDMARK_NAMES = [
        'wrist',
        'thumb mcp', 'thumb pip', 'thumb ip', 'thumb tip',
        'index mcp', 'index pip', 'index dip', 'index tip',
        'middle mcp', 'middle pip', 'middle dip', 'middle tip',
        'ring mcp', 'ring pip', 'ring dip', 'ring tip',
        'pinky mcp', 'pinky pip', 'pinky dip', 'pinky tip'
    ]

    POSE_LANDMARK_NAMES = [
        'nose',
        'left eye (inner)', 'left eye', 'left eye (outer)',
        'right eye (inner)', 'right eye', 'right eye (outer)',
        'left ear', 'right ear',
        'mouth (left)', 'mouth (right)',
        'left shoulder', 'right shoulder',
        'left elbow', 'right elbow',
        'left wrist', 'right wrist',
        'left pinky', 'right pinky',
        'left index', 'right index',
        'left thumb', 'right thumb',
        'left hip', 'right hip',
        'left knee', 'right knee',
        'left ankle', 'right ankle',
        'left heel', 'right heel',
        'left foot index', 'right foot index'
    ]

class OtherLandmarkAttributes:
    BODY_WEIGHT_RATIOS = [
        0.08,
        0.5,
        0.007,
        0.007,
        0.016,
        0.016,
        0.027,
        0.027,
        0.015,
        0.015,
        0.045,
        0.045,
        0.1,
        0.1
    ]

class LandmarkContainer:
    model_params = (
        {'model_name':'hand',
         'model_main':HandLandmarker,
         'model_options':HandLandmarkerOptions,
         'model_result':HandLandmarkerResult,
         'model_path':HandModelPath,
         'model_attributes':('handedness', 'hand_landmarks','hand_world_landmarks')},
        {'model_name':'pose',
         'model_main':PoseLandmarker,
         'model_options':PoseLandmarkerOptions,
         'model_result':PoseLandmarkerResult,
         'model_path':PoseModelPath,
         'model_attributes':('pose_landmarks','pose_world_landmarks')},
    )

    GeneralLandmarkerDefaultOptions = {
        'running_mode':VisionRunningMode.LIVE_STREAM,
    }   

    def __construct_model(self, model:int):
        model_params = LandmarkContainer.model_params

        if model>len(model_params):
            raise IndexError("Model enum not recognised!")
        
        self.model_index = model
        model_dict = model_params[model]

        model_dict['landmark_names'] = getattr(LandmarkNames,
                                               f"{model_dict['model_name'].upper()}_LANDMARK_NAMES")
        model_dict['default_landmark_connections'] = getattr(LandmarkConnections,
                                                             f"{model_dict['model_name'].upper()}_LANDMARK_CONNECTIONS")
        
        for k,v in model_params[model].items():
            setattr(self, k, v)

        self.landmark_connections = [x for x in self.default_landmark_connections]

        additional_options = model and {'output_segmentation_masks':True} or {}

        other_options = {
            f'num_{self.model_name}s':2,
            f'min_{self.model_name}_detection_confidence':0.3,
            f'min_{self.model_name}_presence_confidence':0.3,
            f'min_tracking_confidence':0.3
        }

        self.default_options = LandmarkContainer.GeneralLandmarkerDefaultOptions\
                                | other_options\
                                | additional_options

    """
    General container for running the mediapipe landmarking models
    """
    def __init__(self, 
                 model:int,
                 options:None|dict=None,
                 renderer:int=EasyDrawer.PYGAME, 
                 max_data_age_ms:int = 8000):
        '''
        Initialiser for landmark container

        Params:
         - model(int): the enumeration index of the model accessed by ModelIndices
         - options(None|dict): options for the mediapipe landmarking model
         - renderer(int): the enumeration index of the renderer accessed by EasyDrawer
         - max_data_age_ms(int): the maximum age of landmark_list in data_storage

        Returns:
         None, creates the container
        '''
        self.__construct_model(model)

        if options is None:
            options = self.default_options
        else:
            new_options = self.default_options
            for k,v in options.items():
                if k not in self.default_options:
                    continue
                new_options[k] = v
            options = new_options
        
        options = self.model_options(
            **options,
            base_options=BaseOptions(model_asset_path=self.model_path),
            result_callback=self.__do_callback
        )
        self.detector = self.model_main.create_from_options(options)

        self.renderer = EasyDrawer(renderer)
        self.max_data_age_ms = max_data_age_ms

        self.VD_screen = (0.5,0,0)
        self.VU_screen = (0.5,1,0)
        self.HR_screen = (1,0.5,0)
        self.HL_screen = (0,0.5,0)
        
        self.data_storage = []
        self.landmark_list = []
        self.landmark_indices = []
        self.timestamp_storage = []

    def __do_callback(self, result, output_image, timestamp_ms):
        '''
        Internal function for the detection callback function
        '''   
        # self.data_storage.append(result)
        self.timestamp_storage.append(timestamp_ms)

        pose_attributes = list(zip(*[getattr(result, attribute_) 
                                    for attribute_ 
                                    in self.model_attributes]))

        self.default_landmark_list = []

        try:
            # ADD CUSTOM_LANDMARKS TO LANDMARK_LIST
            for attributes in pose_attributes:
                if self.model_index == ModelIndices.HAND_MODEL:
                    handedness, screen_mark_list, world_mark_list = attributes
                    display_name = handedness[0].display_name # Get side
                else:  # POSE_MODEL
                    screen_mark_list, world_mark_list = attributes
                    display_name = '' # no sides
                
                self.default_landmark_list += [[
                    CustomLandmark(
                        screen=screen_mark,
                        world=world_mark,
                        name=self.landmark_names[idx],
                        idx=idx,
                        side=display_name
                    )
                    for idx, (screen_mark, world_mark) in enumerate(zip(screen_mark_list, world_mark_list))
                ]]

            world_x, world_y, world_z = zip(*[landmark.world for landmark in sum(self.landmark_list,[])])
            max_world = max(world_x), max(world_y), max(world_z)
            min_world = min(world_x), min(world_y), min(world_z)
            self.VU_world = (max_world[0]/2 + min_world[0]/2, max_world[1], max_world[2]/2 + min_world[2]/2)
            self.VD_world = (max_world[0]/2 + min_world[0]/2, min_world[1], max_world[2]/2 + min_world[2]/2)
            self.HR_world = (max_world[0], max_world[1]/2 + min_world[1]/2 , max_world[2]/2 + min_world[2]/2)
            self.HL_world = (max_world[0], max_world[1]/2 + min_world[1]/2 , max_world[2]/2 + min_world[2]/2)

        except Exception as e:
            # print(e) 
            pass

        self.landmark_list = deepcopy(self.default_landmark_list)
        try:
            self.default_landmark_indices = [landmark.idx for landmark in self.landmark_list[0]]
        except IndexError:
            self.default_landmark_indices = []

        self.landmark_indices = deepcopy(self.default_landmark_indices)
        self.data_storage.append(self.landmark_list)

        if self.timestamp_storage[0] <= timestamp_ms-self.max_data_age_ms:
            self.timestamp_storage.pop(0)
            self.data_storage.pop(0)

    def close(self):
        '''
        Wrapper for closing mediapipe
        '''
        self.detector.close()

    def detect_async(self, cv_image:np.ndarray, timestamp_ms:int):
        '''
        Wrapper for mediapipe's detect_async function

        Params:
         - cv_image(np.ndarray): Image data from opencv in RGB color space
         - timestamp_ms(int): Current timestamp
        
        Returns:
         None, calls detect_async and stores processed information in callback storage
         callback storage is {instance}.data_storage
        '''
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_image)
        self.detector.detect_async(mp_image, timestamp_ms)

    def set_display(self, image, flip:bool=False):
        '''
        Sets screen information for screen marker position display

        Params:
         - image(any)
        '''
        self.renderer.set_image(image, flip)
        self.image_info = self.renderer.image_info

    def calibrate(self):
        '''
        [NOT IMPLEMENTED]
        Calibrates the length of connections
        '''
        raise NotImplementedError('Calibration is not implemenetd')
        self.local_vectors = None
        self.calibrated_length = None

    def relative_displace(self, index:int, space:str='world') -> None:
        '''
        Sets a landmark as the origin for the rest of the landmarks
        recalculates the position of all other landmarks in the container
        relative to the new origin

        Params:
         - index(int): the index of the new origin landmark
         - space(str): the space in which to recalculate, defaults to world
        
        Returns:
         Non, updates the thingy
        '''
        try:
            self.landmark_indices.index(index)
        except ValueError:
            return
        
        landmark_indices = self.landmark_indices
        for group in self.landmark_list:
            
            new_origin = getattr(group[landmark_indices.index(index)], space)
    
            for landmark in group:
                position = getattr(landmark, space)
                setattr(landmark, space, [element-origin 
                                            for element, origin 
                                            in zip(position, new_origin)])
            
    def center_of_mass(self):
        for group in self.landmark_list:
            self.com = sum([])

    def __new_landmark_parse(self, 
                             instruction:str, 
                             group_idx:int,
                             insert_index:int) -> CustomLandmark:
        '''
        Parses landmark stuff
        '''
        attributes = ['screen','world']
        attributes_dict = {}
        group = self.default_landmark_list[group_idx]

        reference_indices, factors, name = instruction.split(' ', maxsplit=2)
        reference_indices = [int(index_) for index_ in reference_indices.split(',')]
        factors = [float(factor) for factor in factors.split(',')]
        
        if len(factors) != len(reference_indices)-1:
            raise Exception("Number of factors unmatchable to indices")
        
        if sum(factors) >= 1:
            raise Exception("Factors don't add up to 1")
        
        factors += [1-sum(factors)]
        
        for attribute in attributes:
            reference_landmarks = [getattr(group[index], attribute) for index in reference_indices]
            attributes_dict[attribute] = [sum([element*factor for element, factor in zip(elements, factors)]) 
                                          for elements in zip(*reference_landmarks)]
            
        side = group[0].side

        return CustomLandmark(**attributes_dict, name=name, side=side, idx=insert_index)

    def reorder_landmarks(self, landmark_dict:dict):
        '''
        Inserts new landmarks at certain positions

        {13:'11,12 0.5 mid shoulder'} -> adds a new landmark at index 13 which is halfway between index 11 and 12 called mid shoulder
        {14:'11,12,28,29 0.2,0.2,0.3 navel'} -> adds a new landmark at index 14 whic is closer to 28 and 29 than 11 and 12 called navel

        params:
         - landmark_dict(dict): A dictionary

        returns:
         updates the landmark_list
        '''
        self.landmark_connections = deepcopy(self.default_landmark_connections)

        try:
            self.landmark_indices = deepcopy(self.default_landmark_indices)
        except AttributeError:
            return
        
        for insert_index, instruction in landmark_dict.items():
            for group_idx, group in enumerate(self.landmark_list):
                group.append( 
                             self.__new_landmark_parse(instruction, group_idx, insert_index))
                
                for landmark in group[:-1]:
                    if landmark.idx>=insert_index:
                        landmark.idx += 1

            self.landmark_indices += [insert_index]
            for index_idx, index_ in enumerate(self.landmark_indices[:-1]):
                if index_>=insert_index:
                    self.landmark_indices[index_idx] += 1

            


            for connection in self.landmark_connections:
                for endpoint_idx, endpoint in enumerate(connection):
                    if endpoint>=insert_index:
                        connection[endpoint_idx] = endpoint+1
        

    def localise_vectors(self, 
                         x_bas:tuple|list, 
                         y_bas:tuple|list):
        '''
        Localises position vectors of landmarks according to given basis vectors

        the new basis vectors 
        '''
        make_vector, normalise_vector = toolbox.make_vector, toolbox.normalise_vector2
        cross_product = toolbox.cross_product
        landmark_indices = self.landmark_indices

        try:
            assert len(self.landmark_list)
        except:
            return
        
        self.new_basis = []

        for group in self.landmark_list:
            new_basis = []
            for input_ in (x_bas, y_bas):
                if input_[0] == 'index':
                    new_basis += [normalise_vector(
                        make_vector(*[group[landmark_indices.index(p)].world 
                                    for p 
                                    in input_[1:]]))]
                    
                if input_[0] == 'vector':
                    new_basis += [normalise_vector(input_[1:])]

            new_basis += [cross_product(*new_basis)]
            self.new_basis += [new_basis]

            for landmark in group:
                new_vec = np.linalg.solve(new_basis, landmark.world)
                landmark.local_coor = list(new_vec)

    def flip_axes(self, axes):
        '''
        Flips all coordinate along specified axes

        Params:
        - axes(str): Axis is x,y,z or 0,1,2

        Returns:
         Flips axeses
        '''
        for landmark_group in self.landmark_list:
            for landmark in landmark_group:
                landmark.flip_axes(axes)

    def __draw_landmark(self, 
                   position:tuple|list, 
                   radius:int = 5, 
                   color:tuple|list = FontColorWhite, 
                   scale:float = 1.0, 
                   color2:tuple|list = None):
        '''
        Internal function to draw a landmark marker at a certain position

        Params:
         - position(tuple|list): Coordinate to draw landmark at
         - radius(int): radius of landmark, defaults to 5
         - color(tuple|list): BGR Color, defaults to style.FontColorWhite
         - scale(float): scaling factor of landmark, defaults to 1.0
         - color2(tuple|list): BGR Color, if left None will become the same as color, defaults to None

        Returns:
         None, draws landmark

        '''
        position = [int(pos) for pos in position]
            
        self.renderer.render_landmark(position, color1=color, radius=radius, scale=scale, color2=color2)

    def __draw_landmark_attribute(self, 
                  landmark:CustomLandmark, 
                  position:tuple|list, 
                  landmark_attributes:str,
                  landmark_index:int, 
                  scale:float = 1):
        '''
        Draws landmark attributes as text
        
        Params:
         - landmark(custom_landmark): A custom landmark object containing inofmration
         - position(tuple|list): Position to render the information to (center)
         - landmark_attributes(str): Attributes to draw separated by space
         - scale(float): How big the texts are, defaults to 0.5
        
        Returns:
         None, draws stuff
        '''
        landmark_attributes = landmark_attributes.split()

        # Grouping of landmarks are set here
        relative_positions = [
            (position[0],position[1]-90*scale),
            (position[0],position[1]-60*scale),
            (position[0]+70*scale,position[1]-60*scale),
            (position[0]+140*scale,position[1]-60*scale)
        ]

        for_printing = []
        
        if 'index' in landmark_attributes:
            for_printing += [(f'{landmark.idx}', relative_positions[0],
                              (35*scale,0), 0.8, FontColorBlack, 1)]
            
        if 'name' in landmark_attributes:
            for_printing += [(f'{landmark.side} {landmark.name}', relative_positions[0],
                              (35*scale,0), 0.8, FontColorBlack, 1)]
                         
        if 'screen_coor' in landmark_attributes:
            for coor in landmark.screen:
                for_printing += [(f'{coor:0.2f}', relative_positions[1],
                              (0,20*scale), 0.8, FontColorOrange, 1)]
                
        if 'world_coor' in landmark_attributes:
            for coor in landmark.world:
                for_printing += [(f'{coor:0.2f}', relative_positions[2],
                              (0,20*scale), 0.8, FontColorCyan, 1)]
        try:
            if 'local_coor' in landmark_attributes:
                for coor in landmark.local_coor:
                    for_printing += [(f'{coor:0.2f}', relative_positions[3],
                                (0,20*scale), 0.8, FontColorWhite, 1)]
        except AttributeError:
            pass

        if 'visibility' in landmark_attributes:
            for_printing += [(landmark.presence, relative_positions[3],
                              (0,20*scale), 0.8, FontColorWhite, 1)]

        for message, pos, displacer, f_scale, color, thickness in for_printing:
            self.renderer.render_text(message, pos, displacer,
                         color, f_scale*scale)
        
    def draw(self, 
             current_timestamp: int,
             information = None, 
             indices = True, 
             attributes = 'index name screen_coor world_coor local_coor',
             connector = 'line',
             flipped:bool=True):
        """
        Draws information on the screen set by set_display

        drawables include:
         - index: landmark index
         - name: landmark name
         - screen_coor: screen coordinate of landmark
         - world_coor: world coordinate of landmark
         - local_coor: local coordinate of landmark

        connectors include:
         - line: a simple line
         - bone: a kite-like shape akin to a Bone object in 3D modelling
        
        Params:
         - current_timestamp(int): The current time in ms, used to display real vs processed
         - information(None): Unused
         - indices(iterable|bool): Used to select landmarks to show [UNINMPLEMENTED!]
         - attributes(str): drawables separated by space
         - connector(str): connector between landmarks
        
        Returns:
         None
        """
        drawn = [] # Keeps track of rendered info

        # DEBUGGING STUFF
        self.renderer.render_text(f'{len(self.timestamp_storage)/(self.max_data_age_ms/1000):0.1f} fps', 
                     position = (10,60),
                     scale = 2,
                     color = FontColorYellow)
        
        self.renderer.render_text(f'{len(self.timestamp_storage)} frames in storage',
                     position = (10,60),
                     displacer = (0,40),
                     color = FontColorYellow)
        
        self.renderer.render_text(f'real time {current_timestamp} ms',
                     position = (10,60),
                     displacer = (0,40),
                     color = FontColorYellow)
        
        try:
            self.renderer.render_text(f'latest processed {self.timestamp_storage[-1]} ms',
                        position = (10,60),
                        displacer = (0,40),
                        color = FontColorYellow)
        except IndexError:
            pass

        try:
            for basis in self.hand_basis:
                self.renderer.render_text(f'[{','.join([f'{el:0.2f}' for el in basis])}] basis',
                             position = (10,60),
                             displacer = (0,40),
                             color = FontColorYellow)
        except:
            pass
        
        for group_id, landmark_group in enumerate(self.landmark_list):

            positions = [None for i in range(len(landmark_group))]

            for landmark in landmark_group:
                index_ = self.landmark_indices.index(landmark.idx)
                drawn_index = index_ + group_id*len(landmark_group)
                landmark_screen_position = [int(pos*dim) 
                                            for pos, dim 
                                            in zip(landmark.screen, self.image_info)]
                self.__draw_landmark(landmark_screen_position)
                self.__draw_landmark_attribute(landmark, landmark_screen_position, attributes, index_)
                positions[index_] = landmark_screen_position
                drawn.append(drawn_index)

            try:
                for endpoint1, endpoint2 in self.landmark_connections:
                    getattr(self.renderer, f'render_{connector}')(positions[self.landmark_indices.index(endpoint1)],
                                                                  positions[self.landmark_indices.index(endpoint2)])

            except Exception as e:
                self.renderer.render_text(e, self.renderer.image_center_left)

        if flipped:
            self.renderer.flip_render()

        return self.renderer.image
        

    def __sanitise_arguments(self, instruction:str, arguments:tuple|list, space:str):
        def determine_factor(primitive:str):
            if 'V' in primitive:
                return (0,1,0)
            if 'H' in primitive:
                return (1,0,0)
            raise ValueError("No primitive found!")
        
        def construct_new_arguments(anchor_point, arguments):
            return [toolbox.mask_factor(anchor_point, 
                                        getattr(self,f'{arg}_{space}'), 
                                        determine_factor(arg))
                    if isinstance(arg, str)
                    else arg
                    for arg
                    in arguments]
        
        if all([isinstance(arg, tuple) for arg in arguments]): # No primitive vector
            return arguments
        
        if all([isinstance(arg,str) for arg in arguments]): # All primitive vector
            return [getattr(self, f'{arg}_{space}') for arg in arguments]
        
        if instruction in ('distance', 'displacement'):
            anchor_point = arguments[0] if isinstance(arguments[0], (tuple, list)) else arguments[1]
            return construct_new_arguments(anchor_point, arguments)

        if instruction == 'angle_point':
            anchor_is_primitive = isinstance(arguments[1], str)
            if anchor_is_primitive:
                anchor_point = toolbox.mask_factor(toolbox.middle_point(arguments[0], arguments[2]),
                                                   getattr(self, f'{arguments[1]}_{space}'),
                                                   determine_factor(arguments[1]))
                new_arguments = [arguments[0],anchor_point,arguments[2]]
                return new_arguments
            anchor_point = arguments[1]
            return construct_new_arguments(anchor_point, arguments)

        return [getattr(self, f'{arg}_{space}') if isinstance(arg,str) else arg 
                for arg
                in arguments]
    
    def __parse_measure(self, instruction):
        axis_dict = dict(('x0','y1','z2','w3'))
        sanitise_arguments = self.__sanitise_arguments
        function, space, axis, group_idx, indices, additional = instruction.split(' ',5)

        #Find measure instruction
        try:
            use_function = getattr(toolbox, function)
        except AttributeError:
            raise AttributeError(f'{function} is not a measure method')

        arguments = []
        for idx in indices.split(','):
            try:
                idx = self.landmark_indices.index(int(idx))
                landmark_group = int(group_idx)
                arguments += [getattr(self.landmark_list[landmark_group][idx],space)]
                continue
            except ValueError:
                pass
            except IndexError:
                return []
            arguments += [idx.upper()]
                
        arguments = sanitise_arguments(instruction, arguments, space)
        return use_function(*arguments)
            
    def measure(self,
                *inputs:str):
        '''
        Measures properties using toolboxes

        Accepted instructions:
         - angle_point
         - angle_vector
         - displacement
         - distance
        
        Look at toolbox documentation for usage info

        Params:
         - inputs(any): Accepts any arguments

        Returns:
         None, measured properties are accesible from 'measured' attribute a list
        '''
            
        measured = []

        for group in self.landmark_list:
            tmp = []
            for in_ in inputs:
                tmp += [self.__parse_measure(in_)]
            measured += [tmp]

        self.measured:list = measured
                
if __name__=='__main__':
    pass
        