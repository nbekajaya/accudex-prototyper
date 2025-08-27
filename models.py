import numpy as np
import mediapipe as mp

from copy import deepcopy
from drawer import EasyDrawer
from models_utils import LandmarkNames, LandmarkConnections, ModelIndices
from toolbox import Toolbox as toolbox
from style import *

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
                 idx:int=None,
                 name:str=None,
                 screen:list[float]=None, 
                 world:list[float]=None,
                 additional:str=None,
                 instruction:str=None):
        
        self.idx = idx
        self.name = name
        self.additional = additional

        # ADDED LANDMARKS WILL ONLY CONTAIN INSTRUCTIONS
        if instruction is not None:
            self.instruction = instruction
            return
        
        if all([element is None 
                for element in (screen, world, additional)]):
            return

        try: self.from_NormalizedLandmark(screen, world)
        except AttributeError: self.from_List(screen, world)

    def from_NormalizedLandmark(self, 
                                landmark_screen,
                                landmark_world):
        self.screen = [getattr(landmark_screen, el) for el in 'xyz']
        self.world = [getattr(landmark_world, el) for el in 'xyz']

    def from_List(self,
                  landmark_screen:list[float],
                  landmark_world:list[float]):
        self.screen = landmark_screen
        self.world = landmark_world

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
        landmark_identity = ['-'.join([
            value for identity in ('idx','name','additional')
            if (value:=f"{getattr(self,identity,'')}") != ''])
        ]

        landmark_position = [
            f'{info_key}: {getattr(self, info_key, None)}' 
            for info_key in ('screen','world')
        ]

        landmark_instruction = [
            f'instructions: {getattr(self,'instruction',None)}'
        ]

        landmark_information = ' | '.join(
            landmark_identity+landmark_position+landmark_instruction
        )

        return landmark_information

class LandmarkContainer:
    model_params = (
        {'model_name':'hand',
         'model_main':HandLandmarker,
         'model_options':HandLandmarkerOptions,
         'model_result':HandLandmarkerResult,
         'model_path':HandModelPath,
         'model_attributes':('handedness', 
                             'hand_landmarks',
                             'hand_world_landmarks')},
        {'model_name':'pose',
         'model_main':PoseLandmarker,
         'model_options':PoseLandmarkerOptions,
         'model_result':PoseLandmarkerResult,
         'model_path':PoseModelPath,
         'model_attributes':('pose_landmarks',
                             'pose_world_landmarks')},
    )

    GeneralLandmarkerDefaultOptions = {
        'running_mode':VisionRunningMode.LIVE_STREAM,
    }   

    # LANDMARK_LIST TRANSFORMATION
    def __construct_model(self, model_flag:int):
        model_params = LandmarkContainer.model_params

        if model_flag>len(model_params):
            raise IndexError("Model enum not recognised!")
        
        self.model_flag = model_flag
        model_dict = model_params[model_flag]

        self.landmark_names : list[str]
        self.model_result_amt : int
        self.default_landmark_connections : list[list[int,int]]

        self.landmark_names = getattr(
            LandmarkNames,
            f"{model_dict['model_name'].upper()}_LANDMARK_NAMES"
        )
        self.model_result_amt = len(self.landmark_names)

        self.default_landmark_connections = getattr(
            LandmarkConnections,
            f"{model_dict['model_name'].upper()}_LANDMARK_CONNECTIONS"
        )
        
        self.landmark_connections = deepcopy(self.default_landmark_connections)
        
        for k,v in model_dict.items():
            setattr(self, k, v)

        additional_options = model_flag and {'output_segmentation_masks':True} or {}

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

        # SCREEN EXTREMES
        self.VD_screen = (0.5,1,0)
        self.VU_screen = (0.5,0,0)
        self.HR_screen = (1,0.5,0)
        self.HL_screen = (0,0.5,0)
        
        self.data_storage:list[list[list[CustomLandmark]]] = []
        self.timestamp_storage:list[int] = []
        self.set_landmarks()

    # LANDMARK ORDER FUNCTIONS
    def set_landmarks(self, landmarks_information:list[str] = []) -> None:
        """Sets the landmark list for the LandmarkContainer instance

        Args:
            landmarks_information (list[str], optional): List of landmarks to add. 
                Defaults to [], which is default mediapipe landmarks.
        """
        landmark_list_template : list[CustomLandmark]
        default_amount : int

        default_amount = self.model_result_amt

        self.default_landmark_idx_map = list(range(default_amount))
        landmark_idx_map = deepcopy(self.default_landmark_idx_map)

        landmark_list_template = [None for _ in range(default_amount)]

        for instruction_number, instruction in enumerate(landmarks_information,
                                                         start=default_amount):
            instruction_split = instruction.split(maxsplit=4)

            insert_idx = int(instruction_split[0])
            ref_group_key = instruction_split[1]
            ref_indices = instruction_split[2]
            ref_weights = instruction_split[3]
            name = instruction_split[4]
            
            landmark_idx_map.insert(insert_idx,
                                    instruction_number)
            
            landmark_list_template.append(
                CustomLandmark(
                    instruction={
                        'group_key':ref_group_key,
                        'indices':ref_indices,
                        'weights':ref_weights},
                    idx=insert_idx,
                    name=name)
                )
        
        # GIVES idx ATTRIBUTES
        for idx, landmark in enumerate(landmark_list_template):
            if landmark is not None:
                continue
            landmark_index = landmark_idx_map.index(idx)
            landmark_name = self.landmark_names[idx]
            landmark_list_template[idx] = CustomLandmark(idx = landmark_index,
                                                         name = landmark_name)
            
        # UPDATE LANDMARK IDX MAP
        landmark_idx_map = [
            getattr(landmark,'idx') for landmark in landmark_list_template
        ]

        self.landmark_names = [
            getattr(landmark,'name') for landmark in landmark_list_template
        ]
        print(f'set_landmarks: landmark names updated!\n\t{self.landmark_names}')

        self.landmark_idx_map = landmark_idx_map
        self.landmark_list_template = deepcopy(landmark_list_template)

        print(f'set_landmarks: {self.model_name} landmark_list_template configured:')
        print("\n\t".join(
                [landmark.__repr__() 
                for landmark 
                in self.landmark_list_template]
            )
        )
        
    def __landmark_instruction_parse(self, 
                                     landmark:CustomLandmark,
                                     landmark_group_idx:int) -> CustomLandmark:
        """Parses landmark stuff
        """
        attributes = ('screen','world')
        
        attributes_dict = {}
        group_dict = {
            'ORIGINAL':self.default_landmark_idx_map,
            'LIVE':self.landmark_idx_map
        }

        ref_group : list[CustomLandmark]
        instructions : dict
        
        instructions = getattr(landmark, 'instruction')

        ref_group_key, ref_indices, ref_weights = [
            instructions.get(key) for key in ('group_key', 'indices','weights')
        ]

        ref_map : list[int] = group_dict.get(ref_group_key, 
                                   self.default_landmark_idx_map)
        
        ref_indices = [
            ref_map.index(int(index_)) for index_ in ref_indices.split(',')
        ]

        ref_group  = self.landmark_list[landmark_group_idx]

        ref_weights = [float(factor) for factor in ref_weights.split(',')]

        if len(ref_weights) == 1: # If only one factor, apply to all
            ref_weights = [ref_weights[0]] * len(ref_indices)
        
        if len(ref_weights) != len(ref_indices):
            raise Exception("Number of ref_weights unmatchable to indices")
        
        if sum(ref_weights) > 1:
            raise Exception("Factor sum is over 1")
        
        for attribute in attributes:
            ref_landmarks = [
                getattr(ref_group[index], attribute) for index in ref_indices
            ]
            attributes_dict[attribute] = toolbox.coordinate_weighter(
                ref_landmarks, ref_weights
            )
            
        landmark.additional = ref_group[0].additional

        for k, v in attributes_dict.items():
            setattr(landmark,k,v)

        return landmark
    
    def __update_landmark_list(self):
        # PLACES THE DEFAULT LANDMARKS IN THE CORRECT INDICES
        for group_idx, (ref_group, tmpl_group) in enumerate(
                zip(self.default_landmark_list, self.landmark_list)
            ):
            for landmark_ref in ref_group:
                try:
                    for key in ('screen', 'world', 'additional'):
                        setattr(
                            tmpl_group[landmark_ref.idx],
                            key,
                            getattr(landmark_ref,key)
                        )
                except Exception as e:
                    raise Exception(
                        f'Failed to set landmark_info for {landmark_ref.idx} in {group_idx}-th group', e
                    )
        # print(f'\t{self.model_name} landmark_list FILLED')

        # ADDS ADDITIONAL LANDMARK IF ANY
        for group_idx, group in enumerate(self.landmark_list):
            for landmark_idx, landmark in enumerate(group):
                if not hasattr(landmark, 'instruction'):
                    continue

                try:
                    group[landmark_idx] = self.__landmark_instruction_parse(
                         landmark,
                         landmark_group_idx=group_idx
                    )
                except Exception as e:
                    print(f'Error adding additional landmark: {e}')
                    print(f'\tlandmark_idx:{landmark_idx}\n\tlandmark:{landmark}')

        print(f'\t{self.model_name} landmark_list MODIFIED')

    # MAIN LANDMARK FUNCTIONALITIES FUNCTIONS
    def __do_callback(self, result, output_image, timestamp_ms:int):
        '''
        Internal function for the detection callback function
        '''   
        landmark_names : list[str]
        self.default_landmark_list : list[list[CustomLandmark]]
        self.landmark_list : list[list[CustomLandmark]]

        self.current_processed_timestamp = timestamp_ms
        self.timestamp_storage.append(timestamp_ms)
        # print(f'----------- {self.model_name} CALLBACK at {timestamp_ms} ms -----------')
        
        pose_attributes = list(zip(*[getattr(result, attribute_) 
                                    for attribute_ 
                                    in self.model_attributes]))
        
        landmark_names = self.landmark_names
        self.default_landmark_list = []

        # ADD CUSTOM_LANDMARKS TO LANDMARK_LIST
        try:
            for attributes in pose_attributes:
                if self.model_flag == ModelIndices.HAND_MODEL:
                    handedness, screen_mark_list, world_mark_list = attributes
                    display_name:str = handedness[0].display_name # Get side
                else:  # POSE_MODEL
                    screen_mark_list, world_mark_list = attributes
                    display_name = '' # no sides
                
                landmark_group = [
                    CustomLandmark(
                        screen=screen_mark,
                        world=world_mark,
                        name=landmark_names[idx],
                        additional=display_name,
                        idx=idx
                    )
                    for idx, (screen_mark, world_mark) 
                    in enumerate(zip(screen_mark_list, world_mark_list))
                ]
                self.default_landmark_list.append(landmark_group)
        except Exception as e:
            pass
        # print(f'\t{self.model_name} default_landmark_list GENERATED')

        self.landmark_list = [
            deepcopy(self.landmark_list_template) 
            for _ in range(len(self.default_landmark_list))
        ]

        # UPDATING LANDMARK LIST TO CONTAIN ADDITIONAL POINTS
        try:
            self.__update_landmark_list()
        except Exception as e:
            # print(f'\t !!!! LANDMARK LIST UPDATE FAILED !!!!\n\t    {e}')
            pass
        
        # LANDMARK TRANSFORMATION FUNCTIONS
        try:
            self.__flip_axes(*self.flip_axes_values)
        except AttributeError:
            pass

        try:
            self.__relative_displace(**self.relative_displace_values)
        except AttributeError:
            pass

        try:
            self.__localise_vectors(**self.local_vectors_values)
        except AttributeError:
            pass
        
        # GETS EXTREMES FOR MEASUREMENT
        try:
            self.__get_world_extremes()
        except Exception as e:
            pass
        
        # DO ASSESSMENT PROTOCOLS MEASUREMENT
        try:
            assert hasattr(self,'assessment_protocols')
            self.measure(self.assessment_protocols)
        except AssertionError:
            pass
        except Exception as e:
            print(f"Measurement failed, {e}")
            pass
        
        # ADDS DATA TO DATA STORAGE
        self.data_storage.append(self.landmark_list)
        
        # CLEAR DATA_STORAGE
        if self.timestamp_storage[0] <= timestamp_ms-self.max_data_age_ms:
            self.timestamp_storage.pop(0)
            self.data_storage.pop(0)
        print(f'---------- {self.model_name} CALLBACK at {timestamp_ms} ms FINISHED ----------')

    def close(self):
        '''Wrapper for closing mediapipe'''
        self.detector.close()

    def detect_async(self, cv_image:np.ndarray, timestamp_ms:int):
        '''Wrapper for mediapipe's detect_async function

        Params:
         - cv_image(np.ndarray): Image data from opencv in RGB color space
         - timestamp_ms(int): Current timestamp
        
        Returns:
         None, calls detect_async and stores processed information in callback storage
         callback storage is {instance}.data_storage
        '''
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_image)
        self.detector.detect_async(mp_image, timestamp_ms)

    # STORAGE RECORDING
    data_recording: dict[int,tuple[dict[str,list], ...]]

    def record_data(self):
        """Record data saves the information of all points and
        measurements in a frame in the data_recording attribute

        data_recording structure:
            A dictionary where the key is the frame timestamp,
            each value is a list containing sub-dictionaries
            sub dictionaries has key: landmark, measure
        """
        recording: dict[int,tuple[dict[str,list], ...]] = getattr(
            self, 'data_recording', dict()
        )
        measurements = getattr(
            self, 'measured', [[]*len(self.landmark_list)]
        )
        bundle = []

        for landmark_group, measure_group in zip(self.landmark_list, measurements):
            sub_bundle = {
                'landmark': landmark_group,
                'measure': measure_group
            }
            bundle.append(sub_bundle)
        recording[self.current_processed_timestamp] = bundle
        setattr(self, 'data_recording', recording)

    def record_assessment(self):
        if not hasattr(self, 'assessment_condition'):
            return
        self.record_data()
        return
    
    # ASSESSMENT FUNCTIONS
    def set_assessment(self, assessment_dict:dict):
        """Sets the assessment information for the model

        Args:
            assessment_dict (dict): A dictionary with three keys
                - protocol_name: the name of the protocol (str)
                - protocol_time_seconds: the duration of the protocol (int)
                - protocol_measurements: a list of JSON-like dictionaries
                    pertaining to the measured properties (list[dict])
        """
        self.assessment_name:str = assessment_dict['protocol_name']
        self.assessment_time:int = assessment_dict['protocol_time_seconds']
        self.assessment_condition:any = assessment_dict['protocol_state_condition']
        self.assessment_protocols:list[dict] = assessment_dict['protocol_measurements']

    def create_assessment_summary(self):
        """_summary_
        """
        return
        
    # CALIBRATION FUNCTIONS
    def calibrate(self, timestamp):
        '''
        Saves the position of all landmarks and connections
        '''
        self.calibrated_groups = deepcopy(self.landmark_list)
        self.calibration_time = timestamp

    def multi_calibrate(self, timestamp, max_calibrations=4):
        multi_exists = hasattr(self, 'multi_calibrated_groups')

        if not multi_exists:
            self. multi_calibrated_groups = []

        multi_maxed_out = len(self.multi_calibrated_groups) == max_calibrations

        if multi_maxed_out:
            self.multi_calibrated_groups = []
        
        self.multi_calibrated_groups += [deepcopy(self.landmark_list)]
        self.calibration_time = timestamp

    # LANDMARK TRANSFORMATION FUNCTIONS
    def relative_displace(self, index:int, space:str='world'):
        '''Sets a landmark as the origin for the rest of the landmarks
        recalculates the position of all other landmarks in the container
        relative to the new origin

        Args:
            index (int): the index of the new origin landmark.
            space (str): the space in which to recalculate, defaults to world.
        '''
        self.relative_displace_values = {'index':index,'space':space}

    def flip_axes(self, axes:str|int):
        '''Flips all coordinate along specified axes

        Args:
            axes (str): Axis is x,y,z or 0,1,2
        '''
        self.flip_axes_values = {'axes':axes}
    
    def localise_vectors(self, 
                         x_bas: list[str,int,int]|list[str,int,int,int], 
                         y_bas: list[str,int,int]|list[str,int,int,int]):
        """Adds a new local vector based on the given basis

        Args:
            x_bas (list): If first index is 'index' the next two values are landmark indices.
                If first index is 'index' the next two values are landmark indices.
            y_bas (list): If first index is 'index' the next two values are landmark indices.
                If first index is 'index' the next two values are landmark indices.
        """
        self.localise_vectors_vals = {'x_bas':x_bas, 'y_bas':y_bas}
        
    def __flip_axes(self, axes:int|str):
        """Internal function for flip_axes
        """
        for landmark_group in self.landmark_list:
            for landmark in landmark_group:
                landmark.flip_axes(axes)

    def __relative_displace(self, index:int, space:str='world') -> None:
        """Internal function for relative_displace
        """
        try:
            self.landmark_idx_map.index(index)
        except ValueError:
            return
        
        landmark_idx_map = self.landmark_idx_map
        for group in self.landmark_list:
            new_origin = getattr(group[landmark_idx_map.index(index)], space)
    
            for landmark in group:
                position = getattr(landmark, space)
                setattr(
                    landmark, 
                    space, 
                    [
                     element-origin 
                     for element, origin 
                     in zip(position, new_origin)
                    ]
                )

    def __localise_vectors(self, 
                         x_bas: list[str,int,int]|list[str,int,int,int], 
                         y_bas: list[str,int,int]|list[str,int,int,int]):
        '''Internal function for localise_vectors
        '''
        make_vector = toolbox.make_vector, 
        normalise_vector = toolbox.normalise_vector2
        cross_product = toolbox.cross_product
        landmark_idx_map = self.landmark_idx_map

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
                        make_vector(*[group[landmark_idx_map.index(p)].world 
                                    for p 
                                    in input_[1:]]))]
                    
                if input_[0] == 'vector':
                    new_basis += [normalise_vector(input_[1:])]

            new_basis += [cross_product(*new_basis)]
            self.new_basis += [new_basis]

            for landmark in group:
                new_vec = np.linalg.solve(new_basis, landmark.world)
                landmark.local_coor = list(new_vec)

    # UI DRAWING
    def set_display(self, image, flip:bool=False):
        '''
        Sets screen information for screen marker position display

        Params:
         - image(any)
        '''
        self.renderer.set_image(image, flip)
        self.image_info = self.renderer.image_info

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
         - color(tuple|list): RGB Color, defaults to style.FontColorWhite
         - scale(float): scaling factor of landmark, defaults to 1.0
         - color2(tuple|list): RGB Color, 
                                if left None will become the same as color. 
                                Defaults to None.

        Returns:
         None, draws landmark

        '''
        position = [int(pos) for pos in position]
            
        self.renderer.render_landmark(position, 
                                      color1=color, 
                                      radius=radius, 
                                      scale=scale, 
                                      color2=color2)

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
            for_printing += [
                (f'{landmark.idx}', 
                 relative_positions[0],
                 (35*scale,0), 0.5, FontColorBlack, 1)
            ]
            
        if 'name' in landmark_attributes:
            for_printing += [
                (f'{landmark.additional} {landmark.name}', 
                 relative_positions[0],
                 (35*scale,0), 0.5, FontColorBlack, 1)
            ]
                         
        if 'screen_coor' in landmark_attributes:
            for coor in landmark.screen:
                for_printing += [
                    (f'{coor:0.2f}', 
                     relative_positions[1],
                     (0,20*scale), 0.5, FontColorOrange, 1)
                ]
                
        if 'world_coor' in landmark_attributes:
            for coor in landmark.world:
                for_printing += [
                    (f'{coor:0.2f}', 
                     relative_positions[2],
                     (0,20*scale), 0.5, FontColorCyan, 1)
                ]
        try:
            if 'local_coor' in landmark_attributes:
                for coor in landmark.local_coor:
                    for_printing += [
                        (f'{coor:0.2f}', 
                         relative_positions[3],
                         (0,20*scale), 0.5, FontColorWhite, 1)
                    ]
        except AttributeError:
            pass

        if 'visibility' in landmark_attributes:
            for_printing += [
                (landmark.presence, 
                 relative_positions[3],
                 (0,20*scale), 0.8, FontColorWhite, 1)
            ]

        for message, pos, displacer, f_scale, color, thickness in for_printing:
            self.renderer.render_text(message, pos, displacer,
                         color, f_scale*scale)
    
    def __draw_measurement(self, 
                           measurement:dict,
                           position:tuple[int],
                           displace:tuple[int],
                           idx:int, 
                           color1:tuple, 
                           color2:tuple):
        render_text = self.renderer.render_text

        def colorise_bounds(value,params,tolerance = 0.1):
            def one_side_bound(bound, negate:bool):
                tolerance_ = -tolerance if negate else tolerance
                return toolbox.color_lerp(value, 
                                          bound+tolerance_, 
                                          bound, 
                                          color1,
                                          color2)
            
            if all([el not in params for el in ('lower_bound','upper_bound')]):
                return color2
            
            if 'lower_bound' not in params:
                bound = params['upper_bound']
                return one_side_bound(bound, False)

            if 'upper_bound' not in params:
                bound = params['lower_bound']
                return one_side_bound(bound, True)

            bounds = (params['lower_bound'], params['upper_bound'])
            if value > bounds[1]:
                bound = bounds[1]
                return one_side_bound(bound, False)
            
            bound = bounds[0]
            return one_side_bound(bound,True)
            
        params = measurement['params']
        try:
            if params['do_draw'] == False:
                return
        except KeyError:
            pass

        test_value = measurement['result']
        result_is_list = isinstance(measurement['result'], list)
        
        if result_is_list:
            test_value = [float(f'{el:0.2f}') for el in measurement['result']]

        local_position = [el_p + idx*el_d
                          for el_p, el_d
                          in zip(position, displace)]
        
        if result_is_list:
            try:
                test_value = test_value[params['check_index']]
            except KeyError:
                pass
        
        try:
            use_name = params['name'] 
        except KeyError:
            use_name = '-'.join([measurement['function_name']]\
                                +params['default_names'])
            
        if isinstance(test_value,str):
            render_text(test_value, local_position, color=color1)
            return
        
        measure_color = colorise_bounds(test_value, params)

        for element in [use_name, test_value]:
            render_text(element, local_position, color=measure_color)
        
        idx += 1
        
    def draw(self, 
             current_timestamp: int,
             draw_debug:bool=False, 
             draw_measurements:bool=True,
             measurement_color_range:tuple[tuple]=(FontColorRed, FontColorCyan), 
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

        debug_draw_position = (10,60)
        debug_displacer = (0,40)
        # DEBUGGING STUFF
        if draw_debug:
            self.renderer.render_text(f'{len(self.timestamp_storage)/(self.max_data_age_ms/1000):0.1f} fps', 
                        position = debug_draw_position,
                        scale = 2,
                        color = FontColorYellow)
            
            self.renderer.render_text(f'{len(self.timestamp_storage)} frames in storage',
                        position = debug_draw_position,
                        displacer = debug_displacer,
                        color = FontColorYellow)
            
            self.renderer.render_text(f'real time {current_timestamp} ms',
                        position = debug_draw_position,
                        displacer = debug_displacer,
                        color = FontColorYellow)
        
            try:
                self.renderer.render_text(f'latest processed {self.timestamp_storage[-1]} ms',
                            position = debug_draw_position,
                            displacer = debug_displacer,
                            color = FontColorYellow)
            except IndexError:
                pass

        if hasattr(self, 'calibration_time'):
            if (current_timestamp - 2500) < self.calibration_time:
                self.renderer.render_text(f'Calibrated at {self.calibration_time} ms',
                                          debug_draw_position,
                                          color = FontColorYellow,
                                          font_thickness=2)

        # Drawing measurement stuff
        if draw_measurements and hasattr(self,'measured'):
            for measurement_group in self.measured:
                drawn_measurement = 0
                for measurement in measurement_group:
                    try:
                        self.__draw_measurement(measurement,
                                                (10,100),
                                                (0,25),
                                                drawn_measurement,
                                                *measurement_color_range)
                    except Exception as e:
                        print(f'Failed rendering of measurement:{e} \n{measurement}')
        
        for group_id, landmark_group in enumerate(self.landmark_list):
            positions = [None for _ in range(len(landmark_group))]
            for landmark_idx, landmark in enumerate(landmark_group):
                index_ = landmark_idx
                drawn_index = index_+ group_id*len(landmark_group)
                landmark_screen_position = [int(pos*dim) 
                                            for pos, dim 
                                            in zip(landmark.screen, self.image_info)]
                
                # try:
                self.__draw_landmark(landmark_screen_position)
                self.__draw_landmark_attribute(landmark, landmark_screen_position, attributes, index_)
                positions[index_] = landmark_screen_position
                drawn.append(drawn_index)
                # except Exception as e:
                #     print(f'Failed rendering of landmark: {e}\n{landmark}\nin landmark_group:{group_id}')

            try:
                for endpoint1, endpoint2 in self.landmark_connections:
                    getattr(
                        self.renderer, 
                        f'render_{connector}')(positions[endpoint1],positions[endpoint2])
            except Exception as e:
                # self.renderer.render_text(e, self.renderer.image_center_left)
                pass

        if flipped:
            self.renderer.flip_render()

        return self.renderer.image
    
    # MEASUREMENT STUFF
    def __get_world_extremes(self):
        world_x, world_y, world_z = zip(*[
            landmark.world for landmark in sum(self.landmark_list,[])]
        )

        max_world = max(world_x), max(world_y), max(world_z)
        min_world = min(world_x), min(world_y), min(world_z)

        self.VU_world = (
            max_world[0]/2 + min_world[0]/2, max_world[1], max_world[2]/2 + min_world[2]/2
        )
        self.VD_world = (
            max_world[0]/2 + min_world[0]/2, min_world[1], max_world[2]/2 + min_world[2]/2
        )
        self.HR_world = (
            max_world[0], max_world[1]/2 + min_world[1]/2 , max_world[2]/2 + min_world[2]/2
        )
        self.HL_world = (
            max_world[0], max_world[1]/2 + min_world[1]/2 , max_world[2]/2 + min_world[2]/2
        )

    def __sanitise_arguments(self, 
                             instruction:str, 
                             arguments:tuple|list, 
                             space:str):
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
        
        arguments = [getattr(argument,space) if hasattr(argument,space) else argument
                     for argument
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
                anchor_point = toolbox.mask_factor(
                    toolbox.middle_point(arguments[0], arguments[2]),
                    getattr(self, f'{arguments[1]}_{space}'),
                    determine_factor(arguments[1])
                )
                new_arguments = [arguments[0],anchor_point,arguments[2]]
                return new_arguments
            anchor_point = arguments[1]
            return construct_new_arguments(anchor_point, arguments)

        return [getattr(self, f'{arg}_{space}') if isinstance(arg,str) else arg 
                for arg
                in arguments]
    
    def __parse_measure(self, group_idx:int, instruction:dict):
        sanitise_arguments = self.__sanitise_arguments

        # def get_landmark_from_index(landmark_group, indices):
        #     lookup = self.landmark_idx_map
        #     landmarks = []

        #     for idx in indices:
        #         if isinstance(idx, int):
        #             try:
        #                 landmarks += [landmark_group\
        #                               [lookup.index(idx)]]
        #                 continue
        #             except ValueError:
        #                 pass
        #         if isinstance(idx, str):
        #             landmarks += [idx]
        #             continue
        #         landmarks += [None]

        #     return landmarks

        function_name = instruction['function_name']
        indices = instruction['indices']
        # print(f'\n__parse_measure: basic measurement info got! {function_name}, {indices}')

        # Find measure function
        try:
            use_function = getattr(toolbox, function_name)
        except AttributeError:
            raise AttributeError(f'{function_name} is not a measure method')
        # print(f'__parse_measure: measurement tool got! {use_function}')

        if instruction['function_name'] == 'compare':
            try:
                measured_values = [self.measured[group_idx][idx]['result'] 
                                   for idx 
                                   in indices]
                instruction['params']['default_names'] = [str(idx) 
                                                          for idx 
                                                          in indices]
                instruction['result'] = use_function(*measured_values)
            except IndexError as e:
                instruction['result'] = f"{self.measured[group_idx]}"
            except ValueError:
                instruction['result'] = "INCOMPATIBLE LISTS"
            return instruction
        
        if 'use_calibrated' not in instruction:
            instruction['use_calibrated'] = False
        
        if 'use_multi_calibrated' not in instruction:
            instruction['use_multi_calibrated'] = False

        instruction['params']['default_names'] = [
            self.landmark_names[self.landmark_idx_map.index(idx)] 
            for idx in indices
        ]
        # print(f'__parse_measure: default_names GOT! {instruction['params']['default_names']}')
        
        use_calibrated = instruction['use_calibrated']
        use_multi_calibrated = instruction['use_multi_calibrated']
        try:
            use_landmark_group = self.landmark_list[group_idx]
        except Exception as e:
            raise Exception(f'__parse_measure: landmark_group fetching failed!\ngroup_idx: {group_idx}\nerror: {e}')
        # print(f'__parse_measure: use_landmark_group GOT! {use_landmark_group}')

        if use_calibrated:
            if not hasattr(self, 'calibrated_groups'):
                instruction['result'] = "NOT YET CALIBRATED"
                return instruction
            use_landmark_group = self.calibrated_groups[group_idx]
        # print(f'__parse_measure: use_calibrated HANDLED! {use_calibrated}')

        
        if not (use_multi_calibrated is False):
            if not hasattr(self,'multi_calibrated_groups'):
                instruction['result'] = "NOT YET MULTI CALIBRATED"
                return instruction
            if (len(self.multi_calibrated_groups)-1) < use_multi_calibrated:
                instruction['result'] = "MULTI CALIBRATION INDEX DOES NOT EXIST"
                return instruction
            use_landmark_group = self.multi_calibrated_groups[use_multi_calibrated][group_idx]
        # print(f'__parse_measure: use_multi_calibrated HANDLED! {use_multi_calibrated}')
        
        space = instruction['space']

        landmarks = [use_landmark_group[self.landmark_idx_map.index(idx)] for idx in indices]
        # print(f'__parse_measure: landmarks at indices {indices} fetched! {landmarks}')
        
        try:
            arguments = sanitise_arguments(
                function_name, landmarks, space)
        except Exception as e:
            raise Exception(f"__parse_measure: Failed to sanitise arguements {function_name}, {landmarks}, {space}", e)
        
        try:
            result = use_function(*arguments)
        except Exception as e:
            result = [-1]*4
            print(f'__parse_measure: Failed perfoming measurement:\n{instruction}\n{e}')

        return instruction|{'result':result}
            
    def measure(self,
                inputs:dict):
        '''
        Measures properties using toolboxes

        Accepted instructions:
         - angle_point
         - angle_vector
         - displacement
         - distance
         - bounding_box_size

        Example usage:
         Look at measurement_protocols script
        
        Look at toolbox documentation for usage info

        Params:
         - inputs(any): Accepts any arguments

        Returns:
         None, measured properties are accesible from 'measured' attribute a list
        ''' 
        self.measured = []

        try:
            assert len(sum(self.landmark_list,[]))>0
        except:
            return

        for group_idx in range(len(self.landmark_list)):
            self.measured += [[]]
            for in_ in inputs:
                try:
                    self.measured[group_idx] += [self.__parse_measure(group_idx, in_)]
                except Exception as e:
                    raise Exception(f"measure: Failed measuring {in_}, {group_idx}",e)

                
if __name__=='__main__':
    pass
        