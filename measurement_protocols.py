HAND_CROSSED = {
    'protocol_name':'HANDS CROSSED',
    'protocol_time_seconds':30,
    'protocol_state_condition':'time',
    'protocol_measurements':[
        {'function_name':'displacement',
         'space':'world',
         'use_calibrated':False,
         'indices':(19,13),
         'params':{'check_index':0,
                   'lower_bound':0}},
        {'function_name':'displacement',
         'space':'world',
         'use_calibrated':False,
         'indices':(18,13),
         'params':{'check_index':0,
                   'upper_bound':0}},
        {'function_name':'displacement',
         'space':'world', 
         'use_calibrated':False, 
         'indices':(19,300),
         'params':{'check_index':1,
                   'lower_bound':0}},
        {'function_name':'displacement',
         'space':'world',
         'use_calibrated':False,
         'indices':(18,300),
         'params':{'check_index':1,
                   'lower_bound':0}},
    ]
}

LEGS_STRAIGHT = {
    'protocol_name':'LEGS STRAIGHT',
    'protocol_time_seconds':10,
    'protocol_state_condition':'time',
    'protocol_measurements':[
        {'function_name':'angle_point',
        'indices':(32,29,12),
        'space':'world',
        'use_calibrated':False,
        'params':{'check_index':3,
                'lower_bound':150}},
        {'function_name':'angle_point',
        'indices':(31,28,11),
        'space':'world',
        'use_calibrated':False,
        'params':{'check_index':3,
                'lower_bound':150}},
        {'function_name':'angle_point',
        'indices':(36,32,29),
        'space':'world',
        'use_calibrated':False,
        'params':{'check_index':0,
                'lower_bound':145}},
        {'function_name':'angle_point',
        'indices':(35,31,28),
        'space':'world',
        'use_calibrated':False,
        'params':{'check_index':0,
                'lower_bound':145}}
    ]
}
CTSIB = {
    'protocol_name':'CTSIB',
    'protocol_time_seconds':30,
    'protocol_state_condition':'time',
    'protocol_measurements':[
        # Calibrations
        {'function_name':'distance',
        'space':'world',
        'use_calibrated':True,
        'indices':(35,36),
        'params':{'do_draw':True}},
        {'function_name':'distance',
        'space':'world', 
        'use_calibrated':True ,
        'indices':(41,42),
        'params':{'do_draw':True}},
        {'function_name':'displacement',
        'space':'world',
        'use_calibrated':True,
        'indices':(28,35),
        'params':{'do_draw':True}},
        {'function_name':'displacement',
        'space':'world',
        'use_calibrated':True,
        'indices':(29,36),
        'params':{'do_draw':True}},

        # Arms crossed on torso
        *HAND_CROSSED['protocol_measurements'],

        # Feet not moving
        {'function_name':'distance', 
        'space':'world', 
        'use_calibrated':False, 
        'indices':(35,36),
        'params':{'do_draw':False}},
        {'function_name':'distance',
        'space':'world',
        'use_calibrated':False, 
        'indices':(41,42), 
        'params':{'do_draw':False}},
        {'function_name':'displacement', 
        'space':'world', 
        'use_calibrated':False, 
        'indices':(28,35),
        'params':{'do_draw':False}},
        {'function_name':'displacement',
        'space':'world', 
        'use_calibrated':False,
        'indices':(29,36),
        'params':{'do_draw':False}},
        {'function_name':'compare_ratio',
        'indices':(8,0),
        'params':{'check_index':0,
                'lower_bound':0.95,
                'upper_bound':1.05}},
        {'function_name':'compare_ratio',
        'indices':(9,1),         
        'params':{'check_index':0,
                'lower_bound':0.95,
                'upper_bound':1.05}},
        {'function_name':'compare_ratio',
        'indices':(10,2),
        'params':{'check_index':1,
                'lower_bound':0.9,
                'upper_bound':1.1}},
        {'function_name':'compare_ratio',
        'indices':(11,3),
        'params':{'check_index':1,
                'lower_bound':0.9,
                'upper_bound':1.1}}
    ]
}
DEMMI_RTS = {
    'protocol_name':'DEMMI ROLL TO SIDE',
    'protocol_time_seconds':1000,
    'protocol_state_condition':'time',
    'protocol_measurements':[
        {'function_name':'angle_point',
        'indices':(11,12,'HL'),
        'use_calibrated':False,
        'space':'screen',
        'params':{'check_index':3,
                'lower_bound':80}},
        {'function_name':'angle_point',
        'indices':(28,29,'HL'),
        'use_calibrated':False,
        'space':'screen',
        'params':{'check_index':3,
                'lower_bound':80}},
    ]
}

DEMMI_LTS = {
    'protocol_name':'DEMMI LYING TO SITTING',
    'protocol_time_seconds':1000,
    'protocol_state_condition':None,
    'protocol_measurements':[
        {'function_name':'angle_point',
         'indices':(11,28,33),
         'space':'screen',
         'params':{
             'check_index':3,
             'upper_bound':15,
         }},
         {'function_name':'angle_point',
         'indices':(11,28,33),
         'space':'screen',
         'params':{
             'check_index':3,
             'lower_bound':70,
         }}
    ]
}

DEMMI_SIT_UNSUPPORTED = {
    'protocol_name':'DEMMI SIT UNSUPPORTED',
    'protocol_time_seconds':10,
    'protocol_state_condition':None,
    'protocol_measurements':[
        *LEGS_STRAIGHT['protocol_measurements'],
        *HAND_CROSSED['protocol_measurements'],
    ]
}

DEMMI_BRIDGE = {
    'protocol_name':'DEMMI BRIDGE',
    'protocol_time_seconds':1000,
    'protocol_state_condition':'time',
    'protocol_measurements':[
        {'function_name':'distance',
        'indices':(36,29),
        'space':'world',
        'use_calibrated':True,
        'params':{'do_draw':False}},
        {'function_name':'distance',
        'indices':(36,29),
        'space':'world',
        'use_calibrated':False,
        'params':{'do_draw':False}},
        {'function_name':'compare_ratio',
        'indices':(1,0),
        'params':{'check_index':0,
                'lower_bound':0.58,
                'upper_bound':0.62}},

        # Rest
        {'function_name':'angle_point',
        'indices':(36,32,29),
        'space':'world',
        'use_calibrated':False,
        'params':{'check_index':3,
                'lower_bound':50,
                'upper_bound':70,}},
        {'function_name':'angle_point',
        'indices':(32,29,12),
        'space':'world',
        'use_calibrated':False,
        'params':{'check_index':3,
                'lower_bound':110,
                'upper_bound':130,}},
        {'function_name':'angle_point',
        'indices':(36,29,12),
        'space':'world',
        'use_calibrated':False,
        'params':{'check_index':3,
                'lower_bound':170,
                'upper_bound':180,}},

        # Bridge
        {'function_name':'angle_point',
        'indices':(36,32,29),
        'space':'world',
        'use_calibrated':False,
        'params':{'check_index':3,
                'lower_bound':70}},
        {'function_name':'angle_point',
        'indices':(32,29,12),
        'space':'world',
        'use_calibrated':False,
        'params':{'check_index':3,
                'lower_bound':170}},
        {'function_name':'angle_point',
        'indices':(36,29,12),
        'space':'world',
        'use_calibrated':False,
        'params':{'check_index':3,
                'upper_bound':150}}
    ]
}

DEMMI_STAND = {
    'protocol_name':'DEMMI STAND',
    'protocol_time_seconds': 1000,
    'protocol_state_condition':'time',
    'protocol_measurements': [
        *LEGS_STRAIGHT['protocol_measurements'],
        *HAND_CROSSED['protocol_measurements']
    ]
}

DEMMI_TANDEM_STAND_LEFT = {
    'protocol_name':'DEMMI TANDEM STAND LEFT FOOT IN FRONT',
    'protocol_time_seconds': 1000,
    'protocol_state_condition':'time',
    'protocol_measurements': [
        {'function_name':'displacement',
         'indices':(39,42),
         'space':'world',
         'use_calibrated':False,
         'params':{'check_index':0,
                   'upper_bound':0}}
    ]
}
DEMMI_TANDEM_STAND_RIGHT = {
    'protocol_name':'DEMMI TANDEM STAND RIGHT FOOT IN FRONT',
    'protocol_time_seconds': 1000,
    'protocol_state_condition':'time',
    'protocol_measurements': [
        {'function_name':'displacement',
         'indices':(40,41),
         'space':'world',
         'use_calibrated':False,
         'params':{'check_index':0,
                   'lower_bound':0}}
    ]
}

DEMMI_TOES_STAND = {
    'protocol_name':'DEMMI STAND ON TOES',
    'protocol_time_seconds': 1000,
    'protocol_state_condition':'time',
    'protocol_measurements': [
        {'function_name':'displacement',
         'indices':(39,41),
         'space':'world',
         'use_calibrated':False,
         'params':{
             'check_index':1,
             'upper_bound':0}},
        {'function_name':'displacement',
         'indices':(40,42),
         'space':'world',
         'use_calibrated':False,
         'params':{
             'check_index':1,
             'upper_bound':0}}
    ]
}

DEMMI_TOES_STAND = {
    'protocol_name':'DEMMI STAND ON TOES',
    'protocol_time_seconds': 1000,
    'protocol_state_condition':'time',
    'protocol_measurements': [
        {'function_name':'displacement',
         'indices':(39,41),
         'space':'world',
         'use_calibrated':False,
         'params':{
             'check_index':1,
             'upper_bound':0}},
        {'function_name':'displacement',
         'indices':(40,42),
         'space':'world',
         'use_calibrated':False,
         'params':{
             'check_index':1,
             'upper_bound':0}}
    ]
}

DEMMI_FORWARD_BEND = {
    'protocol_name':'DEMMI PICK UP PEN FROM FLOOR',
    'protocol_time_seconds': 1000,
    'protocol_state_condition':'time',
    'protocol_measurements':[
        {'function_name':'displacement',
         'indices':(19,36),
         'use_calibrated':False,
         'space':'world',
         'params':{
             'check_index':1,
             'upper_bound':0}},
        {'function_name':'displacement',
         'indices':(18,35),
         'use_calibrated':False,
         'space':'world',
         'params':{
             'check_index':1,
             'upper_bound':0}}
    ]
}

def WALK_BACKWARDS_DETECTION(measurement_information:list):
    """[UNIMPLEMENTED]
    Detects walking backwards

    Args:
        measurement_information (list): Recorded frames with measurement
    """
    # EVENTS
    # for 

DEMMI_WALK_BACKWARDS = {
    'protocol_name':'DEMMI WALK BACKWARDS 4 STEPS',
    'protocol_time_seconds':30,
    'protocol_state_condition': WALK_BACKWARDS_DETECTION,
    'protocol_measurements':[
        {'function_name':'displacement',
         'indices':(30,41),
         'space':'world',
         'use_calibrated':False,
         'params':{
             'check_index':0,
             'upper_bound':-20}},
        {'function_name':'displacement',
         'indices':(30,42),
         'space':'world',
         'use_calibrated':False,
         'params':{
             'check_index':0,
             'upper_bound':-20}}
    ]
}

def DEMMI_JUMP_DETECTION(measurement_information:list):
    """[UNIMPLEMENTED]
    Detects jumping

    Args:
        measurement_information (list): Recorded frames with measurement
    """
    return

DEMMI_JUMP = {
    'protocol_name':'DEMMI JUMP',
    'protocol_time_seconds':1000,
    'protocol_state_condition': DEMMI_JUMP_DETECTION,
    'protocol_measurements':[
        {'function_name':'displacement',
         'indices':(30,'VD'),
         'space':'screen',
         'use_calibrated':True,
         'params':{'do_draw':'True'}},
        {'function_name':'displacement',
         'indices':(30,'VD'),
         'space':'screen',
         'params':{'do_draw':'True'}},
        {'function_name':'compare_substract',
         'indices':(0,1),
         'params':{
             'check_index':1,
             'lower_bound':0.05}},
    ]
}

HIMAT_WALK = {
    'protocol_name':'WALK TOWARDS CAMERA',
    'protocol_time_seconds':1000,
    'protocol_state_condition':None,
    'protocol_measurements':[
        {'function_name':'distance',
         'indices':(13,30),
         'space':'screen',
         'use_multi_calibrated':0,
         'params':{'do_draw':True}},
        {'function_name':'distance',
         'indices':(13,30),
         'space':'screen',
         'use_multi_calibrated':1,
         'params':{'do_draw':True}},
        {'function_name':'distance', #LIVE DISTANCE
        'indices':(13,30),
        'space':'screen',
        'use_calibrated':False,
        'params':{'do_draw':True}},
        {'function_name':'compare_ratio',
         'indices':(2,0),
         'params':{
             'check_index':3,
             'lower_bound':0.9,
             'upper_bound':1.1,
         }},
         {'function_name':'compare_ratio',
         'indices':(2,1),
         'params':{
             'check_index':3,
             'lower_bound':0.9,
             'upper_bound':1.1}}
    ]
}

HIMAT_WALK_BACKWARDS = {
    'protocol_name':'HIMAT BACKWARDS WALK',
    'protocol_time_seconds':1000,
    'protocol_state_condition':None,
    'protocol_measurements':[
        {'function_name':'displacement',
         'indices':(11,12),
         'space':'screen',
         'use_calibrated':False,
         'params':{
             'check_index':0,
             'upper_bound':0}},
        *HIMAT_WALK['protocol_measurements']
    ]
}

HIMAT_WALK_ON_TOES = {
    'protocol_name': 'WALK ON TOES',
    'protocol_time_seconds': 1000,
    'protocol_state_condition': None,
    'protocol_measurements': [
        # Calibration measurements - standing heights (Y axis)
        {'function_name': 'distance',
         'indices': (13,30),
         'space': 'screen',
         'use_multi_calibrated': 0,
         'params': {'do_draw': True}},
        {'function_name': 'distance',
         'indices': (13,30),
         'space': 'screen',
         'use_multi_calibrated': 1,
         'params': {'do_draw': True}},
        
        # Live test measurement (Y axis of distance)
        {'function_name': 'distance',
         'indices': (13,30),
         'space': 'screen',
         'use_calibrated': False,
         'params': {'do_draw': True}},
        
        # Ankle and toe height comparisons (Y axis in world space)
        {'function_name': 'compare_ratio',
         'indices': (39,41),  # Left heel vs Left toe - check Y axis (1)
         'space': 'world',
         'use_calibrated': False,
         'params': {
             'check_index': 1,  # Y axis for height comparison
             'upper_bound': 0   # VAR < 0 (toe Y > heel Y)
         }},
        {'function_name': 'compare_ratio',
         'indices': (40,42),  # Right heel vs Right toe - check Y axis (1)
         'space': 'world',
         'use_calibrated': False,
         'params': {
             'check_index': 1,  # Y axis for height comparison
             'upper_bound': 0   # VAR < 0 (toe Y > heel Y)
         }},
        
        # Distance comparisons (using general measurement index 3)
        {'function_name': 'compare_ratio',
         'indices': (2,0),  # Compare_ratio live distance to calibration 0
         'params': {
             'check_index': 1,  # General distance measurement
             'lower_bound': 0.9,
             'upper_bound': 1.1,
         }},
        {'function_name': 'compare_ratio',
         'indices': (2,1),  # Compare live distance to calibration 1
         'params': {
             'check_index': 1,  # General distance measurement
             'lower_bound': 0.9,
             'upper_bound': 1.1
         }}
    ]
}
HIMAT_WALK_OVER_OBSTACLE = {
    'protocol_name': 'WALK OVER OBSTACLE',
    'protocol_time_seconds': 1000,
    'protocol_state_condition': None,
    'protocol_measurements': [
        # Calibration measurements
        {'function_name': 'distance',
         'indices': (13,30),
         'space': 'screen',
         'use_multi_calibrated': 0,
         'params': {'do_draw': True}},
        {'function_name': 'distance',
         'indices': (13,30),
         'space': 'screen',
         'use_multi_calibrated': 1,
         'params': {'do_draw': True}},
        {'function_name': 'distance',
         'indices': (13,30),
         'space': 'screen',
         'use_multi_calibrated': 2,
         'params': {'do_draw': True}},
        
        # Live distance measurement
        {'function_name': 'distance',
         'indices': (13,30),
         'space': 'screen',
         'use_calibrated': False,
         'params': {'do_draw': True}},
        
        # Obstacle clearance checks (Y axis comparisons)
        {'function_name': 'compare_ratio',
         'indices': (42,37),  # Right toe vs Left calf - check Y axis
         'space': 'screen',
         'use_calibrated': False,
         'params': {
             'check_index': 1,  # Y axis height comparison
             'lower_bound': 0   # 0 or + (toe Y >= calf Y)
         }},
        {'function_name': 'compare_ratio',
         'indices': (41,38),  # Left toe vs Right calf - check Y axis
         'space': 'screen',
         'use_calibrated': False,
         'params': {
             'check_index': 1,  # Y axis height comparison
             'lower_bound': 0   # 0 or + (toe Y >= calf Y)
         }},
        
        # Distance timing comparisons
        {'function_name': 'compare_ratio',
         'indices': (3,0),  # Start: live vs 15m calibration
         'params': {
             'check_index': 1,  # General distance
             'lower_bound': 0.9,
             'upper_bound': 1.1,
         }},
        {'function_name': 'compare_ratio',
         'indices': (3,2),  # At obstacle: live vs 10m calibration
         'params': {
             'check_index': 1,  # General distance
             'lower_bound': 0.9,
             'upper_bound': 1.1
         }},
        {'function_name': 'compare_ratio',
         'indices': (3,1),  # Stop: live vs 5m calibration
         'params': {
             'check_index': 1,  # General distance
             'lower_bound': 0.9,
             'upper_bound': 1.1
         }}
    ]
}

HIMAT_RUNNING = {
    'protocol_name': 'RUNNING',
    'protocol_time_seconds': 1000,
    'protocol_state_condition': None,
    'protocol_measurements': [
        # Calibration measurements
        {'function_name': 'distance',
         'indices': (13,30),
         'space': 'screen',
         'use_multi_calibrated': 0,
         'params': {'do_draw': True}},
        {'function_name': 'distance',
         'indices': (13,30),
         'space': 'screen',
         'use_multi_calibrated': 1,
         'params': {'do_draw': True}},
        
        # Live distance measurement
        {'function_name': 'distance',
         'indices': (13,30),
         'space': 'screen',
         'use_calibrated': False,
         'params': {'do_draw': True}},
        
        # Timing comparisons
        {'function_name': 'compare_ratio',
         'indices': (2,0),  # Start: 90-110% of var 0
         'params': {
             'check_index': 1,
             'lower_bound': 0.9,
             'upper_bound': 1.1
         }},
        {'function_name': 'compare_ratio',
         'indices': (2,1),  # Stop: 90-110% of var 1
         'params': {
             'check_index': 1,
             'lower_bound': 0.9,
             'upper_bound': 1.1
         }}
    ]
}
HIMAT_SKIPPING = {
    'protocol_name': 'SKIPPING',
    'protocol_time_seconds': 1000,
    'protocol_state_condition': None,
    'protocol_measurements': [
        # Calibration measurements
        {'function_name': 'distance',
         'indices': (13,30),
         'space': 'screen',
         'use_multi_calibrated': 0,
         'params': {'do_draw': True}},
        {'function_name': 'distance',
         'indices': (13,30),
         'space': 'screen',
         'use_multi_calibrated': 1,
         'params': {'do_draw': True}},
        
        # Live distance measurement
        {'function_name': 'distance',
         'indices': (13,30),
         'space': 'screen',
         'use_calibrated': False,
         'params': {'do_draw': True}},
        
        # Screen center reference
        {'function_name': 'screen_center',
         'indices': (0,0),  # Middle of screen
         'space': 'screen',
         'use_calibrated': False,
         'params': {'do_draw': True}},
        
        # Position tracking
        {'function_name': 'position',
         'indices': (0,),  # Noise position
         'space': 'screen',
         'use_calibrated': False,
         'params': {'check_index': 1}},  # Y pixel
        {'function_name': 'position',
         'indices': (13,),  # Mid shoulder position
         'space': 'screen',
         'use_calibrated': False,
         'params': {'check_index': 1}},  # Y pixel
        
        # Timing comparisons
        {'function_name': 'compare_ratio',
         'indices': (2,0),  # Start: 90-110% of var 0
         'params': {
             'check_index': 1,
             'lower_bound': 0.9,
             'upper_bound': 1.1
         }},
        {'function_name': 'compare_ratio',
         'indices': (2,1),  # Stop: 90-110% of var 1
         'params': {
             'check_index': 1,
             'lower_bound': 0.9,
             'upper_bound': 1.1
         }}
    ]
}

HIMAT_HOP_FORWARD = {
    'protocol_name': 'HOP FORWARD',
    'protocol_time_seconds': 1000,
    'protocol_state_condition': None,
    'protocol_measurements': [
        # Calibration measurements
        {'function_name': 'distance',
         'indices': (13,30),  # 11m away
         'space': 'screen',
         'use_multi_calibrated': 0,
         'params': {'do_draw': True}},
        {'function_name': 'distance',
         'indices': (13,30),  # 1m away
         'space': 'screen',
         'use_multi_calibrated': 1,
         'params': {'do_draw': True}},
        
        # Live distance measurement
        {'function_name': 'distance',
         'indices': (13,30),
         'space': 'screen',
         'use_calibrated': False,
         'params': {'do_draw': True}},
        
        # Foot off floor checks
        {'function_name': 'compare_ratio',
         'indices': (42,41),  # R foot off floor (right toe vs left toe)
         'space': 'screen',
         'use_calibrated': False,
         'params': {
             'check_index': 1,  # Y axis
             'lower_bound': 0   # + (right toe Y > left toe Y)
         }},
        {'function_name': 'compare_ratio',
         'indices': (41,42),  # L foot off floor (left toe vs right toe)
         'space': 'screen',
         'use_calibrated': False,
         'params': {
             'check_index': 1,  # Y axis
             'lower_bound': 0   # + (left toe Y > right toe Y)
         }},
        
        # Timing comparisons
        {'function_name': 'compare_ratio',
         'indices': (2,0),  # Start: 90-110% of var 0
         'params': {
             'check_index': 1,
             'lower_bound': 0.9,
             'upper_bound': 1.1
         }},
        {'function_name': 'compare_ratio',
         'indices': (2,1),  # Stop: 90-110% of var 1
         'params': {
             'check_index': 1,
             'lower_bound': 0.9,
             'upper_bound': 1.1
         }}
    ]
}

WALCT = {
    'protocol_name':'WALCT',
    'protocol_time_seconds':1000,
    'protocol_state_condition':None,
    'protocol_measurements':[
        {'function_name':'distance',
         'indices':(13,30),
         'space':'screen',
         'use_multi_calibrate':0,         
         'params':{
            'do_draw':True}},
        {'function_name':'displacement',
         'indices':('HL',5000),
         'space':'screen',
         'use_multi_calibrate':0,         
         'params':{
            'do_draw':True}},
        
        {'function_name':'distance',
         'indices':(13,30),
         'space':'screen',
         'use_multi_calibrate':1,         
         'params':{
            'do_draw':True}},
        {'function_name':'displacement',
         'indices':('HL',5000),
         'space':'screen',
         'use_multi_calibrate':1,         
         'params':{
            'do_draw':True}},

        {'function_name':'distance',
         'indices':(13,30),
         'space':'screen',
         'use_multi_calibrate':2,         
         'params':{
            'do_draw':True}},
        {'function_name':'displacement',
         'indices':('HL',5000),
         'space':'screen',
         'use_multi_calibrate':2,         
         'params':{
            'do_draw':True}},

        {'function_name':'distance',
         'indices':(13,30),
         'space':'screen',
         'use_multi_calibrate':3,         
         'params':{
            'do_draw':True}},
        {'function_name':'displacement',
         'indices':('HL',5000),
         'space':'screen',
         'use_multi_calibrate':3,         
         'params':{
            'do_draw':True}},

        {'function_name':'distance',
         'indices':(13,30),
         'space':'screen',       
         'params':{
            'do_draw':True}},
        {'function_name':'displacement',
         'indices':('HL',5000),
         'space':'screen',       
         'params':{
            'do_draw':True}},

    ]
}

MSOT = {
    'protocol_name':'MSOT',
    'protocol_time_seconds':1000,
    'protocol_state_condition':None,
    'protocol_measurements':[
        {'function_name':'distance',
         'indices':(13,30),
         'space':'screen',
         'use_multi_calibrated':0,
         'params':{'do_draw':True}},
        {'function_name':'distance',
         'indices':(13,30),
         'space':'screen',
         'use_multi_calibrated':1,
         'params':{'do_draw':True}},
        {'function_name':'distance',
         'indices':(13,30),
         'space':'screen',
         'use_multi_calibrated':2,
         'params':{'do_draw':True}},
        {'function_name':'distance',
         'indices':(13,30),
         'space':'screen',
         'use_multi_calibrated':3,
         'params':{'do_draw':True}},
        {'function_name':'distance',
         'indices':(13,30),
         'space':'screen',
         'use_multi_calibrated':4,
         'params':{'do_draw':True}},
        {'function_name':'distance',
         'indices':(13,30),
         'space':'screen',
         'use_multi_calibrated':5,
         'params':{'do_draw':True}},

        {'function_name':'distance',
         'indices':(13,30),
         'space':'screen',
         'params':{'do_draw':True}},

        {'function_name':'compare_ratio',
         'indices':(13,30),
         'space':'screen',
         'use_multi_calibrated':5,
         'params':{'do_draw':True}},
    ]
}







