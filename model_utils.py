class AlternateLandmarks:
    DRESIO = [
        '13 ORIGINAL 11,12 0.5 mid shoulder',
        '16 ORIGINAL 11,13 0.5 left upper arm',
        '17 ORIGINAL 12,14 0.5 right upper arm',
        '20 ORIGINAL 13,15 0.5 left forearm',
        '21 ORIGINAL 14,16 0.5 right forearm',
        '30 ORIGINAL 23,24 0.5 mid hip',
        '33 ORIGINAL 23,25 0.5 left thigh',
        '34 ORIGINAL 24,26 0.5 right thigh',
        '37 ORIGINAL 25,27 0.5 left calf',
        '38 ORIGINAL 26,28 0.5 right calf',
    ]

    NAT_CUSTOM = [
        '200 ORIGINAL 29,30,31,32 0.25 mid feet',
        '201 ORIGINAL 11,12,23,24 0.15,0.15,0.35,0.35 navel'
    ]

    TORSO_CENTER = [
        '300 LIVE 13,30 0.5 average torso'
    ]

    SPINE = [
        '400 LIVE 13,30 0.8,0.2 spine 1',
        '401 LIVE 13,30 0.4,0.6 spine 2',
        '402 LIVE 13,30 0.2,0.8 spine 3',
    ]

    CENTER_OF_MASS = [
        '5000 LIVE 0,13,30,24,25,20,21,16,17,35,36,37,38,33,34 0.08,0.25,0.25,0.007,0.007,0.016,0.016,0.027,0.027,0.015,0.015,0.045,0.045,0.1,0.1 center of mass'
    ]

class ModelIndices:
    HAND_MODEL = 0
    POSE_MODEL = 1


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