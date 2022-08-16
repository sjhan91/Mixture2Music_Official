import numpy as np


### Standard Program Numbers ###
################################
FLUTE = 40  # flute

PIANO = 0  # acoustic grand piano
PIANO_RANGE = np.arange(0, 8)

GUITAR = 25  # clean electric guitar
GUITAR_RANGE = np.arange(24, 32)

BASS = 33  # finger electric bass
BASS_RANGE = np.arange(32, 40)

STRING = 49  # string ensemble 1
STRING_RANGE = np.arange(48, 56)

### Multi-track Numbers ###
###########################
TOTAL_NUM_INST = 6
GENERAL_NUM_PITCH = 72

MELODY_ABS_NUM = np.arange(24, 96)  # C1 ~ B6 (72)
MELODY_REL_NUM = 0

PIANO_ABS_NUM = np.arange(24, 96)  # C1 ~ B6 (72)
PIANO_REL_NUM = 1

GUITAR_ABS_NUM = np.arange(24, 96)  # C1 ~ B6 (72)
GUITAR_REL_NUM = 2

BASS_ABS_NUM = np.arange(24, 96)  # C1 ~ B6 (72)
BASS_REL_NUM = 3

STRING_ABS_NUM = np.arange(24, 96)  # C1 ~ B6 (72)
STRING_REL_NUM = 4

DRUM_ABS_NUM = np.arange(0, 9)  # 9 components (9)
DRUM_REL_NUM = 5

STANDARD_DRUM = [35, 38, 42, 45, 46, 48, 49, 50, 51]
