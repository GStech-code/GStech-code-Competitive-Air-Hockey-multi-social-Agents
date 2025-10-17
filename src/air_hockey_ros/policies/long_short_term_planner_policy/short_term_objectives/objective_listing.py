from enum import IntEnum
from .defend_line import DefendLine
from .fast_shot import FastShot

OBJECTIVES = [DefendLine, FastShot]

class ObjectiveEnum(IntEnum):
    DEFEND_LINE = 0
    INTERCEPT = 1
    CENTER_POSITION = 2
    PASS = 3
    FAST_SHOT = 4
    FORCAST_SHOT = 5
