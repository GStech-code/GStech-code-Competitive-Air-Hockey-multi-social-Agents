from enum import IntEnum
from .defend_line import DefendLine
from .intercept import Intercept
from .pass_shot import PassShot
from .fast_shot import FastShot

OBJECTIVES = [DefendLine, Intercept, PassShot, FastShot]

class ObjectiveEnum(IntEnum):
    DEFEND_LINE = 0
    INTERCEPT = 1
    PASS_SHOT = 2
    FAST_SHOT = 3

OBJECTIVES_DICT = {
    ObjectiveEnum.DEFEND_LINE: DefendLine,
    ObjectiveEnum.INTERCEPT: Intercept,
    ObjectiveEnum.PASS_SHOT: PassShot,
    ObjectiveEnum.FAST_SHOT: FastShot,
    }

