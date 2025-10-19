from enum import IntEnum
from .defend_line import DefendLine
from .intercept import Intercept
from .pass_shot import Pass_Shot
from .fast_shot import FastShot

OBJECTIVES = [DefendLine, Intercept, Pass_Shot, FastShot]


class ObjectiveEnum(IntEnum):
    DEFEND_LINE = 0
    INTERCEPT = 1
    PASS_SHOT = 2
    FAST_SHOT = 3
