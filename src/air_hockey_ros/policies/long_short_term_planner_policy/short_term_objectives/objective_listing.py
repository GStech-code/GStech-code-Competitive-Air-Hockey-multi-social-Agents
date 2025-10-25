# objective_listing.py
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


OBJECTIVE_ENUMS = [ObjectiveEnum.DEFEND_LINE, ObjectiveEnum.INTERCEPT, ObjectiveEnum.PASS_SHOT, ObjectiveEnum.FAST_SHOT]

OBJECTIVES_DICT = {
    ObjectiveEnum.DEFEND_LINE: DefendLine,
    ObjectiveEnum.INTERCEPT: Intercept,
    ObjectiveEnum.PASS_SHOT: PassShot,
    ObjectiveEnum.FAST_SHOT: FastShot,
}

OBJECTIVE_COMBINATIONS = {
    1: [(ObjectiveEnum.DEFEND_LINE,),
        (ObjectiveEnum.INTERCEPT,),
        (ObjectiveEnum.FAST_SHOT,)],
    2: [(ObjectiveEnum.DEFEND_LINE, ObjectiveEnum.INTERCEPT),
        (ObjectiveEnum.DEFEND_LINE, ObjectiveEnum.FAST_SHOT),
        (ObjectiveEnum.PASS_SHOT, ObjectiveEnum.FAST_SHOT)],
    3: [(ObjectiveEnum.DEFEND_LINE, ObjectiveEnum.DEFEND_LINE, ObjectiveEnum.INTERCEPT),
        (ObjectiveEnum.DEFEND_LINE, ObjectiveEnum.INTERCEPT, ObjectiveEnum.FAST_SHOT),
        (ObjectiveEnum.PASS_SHOT, ObjectiveEnum.PASS_SHOT, ObjectiveEnum.FAST_SHOT),
        (ObjectiveEnum.DEFEND_LINE, ObjectiveEnum.FAST_SHOT, ObjectiveEnum.FAST_SHOT)],
    4: [(ObjectiveEnum.DEFEND_LINE, ObjectiveEnum.DEFEND_LINE, ObjectiveEnum.INTERCEPT, ObjectiveEnum.INTERCEPT),
        (ObjectiveEnum.DEFEND_LINE, ObjectiveEnum.INTERCEPT, ObjectiveEnum.FAST_SHOT, ObjectiveEnum.FAST_SHOT),
        (ObjectiveEnum.DEFEND_LINE, ObjectiveEnum.DEFEND_LINE, ObjectiveEnum.PASS_SHOT, ObjectiveEnum.FAST_SHOT),
        (ObjectiveEnum.PASS_SHOT, ObjectiveEnum.PASS_SHOT, ObjectiveEnum.FAST_SHOT, ObjectiveEnum.FAST_SHOT)]
}
