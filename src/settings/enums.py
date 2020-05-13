from enum import Enum, auto


class ModalityType(Enum):
    JOINTS = auto()
    BOUNDING_BOXES_IOU = auto()
    BOUNDING_BOXES_DISTANCE = auto()


class RefinementType(Enum):
    DISTANCE = auto()
    IOU = auto()
