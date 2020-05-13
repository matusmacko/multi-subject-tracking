from src.settings import enums

# location of source and result files
SOURCE_PATH = "data/source-sdp"
RESULTS_PATH = "data/results"

# type of modality used in the tracker
MODALITY = enums.ModalityType.BOUNDING_BOXES_DISTANCE

# minimum length of trajectory
MINIMUM_TRAJECTORY_LENGTH = 7

# usage of greedy algorithm instead of Munkres
SPEED_ENHANCEMENT = True

# usage of refinement of detection matching
# use None or enums.RefinementType
DETECTION_MATCHING_REFINEMENT = None

# memory (i.e., naive re-identification)
MEMORY_TTL = 10
MEMORY_DECAY_PERCENTAGE = 0.5

# interpolation
INTERPOLATION = True
