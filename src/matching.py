from shapely import geometry
import math
import munkres
import numpy

from . import settings
from .settings import enums


def get_bounding_box_center(bounding_box):
    """Get coordinates of center of given bounding box."""
    x_center = bounding_box[0] + bounding_box[2] / 2
    y_center = bounding_box[1] + bounding_box[3] / 2
    return x_center, y_center


def inverse_iou(first_box, second_box):
    """Calculate IoU between two given bounding boxes.

    Adapted code from here - https://medium.com/koderunners/intersection-over-union-516a3950269c.
    """
    x1, y1, w1, h1 = first_box
    x2, y2, w2, h2 = second_box
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0:  # No overlap
        return 1
    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I  # Union = Total Area - I
    return 1 - I / U


def invalid_match(first_detection, second_detection):
    """Check whether given match of two detections should be refined."""
    if not settings.DETECTION_MATCHING_REFINEMENT:
        return False

    if settings.DETECTION_MATCHING_REFINEMENT == enums.RefinementType.IOU:
        inverse_iou_value = inverse_iou(
            first_detection["bounding_box"], second_detection["bounding_box"]
        )
        return inverse_iou_value >= 1

    if settings.DETECTION_MATCHING_REFINEMENT == enums.RefinementType.DISTANCE:
        allowed_distance = max(
            first_detection["bounding_box"][2],
            first_detection["bounding_box"][3],
            second_detection["bounding_box"][2],
            second_detection["bounding_box"][3],
        )

        first_box_center = get_bounding_box_center(first_detection["bounding_box"])
        second_box_center = get_bounding_box_center(second_detection["bounding_box"])

        center_distance = math.sqrt(
            (first_box_center[0] - second_box_center[0]) ** 2
            + (first_box_center[1] - second_box_center[1]) ** 2
        )

        return allowed_distance < center_distance


def calculate_similarity(first_detection, second_detection):
    """Calculate similarity value between given detections."""
    if settings.MODALITY == enums.ModalityType.BOUNDING_BOXES_DISTANCE:
        first_box_center = get_bounding_box_center(first_detection["bounding_box"])
        second_box_center = get_bounding_box_center(second_detection["bounding_box"])

        return math.sqrt(
            (first_box_center[0] - second_box_center[0]) ** 2
            + (first_box_center[1] - second_box_center[1]) ** 2
        )

    if settings.MODALITY == enums.ModalityType.BOUNDING_BOXES_IOU:
        return inverse_iou(
            first_detection["bounding_box"], second_detection["bounding_box"]
        )

    if settings.MODALITY == enums.ModalityType.JOINTS:
        return sum(
            math.sqrt(
                sum(
                    (a - b) ** 2
                    for a, b in zip(
                        first_detection["joint_coordinates"][index],
                        second_detection["joint_coordinates"][index],
                    )
                )
            )
            for index in range(16)
        )


def greedy_algorithm(matrix):
    """Solve the assignment problem with greedy algorithm."""
    combinations = []
    solution = []

    for prev_index, prev in enumerate(matrix):
        for current_index, current in enumerate(prev):
            combinations.append(
                {
                    "value": current,
                    "current_index": current_index,
                    "prev_index": prev_index,
                }
            )

    combinations = sorted(combinations, key=lambda item: item["value"])

    used_prev = set()
    used_current = set()

    solution_length = min(len(matrix), len(matrix[0]))

    for item in combinations:
        if len(solution) == solution_length:
            return solution

        if (
            item["current_index"] not in used_current
            and item["prev_index"] not in used_prev
        ):
            used_prev.add(item["prev_index"])
            used_current.add(item["current_index"])
            solution.append((item["prev_index"], item["current_index"]))

    return solution


def interpolation(data, first_box, second_box):
    """Interpolate missing detections."""
    if not settings.INTERPOLATION:
        return

    counter = first_box["frame_index"]

    while counter + 1 < second_box["frame_index"]:
        counter += 1

        x = numpy.interp(
            counter,
            [first_box["frame_index"], second_box["frame_index"]],
            [first_box["bounding_box"][0], second_box["bounding_box"][0]],
        )

        y = numpy.interp(
            counter,
            [first_box["frame_index"], second_box["frame_index"]],
            [first_box["bounding_box"][1], second_box["bounding_box"][1]],
        )

        w = (
            numpy.interp(
                counter,
                [first_box["frame_index"], second_box["frame_index"]],
                [
                    first_box["bounding_box"][0] + first_box["bounding_box"][2],
                    second_box["bounding_box"][0] + second_box["bounding_box"][2],
                ],
            )
            - x
        )

        h = (
            numpy.interp(
                counter,
                [first_box["frame_index"], second_box["frame_index"]],
                [
                    first_box["bounding_box"][1] + first_box["bounding_box"][3],
                    second_box["bounding_box"][1] + second_box["bounding_box"][3],
                ],
            )
            - y
        )

        data.setdefault(counter, [])
        data[counter].append(
            {
                "person_id": first_box["person_id"],
                "bounding_box": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "interpolated": True,
            }
        )


def match(data, trajectories, prev_frame, current_frame):
    """Match detections from previous and current frame."""
    prev_count = len(prev_frame)
    current_count = len(current_frame)

    if not prev_count or not current_count:
        return set()

    matrix = [[None for _ in range(current_count)] for _ in range(prev_count)]

    for prev_frame_index in range(prev_count):
        for current_frame_index in range(current_count):
            similarity_value = calculate_similarity(
                prev_frame[prev_frame_index], current_frame[current_frame_index],
            )

            frame_difference = (
                current_frame[current_frame_index]["frame_index"]
                - prev_frame[prev_frame_index]["frame_index"]
            ) - 1

            # print(frame_difference * settings.MEMORY_DECAY_PERCENTAGE + 1)

            # decay
            similarity_value *= frame_difference * settings.MEMORY_DECAY_PERCENTAGE + 1

            matrix[prev_frame_index][current_frame_index] = similarity_value

    reused_entities = set()

    if settings.SPEED_ENHANCEMENT:
        solution = greedy_algorithm(matrix)
    else:
        solution = munkres.Munkres().compute(matrix)

    for prev_frame_index, current_frame_index in solution:
        if invalid_match(
            prev_frame[prev_frame_index], current_frame[current_frame_index],
        ):
            continue

        reused_entities.add(prev_frame[prev_frame_index]["person_id"])

        person_id = prev_frame[prev_frame_index]["person_id"]
        trajectories[person_id] += 1
        current_frame[current_frame_index]["person_id"] = person_id

        interpolation(
            data, prev_frame[prev_frame_index], current_frame[current_frame_index]
        )

    return reused_entities
