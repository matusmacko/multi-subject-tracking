import random
import os
import cv2
from pathlib import Path

# pair of joints that form bones
BONES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [2, 6],
    [3, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [7, 12],
    [7, 13],
    [10, 11],
    [11, 12],
    [13, 14],
    [14, 15],
]

# directory with original images
IMAGE_DIR = "data/visualize/source/04"

# directory where to write images with bbox and skeleton overlay
OUTPUT_DIR = "data/visualize/results/04-base"


def read_joints(joints):
    if joints == ["\n"]:
        return []

    result = []

    for i in range(16):
        result.append((float((joints[i * 2])), float(joints[i * 2 + 1])))

    return result


##
# Visulizes images with bounding box and skeleton overlay
# @ track file - file with tracks in format: frame, id, x, y, w, h, skeleton data
# @ seqID - denotes sequences id (same with directory name where images are stored)
##
def visualize(track_file, seqID):
    counter = 0
    write_dir = OUTPUT_DIR + "/"

    # prepare different color for different ids
    colors = []
    for i in range(0, 1000):
        r, g, b = (
            round(255 * random.random()),
            round(255 * random.random()),
            round(255 * random.random()),
        )
        colors.append((r, g, b))

    if not (os.path.exists(write_dir)):
        os.mkdir(write_dir, 0o755)

    # read and reorganize tracks by frame number as a key
    dict = {}
    with open(track_file, "r") as file:
        for line in file:
            attr = line.split(",")
            frame = int(attr[0])
            id = int(attr[1])
            bb_x = round(float(attr[2]))
            bb_y = round(float(attr[3]))
            w = round(float(attr[4]))
            h = round(float(attr[5]))

            # READ JOINTS
            # depends on the format of joints in the file
            joints = read_joints(attr[6:])

            if frame in dict:
                meta = dict[frame]
            else:
                meta = []

            meta.append([id, bb_x, bb_y, w, h, joints])
            dict[frame] = meta

            # visualize (frame by frame)
            for frame in dict:
                img_path = IMAGE_DIR + "/" + str(frame).zfill(6) + ".jpg"
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # DRAW BOXES
            for index, meta in enumerate(dict[frame]):
                image = cv2.rectangle(
                    image,
                    (meta[1], meta[2]),
                    (meta[1] + meta[3], meta[2] + meta[4]),
                    (255, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                    shift=0,
                )

                continue

                # DRAW JOINTS
                joints = meta[5]
                if (len(joints) < 1):
                    continue
                for joint in joints:
                    image = cv2.circle(
                        image,
                        (int(float(joint[0])), int(float(joint[1]))),
                        5,
                        (255, 0, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                        shift=0
                    )

                # DRAW BONES
                for bone in BONES:
                    image = cv2.line(
                        image,
                        (int(joints[bone[0]][0]), int(joints[bone[0]][1])),
                        (int(joints[bone[1]][0]), int(joints[bone[1]][1])),
                        (2, 106, 253),
                        3,
                        cv2.LINE_AA
                    )

            frame_nr = Path(img_path).resolve().stem
            cv2.imwrite(write_dir + "/" + frame_nr + ".jpg", image)

            counter += 1
            print(" visualized {} / {}    ".format(counter, len(dict)), end="\r")
