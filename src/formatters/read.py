def read(input_file):
    """Read data from given filename."""
    data = {}

    with open(input_file, "r") as f:
        for line in f.readlines():
            frame_bounding_box = line

            b = frame_bounding_box.split(",")
            data.setdefault(int(b[0]), []).append(
                {
                    "bounding_box": [float(x) for x in b[2:6]],
                    "joint_coordinates": [
                        # [float(x) for x in joint.split(",")]
                        # for joint in join_coordinates.split(";")
                    ],
                    "frame_index": int(b[0]),
                }
            )

    return data
