from src import settings


def write(output_file, data, trajectories):
    """Write result to given filename."""
    with open(output_file, "w") as f:
        for frame_number, frame in data.items():
            for bounding_box in frame:
                if bounding_box.get("cache"):
                    continue

                if (
                    trajectories[bounding_box["person_id"]]
                    < settings.MINIMUM_TRAJECTORY_LENGTH
                ):
                    continue

                f.write(
                    "{},{},{},-1,-1,-1,-1\n".format(
                        frame_number,
                        bounding_box["person_id"],
                        ",".join([str(x) for x in bounding_box["bounding_box"]]),
                    )
                )
