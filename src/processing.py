from collections import defaultdict

from src import matching, settings


def process(data):
    """Track detections in given data."""
    id_count = 0
    frame_index = 0
    prev_frame = []
    empty_frames = []
    trajectories = defaultdict(int)
    identities_cache = {}

    while len(empty_frames) < 20:
        frame_index += 1
        current_frame = data.get(frame_index)

        if not current_frame:
            empty_frames.append(frame_index)
            continue

        if empty_frames:
            print("Missing frames no. {}".format(empty_frames))
            empty_frames = []

        reused_entities = matching.match(data, trajectories, prev_frame, current_frame,)

        # save unmatched detections to memory
        for item in prev_frame:
            if not item["person_id"] in reused_entities:
                identities_cache[item["person_id"]] = {
                    "frame_index": item["frame_index"],
                    "bounding_box": item["bounding_box"],
                    "joint_coordinates": item["joint_coordinates"],
                }

        # delete memory after TTL
        identities_cache = {
            person_id: data
            for person_id, data in identities_cache.items()
            if data["frame_index"] >= frame_index - settings.MEMORY_TTL
            and person_id not in reused_entities
        }

        # create new identities
        for bounding_box in current_frame:
            if not bounding_box.get("person_id"):
                id_count += 1
                bounding_box["person_id"] = id_count

            trajectories[bounding_box["person_id"]] += 1

        for person_id, item in identities_cache.items():
            current_frame.append(
                {
                    "bounding_box": item["bounding_box"],
                    "joint_coordinates": item["joint_coordinates"],
                    "person_id": person_id,
                    "cache": True,
                    "frame_index": item["frame_index"],
                }
            )

        prev_frame = current_frame
        print(f"Processed frame {frame_index}")

    return data, trajectories
