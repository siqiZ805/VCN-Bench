ORDINAL_NAMES = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
}
SKIP_ROOMS = [
    "hallway",
    "other room",
    "balcony",
    "entryway/foyer/lobby",
    "porch/terrace/deck",
    "outdoor",
]
SKIP_OBJECTS = [
    "ceiling",
    "floor",
    "wall",
    "object",
    "window",
    "window frame",
    "door",
    "doorframe",
    "rail",
    "column",
    "stairs",
    "step",
    "balcony",
    "roof",
    "light",
    "toilet",
]
DIRECTIONS = ["NORTH", "WEST", "SOUTH", "EAST"]

ROOM_NAME_ALIASES = {
    "workout/gym/exercise": "gymnasium|gym|fitness room",
    "entryway/foyer/lobby": "entryway|foyer|lobby",
    "spa/sauna": "spa room|sauna room",
    "laundryroom/mudroom": "laundry room",
    "rec/game": "rec room|game room",
    "outdoor": "outdoor area",
    "utilityroom/toolroom": "utility room|tool room",
    "familyroom/lounge": "family room|lounge",
    "porch/terrace/deck": "porch|terrace|deck",
    "tv": "TV room",
}

TEST_SCENES = [
    "2t7WUuJeko7",
    "5ZKStnWn8Zo",
    "ARNzJeq3xxb",
    "RPmz2sHmrrY",
    "UwV83HsGsw3",
    "Vt2qJdWjCF2",
    "WYY7iVyf5p8",
    "YFuZgdQ5vWj",
    "YVUC4YcDtcY",
    "fzynW3qQPVF",
    "gYvKGZ5eRqb",
    "gxdoqLR6rwA",
    "jtcxE69GiFV",
    "pa4otMbVnkk",
    "q9vSo1VnCiC",
    "rqfALeAoiTq",
    "wc2JMjhGNzB",
    "yqstnuAEVhm",
]

TASK_TYPE_TO_INDEX = {
    "object_instance": 0,
    "room_temporal_count/other_room": 1,
    "object_category": 2,
    "room_temporal_order/single": 3,
    "room_temporal_count/cur_room": 4,
    "room_spatial_distance": 5,
    "room_spatial_orientation/rotation": 6,
    "room_temporal_order/multi": 7,
}
