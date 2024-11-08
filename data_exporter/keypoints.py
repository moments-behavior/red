def rat20():
    keypoint_names = ["Snout", "EarL", "EarR", "Neck", "SpineL", "TailBase", "ShoulderL", "ElbowL", "WristL", "HandL", "ShoulderR", "ElbowR", "WristR", "HandR", "KneeL", "AnkleL", "FootL", "KneeR", "AnkleR", "FootR"]
    skeleton = [{"keypointA":"Snout","keypointB":"EarL","length":0.0,"name":"Joint 1"},
            {"keypointA":"Snout","keypointB":"EarR","length":0.0,"name":"Joint 2"},
            {"keypointA":"EarL","keypointB":"Neck","length":0.0,"name":"Joint 3"},
            {"keypointA":"EarR","keypointB":"Neck","length":0.0,"name":"Joint 4"},
            {"keypointA":"Neck","keypointB":"SpineL","length":0.0,"name":"Joint 5"},
            {"keypointA":"SpineL","keypointB":"TailBase","length":0.0,"name":"Joint 6"},
            {"keypointA":"Neck","keypointB":"ShoulderL","length":0.0,"name":"Joint 7"},
            {"keypointA":"ShoulderL","keypointB":"ElbowL","length":0.0,"name":"Joint 8"},
            {"keypointA":"ElbowL","keypointB":"WristL","length":0.0,"name":"Joint 9"},
            {"keypointA":"WristL","keypointB":"HandL","length":0.0,"name":"Joint 10"},
            {"keypointA":"Neck","keypointB":"ShoulderR","length":0.0,"name":"Joint 11"},
            {"keypointA":"ShoulderR","keypointB":"ElbowR","length":0.0,"name":"Joint 12"},
            {"keypointA":"ElbowR","keypointB":"WristR","length":0.0,"name":"Joint 13"},
            {"keypointA":"WristR","keypointB":"HandR","length":0.0,"name":"Joint 14"},
            {"keypointA":"SpineL","keypointB":"KneeL","length":0.0,"name":"Joint 15"},
            {"keypointA":"KneeL","keypointB":"AnkleL","length":0.0,"name":"Joint 16"},
            {"keypointA":"AnkleL","keypointB":"FootL","length":0.0,"name":"Joint 17"},
            {"keypointA":"SpineL","keypointB":"KneeR","length":0.0,"name":"Joint 18"},
            {"keypointA":"KneeR","keypointB":"AnkleR","length":0.0,"name":"Joint 19"},
            {"keypointA":"AnkleR","keypointB":"FootR","length":0.0,"name":"Joint 20"},]
    return keypoint_names, skeleton, 20


def rat24():
    keypoint_names = ["Snout", "EarL", "EarR", "Neck", "SpineL", "TailBase", "ShoulderL", "ElbowL", "WristL", "HandL", "ShoulderR", "ElbowR", "WristR", "HandR", "KneeL", "AnkleL", "FootL", "KneeR", "AnkleR", "FootR", "TailTip", "TailMid", "Tail1Q", "Tail3Q"]
    skeleton = [{"keypointA":"Snout","keypointB":"EarL","length":0.0,"name":"Joint 1"},
            {"keypointA":"Snout","keypointB":"EarR","length":0.0,"name":"Joint 2"},
            {"keypointA":"EarL","keypointB":"Neck","length":0.0,"name":"Joint 3"},
            {"keypointA":"EarR","keypointB":"Neck","length":0.0,"name":"Joint 4"},
            {"keypointA":"Neck","keypointB":"SpineL","length":0.0,"name":"Joint 5"},
            {"keypointA":"SpineL","keypointB":"TailBase","length":0.0,"name":"Joint 6"},
            {"keypointA":"Neck","keypointB":"ShoulderL","length":0.0,"name":"Joint 7"},
            {"keypointA":"ShoulderL","keypointB":"ElbowL","length":0.0,"name":"Joint 8"},
            {"keypointA":"ElbowL","keypointB":"WristL","length":0.0,"name":"Joint 9"},
            {"keypointA":"WristL","keypointB":"HandL","length":0.0,"name":"Joint 10"},
            {"keypointA":"Neck","keypointB":"ShoulderR","length":0.0,"name":"Joint 11"},
            {"keypointA":"ShoulderR","keypointB":"ElbowR","length":0.0,"name":"Joint 12"},
            {"keypointA":"ElbowR","keypointB":"WristR","length":0.0,"name":"Joint 13"},
            {"keypointA":"WristR","keypointB":"HandR","length":0.0,"name":"Joint 14"},
            {"keypointA":"SpineL","keypointB":"KneeL","length":0.0,"name":"Joint 15"},
            {"keypointA":"KneeL","keypointB":"AnkleL","length":0.0,"name":"Joint 16"},
            {"keypointA":"AnkleL","keypointB":"FootL","length":0.0,"name":"Joint 17"},
            {"keypointA":"SpineL","keypointB":"KneeR","length":0.0,"name":"Joint 18"},
            {"keypointA":"KneeR","keypointB":"AnkleR","length":0.0,"name":"Joint 19"},
            {"keypointA":"AnkleR","keypointB":"FootR","length":0.0,"name":"Joint 20"},
            {"keypointA":"TailBase","keypointB":"Tail1Q","length":0.0,"name":"Joint 21"},
            {"keypointA":"Tail1Q","keypointB":"TailMid","length":0.0,"name":"Joint 22"},
            {"keypointA":"TailMid","keypointB":"Tail3Q","length":0.0,"name":"Joint 23"},
            {"keypointA":"Tail3Q","keypointB":"TailTip","length":0.0,"name":"Joint 24"}]
    return keypoint_names, skeleton, 24

def rat6():
    keypoint_names = ["Snout", "EarL", "EarR", "Neck", "SpineL", "TailBase"]
    skeleton = [{"keypointA":"Snout","keypointB":"EarL","length":0.0,"name":"Joint 1"},
            {"keypointA":"Snout","keypointB":"EarR","length":0.0,"name":"Joint 2"},
            {"keypointA":"EarL","keypointB":"Neck","length":0.0,"name":"Joint 3"},
            {"keypointA":"EarR","keypointB":"Neck","length":0.0,"name":"Joint 4"},
            {"keypointA":"Neck","keypointB":"SpineL","length":0.0,"name":"Joint 5"},
            {"keypointA":"SpineL","keypointB":"TailBase","length":0.0,"name":"Joint 6"}]
    return keypoint_names, skeleton, 6

def rat4():
    keypoint_names = ["Snout", "EarL", "EarR", "Tail"]
    skeleton = [{"keypointA":"Snout","keypointB":"EarL","length":0.0,"name":"Joint 1"},
            {"keypointA":"Snout","keypointB":"EarR","length":0.0,"name":"Joint 2"},
            {"keypointA":"EarL","keypointB":"Tail","length":0.0,"name":"Joint 3"},
            {"keypointA":"EarR","keypointB":"Tail","length":0.0,"name":"Joint 4"}]
    return keypoint_names, skeleton, 4


skeleton_selector = {
    "rat4": rat4,
    "rat6": rat6,
    "rat20": rat20,
    "rat24": rat24
}