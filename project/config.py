# file containing global variables and objects to be used in scripts

CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
DATA_DIRECTORY = "data"
MODEL_DIRECTORY = "models"
OUTPUT_DIRECTORY = "output"

allowed_predicates = [
    "below", # relative position
    "to the left of (wrt you)", # relative position
    "to the right of (wrt you)", # relative position
    "to the side of", # relative position
    "on", # relative position
    "covering", # occlusion
    "inside", # occlusion
    "in front of", # occlusion, depth
    "behind", # occlusion, depth
    "near", # depth
    "far from", # depth
]