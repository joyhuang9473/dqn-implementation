from PIL import Image
import numpy as np

def preprocess_observation(
    observation,
    rescale_width=84,
    rescale_height=110,
    clip_width=84,
    clip_height=84,
):
    img = Image.fromarray(observation)
    img = img.convert('L')
    img = img.resize((rescale_width, rescale_height))
    obs = np.array(img, dtype=np.uint8)
    obs = obs[-clip_height:, :clip_width]
    return obs

def normalized(states):
    states = states.astype(np.float32)
    states = states / 255.0

    return states
