import copy
import numpy as np

from utils.constants import *
from utils.common_utils import *
from torch.utils.data import Dataset


class DatasetSampler(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        x = np.load(self.file_list[idx])
        melody, mixture, music, track = preprocess(x)

        melody = melody.astype("float32")
        track = track.astype("float32")

        mixture = mixture.astype("float32")
        music = music.astype("float32")

        return melody, mixture, music, track


def preprocess(music: np.ndarray):
    music = np.transpose(music, (2, 0, 1))

    # get melody
    melody = music[MELODY_REL_NUM]

    # get mixtures
    music = music[MELODY_REL_NUM + 1 :]
    mixture = copy.deepcopy(music)
    mixture = np.clip(np.sum(mixture, axis=0), 0, 1)

    # get tracks
    track = np.clip(np.sum(music, axis=(1, 2)), 0, 1)

    return melody, mixture[np.newaxis], music, track
