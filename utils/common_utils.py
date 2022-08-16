import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

from typing import Union
from utils.constants import *
from collections import defaultdict

plt.style.use("seaborn")


def check_time_sign(pm: object, nom: int = 4, denom: int = 4) -> bool:
    """check the time signature of MIDIs with user's specification

    :param midi: pretty_midi object
    :type pm: object
    :param nom: the upper part of the signature, defaults to 4
    :type nom: int, optional
    :param denom: the bottom part of the signature, defaults to 4
    :type denom: int, optional
    :return: true or false
    :rtype: bool
    """

    time_sign_list = pm.time_signature_changes

    # empty or duplicate check
    if len(time_sign_list) != 1:
        return False

    # nom and denom check
    time_sign = time_sign_list[0]
    if time_sign.numerator != nom or time_sign.denominator != denom:
        return False

    return True


def unify_tracks(pm: object, melody_track: int = None) -> object:
    """map tracks into a standard set that we define

    :param midi: pretty_midi object
    :type midi: object
    :param melody_track: the number of melody track, defaults to None
    :type melody_track: int, optional
    :return: pretty_midi object
    :rtype: object
    """

    VALID_RANGE = np.concatenate([PIANO_RANGE, GUITAR_RANGE, BASS_RANGE, STRING_RANGE])

    new_inst = defaultdict()
    for i, track in enumerate(pm.instruments):
        # if a note length of inst is less that 50, we remove
        if len(track.notes) > 50:
            # map melody track to flute
            if track.program == melody_track:
                track.name = "melody"
                track.program = FLUTE
            else:
                if not track.is_drum:
                    # remove tracks not in VALID_RANGE
                    if track.program in VALID_RANGE:
                        # mapping tracks to a standard set
                        if track.program in PIANO_RANGE:
                            track.name = "piano"
                            track.program = PIANO

                        if track.program in GUITAR_RANGE:
                            track.name = "guitar"
                            track.program = GUITAR

                        if track.program in BASS_RANGE:
                            track.name = "bass"
                            track.program = BASS

                        if track.program in STRING_RANGE:
                            track.name = "string"
                            track.program = STRING
                    else:
                        continue
                else:
                    track.name = "drum"
                    track.program = 0  # set the program of drum as 0

            # remove duplicate tracks
            if track.name in new_inst:
                if len(new_inst[track.name].notes) < len(track.notes):
                    new_inst[track.name] = track
            else:
                new_inst[track.name] = track

    pm.instruments = list(new_inst.values())

    return pm


def get_pianoroll(pm: object, res: int = 4) -> Union[np.ndarray, float, float]:
    """convert a pretty_midi object to numpy array pianoroll

    :param pm: pretty_midi object
    :type pm: object
    :param res: the number of ticks in a quarter note, defaults to 4
    :type res: int, optional
    :return: [pianoroll, the onset of pm, the interval time of one tick]
    :rtype: Union[np.ndarray, float, float]
    """

    beat_start = pm.estimate_beat_start(candidates=30, tolerance=1e-3)
    beat_end = pm.get_end_time()

    beats = pm.get_beats(beat_start)
    event_time = get_event_time(beats, res)

    time_grid = np.arange(beat_start, beat_end + event_time, event_time)
    multitrack = np.zeros((time_grid.shape[0], GENERAL_NUM_PITCH, TOTAL_NUM_INST), dtype="bool")

    for track in pm.instruments:
        pianoroll = get_multitrack(track, beat_start, beat_end, event_time)

        if track.name == "melody":
            multitrack[:, :, MELODY_REL_NUM] = pianoroll[:, MELODY_ABS_NUM]

        if track.name == "piano":
            multitrack[:, :, PIANO_REL_NUM] = pianoroll[:, PIANO_ABS_NUM]

        if track.name == "guitar":
            multitrack[:, :, GUITAR_REL_NUM] = pianoroll[:, GUITAR_ABS_NUM]

        if track.name == "bass":
            multitrack[:, :, BASS_REL_NUM] = pianoroll[:, BASS_ABS_NUM]

        if track.name == "string":
            multitrack[:, :, STRING_REL_NUM] = pianoroll[:, STRING_ABS_NUM]

        if track.name == "drum":
            multitrack[:, -(DRUM_ABS_NUM[-1] + 1) :, DRUM_REL_NUM] = standarize_drum(pianoroll)

    return multitrack, beat_start, event_time


def get_event_time(beats: list, res: int) -> float:
    """get interval time of one tick from pretty_midi.get_beats()

    :param beats: beats from pretty_midi.get_beats()
    :type beats: list
    :param res: the number of ticks in a quarter note
    :type res: int
    :return: the interval time of one tick
    :rtype: float
    """

    quarter_time = beats[1] - beats[0]
    event_time = quarter_time / res

    return event_time


def get_multitrack(
    track: object, start_time: float, end_time: float, event_time: float
) -> np.ndarray:
    """convert a pretty_midi object to numpy array pianoroll

    :param track: pretty_midi object
    :type track: object
    :param start_time: the start time of a MIDI
    :type start_time: float
    :param end_time: the end time of a MIDI
    :type end_time: float
    :param event_time: the interval time of one tick
    :type event_time: float
    :return: numpy array pianoroll [hit, offset, play]
    :rtype: np.ndarray
    """

    time_grid = np.arange(start_time, end_time + event_time, event_time)
    pianoroll = np.zeros((time_grid.shape[0], 128), dtype="bool")

    for i, note in enumerate(track.notes):
        if note.start < time_grid[0] or note.end > time_grid[-1]:
            continue

        # offset
        note_start_diff = note.start - time_grid
        note_end_diff = note.end - time_grid

        # hit
        note_start_idx = np.argmin(np.abs(note_start_diff))
        note_end_idx = np.argmin(np.abs(note_end_diff))

        # margin
        if note_start_idx == note_end_idx:
            note_end_idx += 1

        # set offset
        # pianoroll[note_start_idx, note.pitch, 1] = note_start_diff[note_start_idx]
        # pianoroll[note_end_idx - 1, note.pitch, 1] = note_end_diff[note_end_idx]

        # set hit and velocity
        for idx in np.arange(note_start_idx, note_end_idx):
            pianoroll[idx, note.pitch] = 1  # hit
            # pianoroll[idx, note.pitch, 2] = note.velocity / 127  # velocity

    return pianoroll


def standarize_drum(pianoroll: np.ndarray) -> np.ndarray:
    """convert drum pianoroll to have only the standard set
    (0: kick, 1: snare, 2: closed hi-hat, 3: low tom,
     4: mid tom, 5: crash, 6: high tom, 7: ride)

    :param pianoroll: drum pianoroll (time, pitch > 8)
    :type pianoroll: np.ndarray
    :return: drum pianoroll (time, pitch) -> [:, 8]
    :rtype: np.ndarray
    """

    pianoroll[:, 35] = np.clip(np.sum(pianoroll[:, [35, 36]], axis=1), 0, 1)  # kick
    pianoroll[:, 38] = np.clip(np.sum(pianoroll[:, [37, 38, 39, 40]], axis=1), 0, 1)  # snare
    pianoroll[:, 42] = np.clip(np.sum(pianoroll[:, [42, 44]], axis=1), 0, 1)  # closed hi-hat
    pianoroll[:, 45] = np.clip(np.sum(pianoroll[:, [41, 43, 45]], axis=1), 0, 1)  # low tom
    pianoroll[:, 48] = np.clip(np.sum(pianoroll[:, [47, 48]], axis=1), 0, 1)  # mid tom
    pianoroll[:, 49] = np.clip(np.sum(pianoroll[:, [49, 55, 57]], axis=1), 0, 1)  # crash
    pianoroll[:, 51] = np.clip(np.sum(pianoroll[:, [51, 59]], axis=1), 0, 1)  # ride

    # 8 components
    return pianoroll[:, STANDARD_DRUM]


def plot_pianoroll(
    pianoroll: np.ndarray,
    save_path: str = "",
    res: int = 4,
    SIZE: list = [10, 10],
    CHAR_FONT_SIZE: int = 10,
    NUM_FONT_SIZE: int = 10,
    LABEL_PAD: int = 10,
) -> None:

    """plot multi-track pianoroll

    :param pianoroll: multi-track pianoroll
    :type pianoroll: np.ndarray
    :param save_path: save path of the figure, defaults to ""
    :type save_path: str, optional
    :param res: the number of ticks in a quarter note, defaults to 4
    :type res: int, optional
    :param SIZE: [width, height] of the figure, defaults to [10, 10]
    :type SIZE: list, optional
    :param CHAR_FONT_SIZE: font size related with characters, defaults to 10
    :type CHAR_FONT_SIZE: int, optional
    :param NUM_FONT_SIZE: font size related with numbers, defaults to 10
    :type NUM_FONT_SIZE: int, optional
    :param LABEL_PAD: padding size between plot and label, defaults to 10
    :type LABEL_PAD: int, optional
    """

    time, num_pitch, num_inst = pianoroll.shape
    total_pitch = num_pitch * num_inst

    pianoroll = np.reshape(pianoroll, (time, -1), order="F").T
    pallete = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:olive", "tab:red"]

    plt.figure(figsize=(SIZE[0], SIZE[1]))
    plt.imshow(pianoroll)
    plt.xticks(fontsize=NUM_FONT_SIZE)
    plt.yticks(fontsize=NUM_FONT_SIZE)
    plt.xlabel("Time", fontsize=CHAR_FONT_SIZE, labelpad=LABEL_PAD)
    plt.ylabel("Pitch", fontsize=CHAR_FONT_SIZE, labelpad=LABEL_PAD)
    plt.ylim([0, total_pitch])

    # drawing boundaries of bars
    for time_idx in range(res * 4, time, res * 4):
        plt.axvline(x=time_idx, linewidth="1", color="k", alpha=0.5)

    # coloring each instrument
    for pitch_idx in range(TOTAL_NUM_INST):
        plt.axhspan(
            num_pitch * pitch_idx,
            num_pitch * (pitch_idx + 1),
            facecolor=pallete[pitch_idx],
            alpha=0.2,
        )

    # save figure
    if len(save_path) > 0:
        plt.savefig(save_path, dpi=1000, bbox_inches="tight", pad_inches=0)

    plt.show()


def plot_two_pianoroll(
    pianoroll_1: np.ndarray,
    pianoroll_2: np.ndarray,
    save_path: str = "",
    res: int = 4,
    SIZE: list = [10, 10],
    CHAR_FONT_SIZE: int = 10,
    NUM_FONT_SIZE: int = 10,
    LABEL_PAD: int = 10,
) -> None:

    """plot multi-track pianoroll

    :param pianoroll_1: multi-track pianoroll
    :type pianoroll: np.ndarray
    :param pianoroll_2: multi-track pianoroll
    :type pianoroll: np.ndarray
    :param save_path: save path of the figure, defaults to ""
    :type save_path: str, optional
    :param res: the number of ticks in a quarter note, defaults to 4
    :type res: int, optional
    :param SIZE: [width, height] of the figure, defaults to [10, 10]
    :type SIZE: list, optional
    :param CHAR_FONT_SIZE: font size related with characters, defaults to 10
    :type CHAR_FONT_SIZE: int, optional
    :param NUM_FONT_SIZE: font size related with numbers, defaults to 10
    :type NUM_FONT_SIZE: int, optional
    :param LABEL_PAD: padding size between plot and label, defaults to 10
    :type LABEL_PAD: int, optional
    """

    time, num_pitch, num_inst = pianoroll_1.shape
    total_pitch = num_pitch * num_inst

    pianoroll_1 = np.reshape(pianoroll_1, (time, -1), order="F")
    pianoroll_2 = np.reshape(pianoroll_2, (time, -1), order="F")
    pianoroll_set = [pianoroll_1, pianoroll_2]
    title_set = ["Music from Mixture", "Original Music"]
    pallete = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:olive", "tab:red"]

    plt.figure(figsize=(SIZE[0], SIZE[1]))

    for i, pianoroll in enumerate(pianoroll_set):
        plt.subplot(2, 1, i + 1)
        plt.imshow(pianoroll)
        plt.xticks(fontsize=NUM_FONT_SIZE)
        plt.yticks(fontsize=NUM_FONT_SIZE)
        plt.xlabel("Pitch", fontsize=CHAR_FONT_SIZE, labelpad=LABEL_PAD)
        plt.ylabel("Time", fontsize=CHAR_FONT_SIZE, labelpad=LABEL_PAD)
        plt.title(title_set[i], fontsize=CHAR_FONT_SIZE, pad=LABEL_PAD + 2)
        plt.xlim([0, total_pitch])
        plt.xticks([])

        # drawing boundaries of bars
        for time_idx in range(res * 4, time, res * 4):
            plt.axhline(y=time_idx, linewidth="1", color="k", alpha=0.5)

        # coloring each instrument
        for pitch_idx in range(TOTAL_NUM_INST):
            plt.axvspan(
                num_pitch * pitch_idx,
                num_pitch * (pitch_idx + 1),
                facecolor=pallete[pitch_idx],
                alpha=0.2,
            )

    plt.tight_layout(h_pad=-20)

    # save figure
    if len(save_path) > 0:
        plt.savefig(save_path, dpi=1000, bbox_inches="tight", pad_inches=0)

    plt.close()
    # plt.show()


def play_pianoroll(pianoroll: np.ndarray, event_time: float) -> object:
    """convert numpy array pianoroll to pretty_midi to play music

    :param pianoroll: multi-track pianoroll
    :type pianoroll: np.ndarray
    :param event_time: the interval time of one tick
    :type event_time: float
    :return: pretty_midi object
    :rtype: object
    """

    time, pitch, inst = pianoroll.shape
    velocity = 80

    # initialize
    pm = pretty_midi.PrettyMIDI()

    pm.instruments.append(pretty_midi.Instrument(program=FLUTE, name="melody"))
    pm.instruments.append(pretty_midi.Instrument(program=PIANO, name="piano"))
    pm.instruments.append(pretty_midi.Instrument(program=GUITAR, name="guitar"))
    pm.instruments.append(pretty_midi.Instrument(program=BASS, name="bass"))
    pm.instruments.append(pretty_midi.Instrument(program=STRING, name="string"))
    pm.instruments.append(pretty_midi.Instrument(program=0, is_drum=True, name="drum"))

    if inst == 6:
        range_inst = range(inst)
        pianoroll[:, :-9, 5] = 0  # drum clean
    else:
        range_inst = range(1, inst + 1)
        pianoroll[:, :-9, 4] = 0  # drum clean

    # according to pitch
    for inst_idx in range_inst:
        for pitch_idx in range(pitch):
            if inst == 6:
                piano_note = pianoroll[:, pitch_idx, inst_idx]
            else:
                piano_note = pianoroll[:, pitch_idx, inst_idx - 1]

            time_idx_set = np.where(piano_note == 1)[0]

            # emply pitch check
            if time_idx_set.shape[0] == 0:
                continue

            time_diff = np.where(time_idx_set[1:] - time_idx_set[:-1] != 1)[0]
            time_diff = np.concatenate([time_diff, [-1]])
            start_idx = time_idx_set[0]

            for i, time_idx in enumerate(time_diff):
                # until the end
                if time_idx == -1:
                    end_idx = time_idx_set[-1] + 1
                else:
                    end_idx = time_idx_set[time_idx] + 1

                # time_grid + perturb
                start_time = start_idx * event_time
                end_time = end_idx * event_time

                if start_time < 0:
                    start_time = 0

                # assign pitch to each instrument
                if inst_idx == MELODY_REL_NUM:
                    pitch_diff = MELODY_ABS_NUM[0]
                    note = pretty_midi.Note(velocity, pitch_idx + pitch_diff, start_time, end_time)
                    pm.instruments[MELODY_REL_NUM].notes.append(note)

                if inst_idx == PIANO_REL_NUM:
                    pitch_diff = PIANO_ABS_NUM[0]
                    note = pretty_midi.Note(velocity, pitch_idx + pitch_diff, start_time, end_time)
                    pm.instruments[PIANO_REL_NUM].notes.append(note)

                if inst_idx == GUITAR_REL_NUM:
                    pitch_diff = GUITAR_ABS_NUM[0]
                    note = pretty_midi.Note(velocity, pitch_idx + pitch_diff, start_time, end_time)
                    pm.instruments[GUITAR_REL_NUM].notes.append(note)

                if inst_idx == BASS_REL_NUM:
                    pitch_diff = BASS_ABS_NUM[0]
                    note = pretty_midi.Note(velocity, pitch_idx + pitch_diff, start_time, end_time)
                    pm.instruments[BASS_REL_NUM].notes.append(note)

                if inst_idx == STRING_REL_NUM:
                    pitch_diff = STRING_ABS_NUM[0]
                    note = pretty_midi.Note(
                        velocity - 30, pitch_idx + pitch_diff, start_time, end_time
                    )
                    pm.instruments[STRING_REL_NUM].notes.append(note)

                if inst_idx == DRUM_REL_NUM:
                    pitch_diff = -63
                    note = pretty_midi.Note(
                        velocity, STANDARD_DRUM[pitch_idx + pitch_diff], start_time, end_time
                    )
                    pm.instruments[DRUM_REL_NUM].notes.append(note)

                start_idx = time_idx_set[time_idx + 1]

    return pm


def get_window(pianoroll: np.ndarray, res: int = 4, bar: int = 4) -> np.ndarray:
    """segment a pianoroll with a striding window

    :param pianoroll: multi-track pianoroll
    :type pianoroll: np.ndarray
    :param res: the number of ticks in a quarter note, defaults to 4
    :type res: int, optional
    :param bar: the number of bar in a window, defaults to 4
    :type bar: int, optional
    :return: multi-track pianoroll
    :rtype: np.ndarray
    """

    window_size = res * bar * 4

    total_phrase = []
    for idx in range(0, pianoroll.shape[0], res * 4):
        phrase = pianoroll[idx : idx + window_size]

        if phrase.shape[0] == window_size:
            total_phrase.append(phrase[np.newaxis])

    return np.vstack(total_phrase)


def check_empty_bar(pianoroll: np.ndarray, thres: float, res: int = 4) -> bool:
    """return True if the hit ratio is below the specified threshold

    :param pianoroll: multi-track pianoroll
    :type pianoroll: np.ndarray
    :param thres: a ratio to discard
    :type thres: float
    :param res: the number of ticks in a quarter note, defaults to 4
    :type res: int, optional
    :return: if below the threshold, return True
    :rtype: bool
    """

    num_note_in_bar = res * 4
    total_num_in_bar = num_note_in_bar * pianoroll.shape[1] * pianoroll.shape[2]

    for idx in range(0, pianoroll.shape[0], num_note_in_bar):
        num_hit_in_bar = np.sum(pianoroll[idx : idx + num_note_in_bar])

        # empty check
        if (num_hit_in_bar / total_num_in_bar) < thres:
            return True

    return False


def sort_pm(pm: object) -> object:
    """sort notes in each instrument respect to note onset

    :param pm: pretty_midi object
    :type pm: object
    :return: pretty_midi object
    :rtype: object
    """

    for i, inst in enumerate(pm.instruments):
        start_list = list(map(lambda x: x.start, inst.notes))

        if len(start_list) > 0:
            sorted_start_idx = np.argsort(start_list)
            pm.instruments[i].notes = [inst.notes[j] for j in sorted_start_idx]

    return pm
