import numpy as np
import pretty_midi
import pickle
from pathlib import Path
from random import seed, shuffle


def get_p_extension_list(folder, extension='xml', seed_int=666):
  p_folder = Path(folder)
  p_extension_list = list(p_folder.glob('**/*.' + extension))

  seed(seed_int)
  shuffle(p_extension_list)

  return p_extension_list


class MelodyandChordLoader:

  def __init__(self,
               p_midi_list,
               batch_song=16,
               batch_size=256,
               fs=30,
               seq_len=50):

    assert len(p_midi_list) >= batch_song, "p_midi_lsit must be longer than batch_song"

    self.p_midi_list = p_midi_list
    self.batch_song = batch_song
    self.batch_size = batch_size
    self.fs = fs
    self.seq_len = seq_len

    self.batch_song_input = np.empty((0, self.seq_len))
    self.batch_song_target = np.empty((0, 1))
    self.batch_idx_list = np.array([])

  def generate_batch_buffer(self, start_idx, shuffle=True):

    self.batch_song_input = np.empty((0, self.seq_len))
    self.batch_song_target = np.empty((0, 1))

    pianoroll_dict = self.generate_pianoroll_dict(start_idx=start_idx)
    time_note_dict = self.generate_time_note_dict(pianoroll_dict)

    for key in list(time_note_dict.keys()):
      input_list, target_list = self.generate_input_and_target(time_note_dict[key])
      self.batch_song_input = np.append(self.batch_song_input, input_list, axis=0)
      self.batch_song_target = np.append(self.batch_song_target, target_list, axis=0)

    self.batch_idx_list = np.arange(start=0, stop=len(self.batch_song_input))
    if shuffle:
      np.random.shuffle(self.batch_idx_list)

    return

  def get_batch(self, i):
    current_idx = self.batch_idx_list[i: i + self.batch_size]
    batch_input = self.batch_song_input[current_idx]
    batch_target = self.batch_song_target[current_idx]

    return batch_input, batch_target

  def get_batch_song_len(self):
    return len(self.batch_song_input)

  def generate_pianoroll_dict(self, start_idx):

    pianoroll_dict = {}  # key: file_num, value: pianoroll
    idx_list = range(start_idx, min(start_idx + self.batch_song, len(self.p_midi_list)))

    for i in idx_list:
      p_midi = self.p_midi_list[i]
      name_num = int(p_midi.name.split('.')[0])  # ToDo: Rethink about data name
      try:
        midi_pretty_format = pretty_midi.PrettyMIDI(str(p_midi))
        piano_midi = midi_pretty_format.instruments[0]  # Get the piano channels
        piano_roll = piano_midi.get_piano_roll(fs=self.fs)
        pianoroll_dict[name_num] = piano_roll
      except Exception as e:
        print(e)
        print("broken file : {}".format(str(p_midi)))
        pass

    return pianoroll_dict

  def generate_notes_chord_dict(self, start_idx):

    notes_chord_dict = {}  # key: file_num, value: notes_chord
    idx_list = range(start_idx, min(start_idx + self.batch_song, len(self.p_midi_list)))

    for i in idx_list:
      p_midi = self.p_midi_list[i]
      name_num = int(p_midi.name.split('.')[0])  # ToDo: Rethink about data name
      p_chord = p_midi.parent / (str(name_num) + '.chord')
      with open(str(p_chord), "rb") as f:
        chord_symbols = pickle.load(f)  # (chord_num, [chord, start, end])

      chord_list = []
      for chord_info in chord_symbols:
        # get upper code of oncode
        # ToDo: use regular expression
        if '|' in chord_info[0]:
          chord = chord_info[0].split('|')[0]
        elif ' ' in chord_info[0]:
          chord = chord_info[0].split(' ')[0]
        else:
          chord = chord_info[0]
        chord_list.append(chord)

      counter = 0
      notes_chord = []
      for i in range(len(chord_symbols)):
        end_time_sec = chord_symbols[i][2]
        while (counter < int(end_time_sec * fs)):
          notes_chord.append(chord_list[i])
          counter += 1
      notes_chord_dict[name_num] = notes_chord

    return notes_chord_dict

  def generate_time_note_dict(self, pianoroll_dict):
    time_note_dict = {}  # key: file_num, value: time_note_dict

    for name_num in pianoroll_dict.keys():
      pianoroll = pianoroll_dict[name_num]
      pianoroll_T = pianoroll.T
      time_note_list = []

      # add top note idx
      for i in range(pianoroll_T.shape[0]):
        note = np.nonzero(pianoroll_T[i])[0]
        if len(note) == 0:
          time_note_list.append('e')
        else:
          time_note_list.append(max(note))

      time_note_dict[name_num] = time_note_list

    return time_note_dict

  def generate_input_and_target(self, time_note):
    start, end = 0, len(time_note)
    input_list, target_list = [], []

    for idx in range(start, end):
      input_sample, target_sample = [], []
      start_iterate = 0

      if idx < self.seq_len:
        start_iterate = self.seq_len - idx - 1
        for i in range(start_iterate):
          input_sample.append('e')

      for i in range(start_iterate, self.seq_len):
        current_idx = idx - (self.seq_len - i - 1)
        input_sample.append(time_note[current_idx])

      if idx + 1 < end:
        target_sample.append(time_note[idx + 1])
      else:
        target_sample.append('e')

      input_list.append(input_sample)
      target_list.append(target_sample)

    return np.array(input_list), np.array(target_list)

  def align_dicts(self, pianoroll_dict, notes_chord_dict):

    # get key that has .mid and .chord
    pianoroll_keys = list(pianoroll_dict.keys())
    notes_chord_keys = list(notes_chord_dict.keys())
    common_keys = list(set(pianoroll_keys) & set(notes_chord_keys))

    # rm abundant item
    rm_list = list(set(pianoroll_keys) - set(common_keys))
    for i in rm_list:
      del pianoroll_dict[i]
    rm_list = list(set(notes_chord_keys) - set(common_keys))
    for i in rm_list:
      del notes_chord_dict[i]

    # align length of value
    for key in common_keys:
      pianoroll_dict_len = pianoroll_dict[key].shape[1]
      notes_chord_dict_len = len(notes_chord_dict[key])
      if pianoroll_dict_len >= notes_chord_dict_len:
        # Todo: Refine
        pianoroll_dict[key] = pianoroll_dict[key].T[:notes_chord_dict_len]
        pianoroll_dict[key] = pianoroll_dict[key].T
      else:
        notes_chord_dict[key] = notes_chord_dict[key][:pianoroll_dict_len]

    return pianoroll_dict, notes_chord_dict

