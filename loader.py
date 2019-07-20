import os
import numpy as np
import pretty_midi
import pickle
import math
from pathlib import Path
from random import seed, shuffle


seed(666)


def get_p_extension_list(folder, extension='mid'):
  p_folder         = Path(folder)
  p_extension_list = list(p_folder.glob('**/*.' + extension))

  return p_extension_list


class MelodyandChordLoader:

  def __init__(self,
               p_midi_list,
               seq_len,
               class_num,
               batch_song_size=16,
               batch_size=256,
               fs=30):

    assert len(p_midi_list) >= batch_song_size, "p_midi_lsit must be longer than batch_song_size"

    self.p_midi_list       = p_midi_list
    self.total_songs       = len(p_midi_list)
    self.seq_len           = seq_len
    self.class_num         = class_num
    self.rest_note_class   = class_num - 1
    self.batch_song_size   = batch_song_size
    self.batch_size        = batch_size
    self.fs                = fs

    self.batch_song_input  = np.empty((0, self.seq_len))
    self.batch_song_target = np.empty((0))
    self.batch_idx_list    = np.array([])

  def generate_batch_buffer(self, i, shuffle=True):

    start_idx = i * self.batch_song_size
    self.batch_song_input  = np.empty((0, self.seq_len))
    self.batch_song_target = np.empty((0))

    note_data_dict  = self._generate_note_data_dict(start_idx)
    chord_data_dict = self._generate_chord_data_dict(start_idx)
    note_data_dict, chord_data_dict = self._align_dicts(note_data_dict, chord_data_dict)

    for key in list(note_data_dict.keys()):
      input_list, target_list = self._generate_input_and_target(note_data_dict[key])
      self.batch_song_input   = np.append(self.batch_song_input,  input_list,  axis=0)
      self.batch_song_target  = np.append(self.batch_song_target, target_list, axis=0)

    self.batch_idx_list = np.arange(start=0, stop=len(self.batch_song_input))
    if shuffle:
      np.random.shuffle(self.batch_idx_list)

    return

  def get_batch(self, i):
    idx = i * self.batch_size
    current_idx  = self.batch_idx_list[idx: idx + self.batch_size]
    batch_input  = self.batch_song_input[current_idx]
    batch_target = self.batch_song_target[current_idx]

    return batch_input, batch_target

  def get_batch_song_num(self):
    return math.ceil(self.total_songs / self.batch_song_size)

  def get_batch_num(self):
    return math.ceil(len(self.batch_song_input) / self.batch_size)

  def get_total_songs(self):
    return self.total_songs

  def shuffle_midi_list(self):
    shuffle(self.p_midi_list)
    return

  def _generate_input_and_target(self, data):
    start, end = 0, len(data)
    input_list, target_list = [], []

    for idx in range(start, end):
      input_sample, target_sample = [], []
      start_iterate = 0

      if idx < self.seq_len:
        start_iterate = self.seq_len - idx - 1
        for i in range(start_iterate):
          input_sample.append(self.rest_note_class)

      for i in range(start_iterate, self.seq_len):
        current_idx = idx - (self.seq_len - i - 1)
        input_sample.append(data[current_idx])

      if idx + 1 < end:
        target_sample = data[idx + 1]
      else:
        target_sample = self.rest_note_class

      input_list.append(input_sample)
      target_list.append(target_sample)

    return np.array(input_list), np.array(target_list)

  def _preprocess_pianoroll_dict(self, pianoroll_dict):
    note_data_dict = {}  

    for name_num in pianoroll_dict.keys():
      pianoroll      = pianoroll_dict[name_num]
      pianoroll_T    = pianoroll.T
      note_data = []

      # add top note idx
      for i in range(pianoroll_T.shape[0]):
        note = np.nonzero(pianoroll_T[i])[0]
        if len(note) == 0:
          note_data.append(self.rest_note_class) 
        else:
          note_data.append(max(note))

      note_data_dict[name_num] = note_data

    return note_data_dict

  def _generate_note_data_dict(self, start_idx):

    pianoroll_dict = {}  # key: file_num, value: pianoroll
    idx_list = range(start_idx, min(start_idx + self.batch_song_size, len(self.p_midi_list)))

    for i in idx_list:
      p_midi   = self.p_midi_list[i]
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

    note_data_dict = self._preprocess_pianoroll_dict(pianoroll_dict)

    return note_data_dict

  def _generate_chord_data_dict(self, start_idx):

    chord_data_dict = {}  # key: file_num, value: chords
    idx_list = range(start_idx, min(start_idx + self.batch_song_size, len(self.p_midi_list)))

    for i in idx_list:
      p_midi   = self.p_midi_list[i]
      name_num = int(p_midi.name.split('.')[0])  # ToDo: Rethink about data name
      p_chord  = p_midi.parent / (str(name_num) + '.chord')
      try:
        with open(str(p_chord), "rb") as f:
          chord_symbols = pickle.load(f)  # (chord_num, [chord, start, end])
      except Exception as e:
        print(e)
        print("broken file : {}".format(str(p_midi)))
        continue

      chord_list = []
      for chord_info in chord_symbols:
        # get upper code of oncode
        # ToDo: Refactoring here
        if '|' in chord_info[0]:
          chord = chord_info[0].split('|')[0]
        elif ' ' in chord_info[0]:
          chord = chord_info[0].split(' ')[0]
        else:
          chord = chord_info[0]
        chord_list.append(chord)

      counter = 0
      chord_series_list = []
      for i in range(len(chord_symbols)):
        end_time_sec = chord_symbols[i][2]
        while (counter < int(end_time_sec * self.fs)):
          chord_series_list.append(chord_list[i])
          counter += 1
      chord_data_dict[name_num] = chord_series_list

    return chord_data_dict

  def _align_dicts(self, note_data_dict, chord_data_dict):

    # get key that has .mid and .chord
    note_data_keys   = list(note_data_dict.keys())
    chord_data_keys  = list(chord_data_dict.keys())
    common_keys      = list(set(note_data_keys) & set(chord_data_keys))

    # rm abundant item(song)
    rm_list = list(set(note_data_keys)   - set(common_keys))
    for i in rm_list:
      del note_data_dict[i]
    rm_list = list(set(chord_data_keys)  - set(common_keys))
    for i in rm_list:
      del chord_data_dict[i]

    # align length
    for key in common_keys:
      note_data_len   = len(note_data_dict[key])
      chord_data_len  = len(chord_data_dict[key])
      if note_data_len >= chord_data_len:
        note_data_dict[key]   = note_data_dict[key][:chord_data_len]
      else:
        chord_data_dict[key] = chord_data_dict[key][:note_data_len]

    return note_data_dict, chord_data_dict


if __name__ == '__main__':

  save_path = './dataset_debug'
  p_midi_list = get_p_extension_list(save_path, 'mid')

  seq_len = 20
  class_num = 128 + 1
  batch_song_size = 3
  batch_size = 3
  fs = 2  # frame_per_second

  p_midi_list_train = get_p_extension_list(os.path.join(save_path, 'train'), 'mid')
  loader = MelodyandChordLoader(p_midi_list=p_midi_list_train,
                                seq_len=seq_len,
                                class_num=class_num,
                                batch_song_size=batch_song_size,
                                batch_size=batch_size,
                                fs=fs)

  loader.shuffle_midi_list()
  batch_song_num = loader.get_batch_song_num()

  for i in range(0, 3):
    print("{} ======================= ".format(i))
    loader.generate_batch_buffer(i)

    batch_num = loader.get_batch_num()
    for j in range(0, batch_num):
      batch_input, batch_target = loader.get_batch(j)

      # print(batch_target.shape)
      # print(batch_input)
 
      """
      for sample_i in range(batch_input.shape[0]):
        sample = []
        for sample_b_i in batch_input[sample_i]:
          sample.append(int(sample_b_i))
        print(sample)
      """

