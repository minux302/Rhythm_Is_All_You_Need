import os
import numpy as np
import pretty_midi
import pickle
import math
from pathlib import Path
from random import seed, shuffle, randint


seed(666)


def get_p_extension_list(folder, extension='mid'):
  p_folder         = Path(folder)
  p_extension_list = list(p_folder.glob('**/*.' + extension))

  return p_extension_list


class Chord2Id:

  def __init__(self, demo=False):
    self.key_list        = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    self.chord_type_list = ['', 'M7', 'm', 'm7', '7', 'o', 'ø']
    self.chord_list      = self._generate_chord_list()
    self.chord_class_num = len(self.chord_list)
    if demo:
      self.chord_to_note = self._generate_chord_note()

  def _generate_chord_list(self):
    chord_list = []

    for key in self.key_list:
      for chord_type in self.chord_type_list:
        chord = key + chord_type
        chord_list.append(chord)
    return chord_list

  def _generate_chord_note(self, octave=4):
    base_notes = [np.array([0, 4, 7])     + 12*octave,
                  np.array([0, 4, 7, 11]) + 12*octave,
                  np.array([0, 3, 7])     + 12*octave,
                  np.array([0, 3, 7, 10]) + 12*octave,
                  np.array([0, 4, 7, 10]) + 12*octave,
                  np.array([0, 3, 7, 10]) + 12*octave,
                  np.array([0, 3, 7, 11]) + 12*octave]

    chord_to_note = {}
    for i, key in enumerate(self.key_list):
      for j, chord in enumerate(self.chord_type_list):
        chord_notes = base_notes[j] + i
        chord_to_note[key+chord] = chord_notes
    return chord_to_note

  def get_id(self, chord):
    return self.chord_list.index(chord)

  def get_chord_to_note_dict(self):
    return self.chord_to_note


class MelodyandChordLoader:

  def __init__(self,
               p_midi_list,
               seq_len,
               class_num,
               chord_class_num,
               batch_song_size=16,
               batch_size=256,
               fs=30):

    assert len(p_midi_list) >= batch_song_size, "p_midi_lsit must be longer than batch_song_size"

    self.p_midi_list     = p_midi_list
    self.total_songs     = len(p_midi_list)
    self.seq_len         = seq_len
    self.class_num       = class_num
    self.rest_note_class = class_num - 1
    self.batch_song_size = batch_song_size
    self.batch_size      = batch_size
    self.fs              = fs

    self.batch_song_input_note   = np.empty((0, self.seq_len))
    self.batch_song_input_chord  = np.empty((0, self.seq_len))
    self.batch_song_target_note  = np.empty((0))
    self.batch_idx_list          = np.array([])

    self.chord2id        = Chord2Id()
    self.chord_class_num = chord_class_num

  def generate_batch_buffer(self, i, shuffle=True):

    start_idx = i * self.batch_song_size
    self.batch_song_input_note  = np.empty((0, self.seq_len))
    self.batch_song_input_chord = np.empty((0, self.seq_len))
    self.batch_song_target_note = np.empty((0))

    note_data_dict  = self._generate_note_data_dict(start_idx)
    chord_data_dict = self._generate_chord_data_dict(start_idx)
    note_data_dict, chord_data_dict = self._align_dicts(note_data_dict, chord_data_dict)

    for key in list(note_data_dict.keys()):
      input_note_list, input_chord_list, target_note_list \
        = self._generate_input_and_target(note_data_dict[key], chord_data_dict[key])
      self.batch_song_input_note  = np.append(self.batch_song_input_note,  input_note_list,  axis=0)
      self.batch_song_input_chord = np.append(self.batch_song_input_chord, input_chord_list, axis=0)
      self.batch_song_target_note = np.append(self.batch_song_target_note, target_note_list, axis=0)

    self.batch_idx_list = np.arange(start=0, stop=len(self.batch_song_input_note))
    if shuffle:
      np.random.shuffle(self.batch_idx_list)

    # return 
    return chord_data_dict 

  def get_batch(self, i):
    idx = i * self.batch_size
    current_idx  = self.batch_idx_list[idx: idx + self.batch_size]
    batch_input_note   = self.batch_song_input_note[current_idx]
    batch_input_chord  = self.batch_song_input_chord[current_idx]
    batch_target_note  = self.batch_song_target_note[current_idx]

    return batch_input_note, batch_input_chord, batch_target_note

  def get_batch_song_num(self):
    return math.ceil(self.total_songs / self.batch_song_size)

  def get_batch_num(self):
    return math.ceil(len(self.batch_song_input_note) / self.batch_size)

  def get_total_songs(self):
    return self.total_songs

  def shuffle_midi_list(self):
    shuffle(self.p_midi_list)
    return

  def _generate_input_and_target(self, note_data, chord_data):
    start, end = 0, len(note_data) - 1
    input_note_list  = []
    input_chord_list = []
    target_note_list = []

    for idx in range(start, end):
      input_note_sample  = []
      input_chord_sample  = []
      start_iterate = 0

      if idx < self.seq_len:
        start_iterate = self.seq_len - idx - 1
        for i in range(start_iterate):
          # input_note_sample.append(self.rest_note_class) # Todo Rethink
          input_note_sample.append(randint(0, self.rest_note_class))
          if i < start_iterate - 1:
            # input_chord_sample.append('tmp')  # Todo Rethink
            input_chord_sample.append(randint(0, self.chord_class_num - 1))
          else:
            chord_id = self.chord2id.get_id(chord_data[0])
            input_chord_sample.append(chord_id)

      for i in range(start_iterate, self.seq_len):
        current_idx = idx - (self.seq_len - i - 1)
        input_note_sample.append(note_data[current_idx])
        chord_id = self.chord2id.get_id(chord_data[current_idx + 1])
        input_chord_sample.append(chord_id)

      target_sample = note_data[idx + 1]

      input_note_list.append(input_note_sample)
      input_chord_list.append(input_chord_sample)
      target_note_list.append(target_sample)

    return np.array(input_note_list), np.array(input_chord_list), np.array(target_note_list)

  def _preprocess_pianoroll_dict(self, pianoroll_dict):
    note_data_dict = {}  # key: file_num, value: note series

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
        continue

    note_data_dict = self._preprocess_pianoroll_dict(pianoroll_dict)

    return note_data_dict

  def _preprocess_chord_symbols_dict(self, chord_symbols_dict):

    chord_data_dict = {}  # key: file_num, value: chord series

    for name_num in chord_symbols_dict.keys():

      chord_symbols = chord_symbols_dict[name_num]  # (chord_num, [chord, start, end])

      # preprocess for chord notation
      chord_list = []
      for chord_info in chord_symbols:

        if '|' in chord_info[0]: # get upper code of oncode
          chord = chord_info[0].split('|')[0]
          chord = chord.replace(" ", "")
        elif ' ' in chord_info[0]: # rm sus4
          chord = chord_info[0].split(' ')[0]
        else:
          chord = chord_info[0]

        # rm tention (9, 11, 13) notation, leaves 7th exept for ø
        chord = ''.join(c for c in chord if not(c.isdigit() and c != '7'))
        chord = chord.replace('ø7', 'ø')
        chord = chord.replace('maj7', 'maj')
        chord = chord.replace('maj', 'M7')

        # align chord notation
        chord = chord[0].upper() + chord[1:]

        chord_list.append(chord)

      # convert chord_symbols to chord series
      counter = 0
      chord_series_list = []
      for i in range(len(chord_symbols)):
        end_time_sec = chord_symbols[i][2]
        while (counter < int(end_time_sec * self.fs)):
          chord_series_list.append(chord_list[i])
          counter += 1

      chord_data_dict[name_num] = chord_series_list

    return chord_data_dict

  def _generate_chord_data_dict(self, start_idx):

    chord_symbols_dict = {}  # key: file_num, value: chord_symbols
    idx_list = range(start_idx, min(start_idx + self.batch_song_size, len(self.p_midi_list)))

    for i in idx_list:
      p_midi   = self.p_midi_list[i]
      name_num = int(p_midi.name.split('.')[0])  # ToDo: Rethink about data name
      p_chord  = p_midi.parent / (str(name_num) + '.chord')
      try:
        with open(str(p_chord), "rb") as f:
          chord_symbols = pickle.load(f)  # (chord_num, [chord, start, end])
        chord_symbols_dict[name_num] = chord_symbols
      except Exception as e:
        print(e)
        print("broken file : {}".format(str(p_midi)))
        continue

    chord_data_dict = self._preprocess_chord_symbols_dict(chord_symbols_dict)

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

      if note_data_len == 0 or chord_data_len == 0:
        del note_data_dict[key]
        del chord_data_dict[key]
        continue
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
  chord_class_num = 84
  batch_song_size = 1 
  batch_size = 3
  fs = 2  # frame_per_second

  p_midi_list_train = get_p_extension_list(os.path.join(save_path, 'train'), 'mid')
  loader = MelodyandChordLoader(p_midi_list=p_midi_list_train,
                                seq_len=seq_len,
                                class_num=class_num,
                                chord_class_num=chord_class_num,
                                batch_song_size=batch_song_size,
                                batch_size=batch_size,
                                fs=fs)

  loader.shuffle_midi_list()
  batch_song_num = loader.get_batch_song_num()

  chord_dict = {}

  for i in range(0, batch_song_num):
    # print("{} ====================================================== ".format(i))
    loader.generate_batch_buffer(i)
    batch_num = loader.get_batch_num()

    # print(loader.batch_song_input_note.shape)
    # print(loader.batch_song_input_chord.shape)
    for batch_idx in range(0, batch_num):
      batch_input_note, batch_input_chord, batch_target_note = loader.get_batch(batch_idx) 
      print(batch_input_note)
      print(batch_input_chord)
      print(batch_target_note)



