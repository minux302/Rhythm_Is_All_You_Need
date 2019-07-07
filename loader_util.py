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


def generate_pianoroll_dict(p_midi_list, batch_song=16, start_idx=0, fs=30):
  assert len(p_midi_list) >= batch_song

  pianoroll_dict = {}  # key: file_num, value: pianoroll
  idx_list = range(start_idx, min(start_idx + batch_song, len(p_midi_list)))

  for i in idx_list:
    p_midi = p_midi_list[i]
    name_num = int(p_midi.name.split('.')[0])
    try:
      midi_pretty_format = pretty_midi.PrettyMIDI(str(p_midi))
      piano_midi = midi_pretty_format.instruments[0]  # Get the piano channels
      piano_roll = piano_midi.get_piano_roll(fs=fs)
      pianoroll_dict[name_num] = piano_roll
    except Exception as e:
      print(e)
      print("broken file : {}".format(str(p_midi)))
      pass

  return pianoroll_dict


def generate_notes_chord_dict(p_midi_list, batch_song=16, start_idx=0, fs=30):
  assert len(p_midi_list) >= batch_song

  notes_chord_dict = {}  # key: file_num, value: notes_chord
  idx_list = range(start_idx, min(start_idx + batch_song, len(p_midi_list)))

  for i in idx_list:
    p_midi = p_midi_list[i]
    name_num = int(p_midi.name.split('.')[0])

    p_chord = p_midi.parent / (str(name_num) + '.chord')
    with open(str(p_chord), "rb") as f:
      chord_symbols = pickle.load(f)  # (chord_num, [chord, start, end])

    chord_list = []
    for chord_info in chord_symbols:
      # get upper code of oncode
      # Todo: use regular expression
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


def align_dicts(pianoroll_dict, notes_chord_dict):

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
