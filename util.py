import os
import pretty_midi
import numpy as np
from pathlib import Path
from random import seed, shuffle
from loader import MelodyandChordLoader, get_p_extension_list


seed(666)


def split_dataset(dataset_path, train_valid_ratio = 0.2):
    p_folder = Path(dataset_path)
    p_midi_list = list(p_folder.glob('*.mid'))
    shuffle(p_midi_list)

    p_train = p_folder / 'train'
    p_valid = p_folder / 'valid'
    if not(p_train.exists()):
        p_train.mkdir()
    if not(p_valid.exists()):
        p_valid.mkdir()

    # data_num = len(p_midi_list)
    data_num = 100
    train_num = int(data_num * (1 - train_valid_ratio))
    for i in range(train_num):
        file_name = p_midi_list[i].name.split('.')[0]
        file_path = p_folder / (file_name + '.*')
        cmd = 'mv ' + str(file_path) + ' ' + str(p_train)
        os.system(cmd)
        print(cmd)

    for i in range(train_num, data_num):
        file_name = p_midi_list[i].name.split('.')[0]
        file_path = p_folder / (file_name + '.*')
        cmd = 'mv ' + str(file_path) + ' ' + str(p_valid)
        os.system(cmd)
        print(cmd)


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
  """
  from https://github.com/haryoa/note_music_generator
  """

  notes, frames = piano_roll.shape
  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(program=program)

  # pad 1 column of zeros so we can acknowledge inital and ending events
  piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

  # use changes in velocities to find note on / note off events
  velocity_changes = np.nonzero(np.diff(piano_roll).T)

  # keep track on velocities and note on times
  prev_velocities = np.zeros(notes, dtype=int)
  note_on_time = np.zeros(notes)

  for time, note in zip(*velocity_changes):
    # use time + 1 because of padding above
    velocity = piano_roll[note, time + 1]
    time = time / fs
    if velocity > 0:
      if prev_velocities[note] == 0:
        note_on_time[note] = time
        prev_velocities[note] = velocity
      else:
        pm_note = pretty_midi.Note(
            velocity=prev_velocities[note],
            pitch=note,
            start=note_on_time[note],
            end=time)
        instrument.notes.append(pm_note)
        prev_velocities[note] = 0
  pm.instruments.append(instrument)
  return pm


def piano_roll_adder_to_pretty_midi(pm,
                                    piano_roll,
                                    program=0,
                                    is_drum=False,
                                    fs=100,
                                    velocity_ratio=1.0):

  notes, frames = piano_roll.shape
  instrument = pretty_midi.Instrument(program=program, is_drum=is_drum)

  # pad 1 column of zeros so we can acknowledge inital and ending events
  piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

  # use changes in velocities to find note on / note off events
  velocity_changes = np.nonzero(np.diff(piano_roll).T)

  # keep track on velocities and note on times
  prev_velocities = np.zeros(notes, dtype=int)
  note_on_time = np.zeros(notes)

  for time, note in zip(*velocity_changes):
      # use time + 1 because of padding above
      velocity = piano_roll[note, time + 1] * velocity_ratio
      time = time / fs
      if velocity > 0:
          if prev_velocities[note] == 0:
              note_on_time[note] = time
              prev_velocities[note] = velocity
      else:
          pm_note = pretty_midi.Note(
              velocity=prev_velocities[note],
              pitch=note,
              start=note_on_time[note],
              end=time)
          instrument.notes.append(pm_note)
          prev_velocities[note] = 0
  pm.instruments.append(instrument)
  return pm


def get_sample_chord_progression():
  save_path = './dataset_debug'
  p_midi_list = get_p_extension_list(save_path, 'mid')

  seq_len = 20
  class_num = 128 + 1
  chord_class_num = 84
  batch_song_size = 1 
  batch_size = 3
  fs = 3  # frame_per_second

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
  chord_data_dict = loader.generate_batch_buffer(0) 
  print(chord_data_dict)


if __name__ == '__main__':
  # split dataset
  """
  dataset_path = 'dataset_debug_mini' 
  split_dataset(dataset_path)
  """

  # fix backing midi 
  midi_name      = 'autumn_leaves.mid'
  fix_midi_name  = midi_name.split('.')[0] + '_fix.mid'
  start_idx      = 240
  fs             = 120
  velocity_ratio = 1
  pm             = pretty_midi.PrettyMIDI(midi_name)
  fix_pm         = pretty_midi.PrettyMIDI()

  for instrument in pm.instruments:
    program    = instrument.program
    if program == 0:
      continue
    is_drum    = instrument.is_drum
    piano_roll = instrument.get_piano_roll()
    piano_roll = piano_roll[:, start_idx:]
    fix_pm = piano_roll_adder_to_pretty_midi(pm=fix_pm,
                                             piano_roll=piano_roll,
                                             program=program,
                                             is_drum=is_drum,
                                             fs=fs, 
                                             velocity_ratio=velocity_ratio)
    fix_pm.write(fix_midi_name)

  # get_sample_chord_progression()