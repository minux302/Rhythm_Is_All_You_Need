import click
import numpy as np
import random
import tensorflow as tf
import time

import config
from model import Model
from loader import Chord2Id
from util import piano_roll_to_pretty_midi


def generate_from_random(class_num, seq_len=50):
  generate = np.random.randint(0, class_num - 1, seq_len).tolist()
  return generate
    

def create_sample(ckpt_path):

  note_series  = generate_from_random(config.CLASS_NUM, seq_len=config.SEQ_LEN) 
  chord_series = generate_from_random(config.CHORD_CLASS_NUM, seq_len=config.SEQ_LEN - 1) 

  # chord_list = ['C', 'Am', 'Dm', 'G7', 'Em', 'Am', 'Dm', 'G7', 'C']
  chord_list = []
  for i in range(5):
    chord_list += ['B', 'Abm', 'Dbm', 'Gb7', 'Ebm', 'Abm', 'Dbm', 'Gb7']
  chord2id = Chord2Id(demo=True)
  chord_to_note = chord2id.get_chord_to_note_dict()
  append_chord_id_series = []
  append_chord_series = []
  len_per_chord = 4
  for chord in chord_list:
    for _ in range(len_per_chord):
      append_chord_series.append(chord)
      append_chord_id_series.append(chord2id.get_id(chord))

  generate_length = len(append_chord_series)
  chord_series += append_chord_id_series

  with tf.Graph().as_default():

    # model build
    model = Model(seq_len=config.SEQ_LEN,
                  class_num=config.CLASS_NUM,
                  chord_class_num=config.CHORD_CLASS_NUM)
    input_note_pl, input_chord_pl, target_pl = model.placeholders()
    is_training_pl = tf.constant(False, name="is_training")
    pred           = model.infer(input_note_pl, input_chord_pl, is_training_pl)

    saver = tf.train.Saver()
    config_gpu = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

    with tf.Session(config=config_gpu) as sess:

      saver.restore(sess, ckpt_path)

      for i in range(generate_length):
        note_input  = np.array([note_series[i:i+config.SEQ_LEN]])
        chord_input = np.array([chord_series[i:i+config.SEQ_LEN]])
        
        feed_dict = {
          input_note_pl : note_input,
          input_chord_pl: chord_input,
          target_pl     : [1],
        }
        output = sess.run([pred], feed_dict)
        output = np.array(output).flatten()
        output_argsort = np.argsort(output)

        # idx = np.random.randint(2)
        # pred_note = output_argsort[-idx]

        if random.uniform(0,1) > 0.3:
          pred_note = output_argsort[-1]
        else:
          pred_note = 128

        pred_note = output_argsort[-1]
        note_series.append(pred_note)

    print(note_series)

    file_name = "result.mid"
    start_index=config.SEQ_LEN
    fs=config.FRAME_PER_SECOND

    array_piano_roll = np.zeros((128, generate_length + 1), dtype=np.int16)
    for index, note in enumerate(note_series[start_index:]):
      chord_idx = index - config.SEQ_LEN
      if note == 128:
        for chord_note in chord_to_note[append_chord_series[chord_idx]]:
          array_piano_roll[chord_note, index] = 1
        continue
      else:
        for chord_note in chord_to_note[append_chord_series[chord_idx]]:
          array_piano_roll[chord_note, index] = 1
        array_piano_roll[note, index] = 1

    generate_to_midi = piano_roll_to_pretty_midi(array_piano_roll, fs=fs)
    for note in generate_to_midi.instruments[0].notes:
      note.velocity = 100
    generate_to_midi.write(file_name)

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
  "-c",
  "--ckpt_path",
  help="path to ckpt",
  required=True
)
def main(ckpt_path):
  create_sample(ckpt_path)


if __name__ == '__main__':
  main()
