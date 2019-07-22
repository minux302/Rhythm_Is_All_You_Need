import click
import numpy as np
import random
import tensorflow as tf

import config
from model import Model
from loader import Chord2Id
from util import piano_roll_to_pretty_midi

# Bags Groove
chord_list_parts = ['F7', 'Bb7', 'F7', 'F7',
                    'Bb7', 'Bb7', 'F7', 'D7',
                    'Gm7', 'C7', 'F7', 'C7']
# chord_list_parts = ['A7', 'D7', 'A7', 'A7',
#                     'D7', 'D7', 'A7', 'Gb7',
#                    'Bm7', 'E7', 'A7', 'E7']
chord_list = []
for i in range(3):
  chord_list += chord_list_parts
fs = 4

def generate_from_random(class_num, seq_len=50):
  generate = np.random.randint(0, class_num - 1, seq_len).tolist()
  return generate
    

def create_sample(ckpt_path):

  note_series  = generate_from_random(config.CLASS_NUM, seq_len=config.SEQ_LEN) 
  chord_id_series = generate_from_random(config.CHORD_CLASS_NUM, seq_len=config.SEQ_LEN - 1) 

  chord2id = Chord2Id(demo=True)
  chord_to_note = chord2id.get_chord_to_note_dict()

  append_chord_id_series = []
  chord_series = []
  for chord in chord_list:
    for _ in range(fs):
      chord_series.append(chord)
      append_chord_id_series.append(chord2id.get_id(chord))

  generate_length = len(chord_series)
  chord_id_series += append_chord_id_series

  with tf.Graph().as_default():

    model = Model(seq_len=config.SEQ_LEN,
                  class_num=config.CLASS_NUM,
                  chord_class_num=config.CHORD_CLASS_NUM)
    input_note_pl, input_chord_pl, target_pl = model.placeholders()
    is_training_pl = tf.constant(False, name="is_training")
    pred           = model.infer(input_note_pl, input_chord_pl, is_training_pl)

    saver = tf.train.Saver()
    with tf.Session() as sess:

      saver.restore(sess, ckpt_path)

      for i in range(generate_length):
        note_input  = np.array([note_series[i:i+config.SEQ_LEN]])
        chord_input = np.array([chord_id_series[i:i+config.SEQ_LEN]])
        
        feed_dict = {
          input_note_pl : note_input,
          input_chord_pl: chord_input,
          target_pl     : [0],
        }
        output = sess.run([pred], feed_dict)

        output = np.array(output).flatten()
        pred_note = np.argsort(output)[-1]
        note_series.append(pred_note)

    print(note_series)

    file_name = "sample.mid"
    array_piano_roll = np.zeros((config.CLASS_NUM-1, generate_length), dtype=np.int16)
    for idx, note in enumerate(note_series[config.SEQ_LEN:]):
      if note != config.CLASS_NUM - 1:  # rest note class
        array_piano_roll[note, idx] = 1  # add melody note
      for chord_note in chord_to_note[chord_series[idx]]:
        array_piano_roll[chord_note, idx] = 1  # add chord note

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
