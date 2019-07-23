import click
import numpy as np
import random
import tensorflow as tf

import config
from model import Model
from loader import Chord2Id

import os
import time
import pygame
import pygame.midi
import readchar
from multiprocessing import Process


def generate_from_random(class_num, seq_len=50):
  generate = np.random.randint(0, class_num - 1, seq_len).tolist()
  return generate


def backing_music(backing_midi_name):
  os.system('timidity ' + backing_midi_name)


def note_on(midiOutput, pred_note, volume, note_on_time=0.15):
  midiOutput.note_on(pred_note, volume)
  time.sleep(note_on_time)
  midiOutput.note_off(pred_note, volume)


def demo(ckpt_path):

  backing_midi_name = 'backing_fix.mid'
  chord_list = ['F7', 'Bb7', 'F7', 'F7',
                'Bb7', 'Bb7', 'F7', 'D7',
                'Gm7', 'C7', 'F7', 'C7']
  fs = 3
  volume = 300

  # setting for pygame
  pygame.init()
  pygame.midi.init()
  midiOutput = pygame.midi.Output(pygame.midi.get_default_output_id())
  midiOutput.set_instrument(0)  # 0: piano

  # subprossed for backing and sound melody note
  backing_process = Process(target=backing_music, args=(backing_midi_name,))
  note_on_process = Process(target=note_on,       args=())

  # init for input
  note_series     = generate_from_random(config.CLASS_NUM, seq_len=config.SEQ_LEN) 
  chord_id_series = generate_from_random(config.CHORD_CLASS_NUM, seq_len=config.SEQ_LEN - 1) 
  chord2id        = Chord2Id(demo=True)
  chord_to_note   = chord2id.get_chord_to_note_dict()

  append_chord_id_series = []
  for chord in chord_list:
    for _ in range(fs):
      append_chord_id_series.append(chord2id.get_id(chord))
  generate_length =  len(append_chord_id_series)
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

      # subprocess for backing and sound note
      backing_process.start()
      note_on_process.start()

      for i in range(generate_length):

        key = readchar.readchar()

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

        if pred_note == 128:
          pred_note = np.argsort(output)[-2]
        note_series.append(pred_note)
        print(pred_note)
        note_on(midiOutput, pred_note, volume)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
  "-c",
  "--ckpt_path",
  help="path to ckpt",
  required=True
)
def main(ckpt_path):
  demo(ckpt_path)


if __name__ == '__main__':
  main()
