import click
import numpy as np
import random
from random import shuffle
import tensorflow as tf

import config
from model import Model
from loader import Chord2Id

import os
import time
import pretty_midi
import pygame
import pygame.midi
import readchar
from multiprocessing import Process


def generate_from_random(class_num, seq_len=50):
  generate = np.random.randint(0, class_num - 1, seq_len).tolist()
  return generate


def note_on(midiOutput, pred_note, volume, note_on_time=0.12):
  midiOutput.note_on(pred_note, volume)
  time.sleep(note_on_time)
  midiOutput.note_off(pred_note, volume)


def backing_music(backing_midi_name):
  # os.system('timidity ' + backing_midi_name + ' --adjust-tempo=90')
  os.system('timidity ' + backing_midi_name)


def demo(ckpt_path):

  backing_midi_name = 'backing_fix.mid'
  chord_list_part = ['Cm7', 'F7', 'BbM7', 'EbM7',
                'Aø', 'D7', 'Gm7', 'Gm7',
                'Cm7', 'F7', 'BbM7', 'EbM7',
                'Aø', 'D7', 'Gm7', 'Gm7',
                'Aø', 'D7', 'Gm7', 'Gm7',
                'Cm7', 'F7', 'BbM7', 'EbM7',
                'Aø', 'D7', 'Gm7', 'Fm7',
                'Aø', 'D7', 'Gm7', 'Gm7']
  """
  chord_list_part = ['Dm', 'Bb', 'Gm', 'F',
                     'C', 'Dm', 'Dm','Bb',
                     'Bb', 'Am', 'Am', 'Bb',
                     'Bb', 'Dm', 'Dm', 'A']
  """
  chord_list = []
  for i in range(10):
    chord_list += chord_list_part
  pm = pretty_midi.PrettyMIDI(backing_midi_name)
  # tempo = pm.get_tempo_changes()[1]
  tempo = 120
  note_num_per_chord = 4
  second_per_chord = 60 / tempo 
  volume = 300

  # setting for pygame
  pygame.init()
  pygame.midi.init()
  midiOutput = pygame.midi.Output(pygame.midi.get_default_output_id())
  midiOutput.set_instrument(0)  # 0: piano

  # subprossed for backing and sound melody note
  note_on_process = Process(target=note_on,       args=())
  backing_process = Process(target=backing_music, args=(backing_midi_name,))

  # init for input
  note_series     = generate_from_random(config.CLASS_NUM,       seq_len=config.SEQ_LEN) 
  chord_id_series = generate_from_random(config.CHORD_CLASS_NUM, seq_len=config.SEQ_LEN - 1) 
  chord2id        = Chord2Id(demo=True)
  chord_to_note   = chord2id.get_chord_to_note_dict()

  chord_series = []
  append_chord_id_series = []
  for chord in chord_list:
    for _ in range(note_num_per_chord):
      chord_series.append(chord)
      append_chord_id_series.append(chord2id.get_id(chord))
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
      note_on_process.start()
      backing_process.start()
      start_time = time.time()
      last_time = time.time()

      while True:
        key = readchar.readchar()
        if key == 'q':
          break
        elapsed_time = time.time() - start_time
        current_chord_idx = int(elapsed_time / second_per_chord) + config.SEQ_LEN 

        from_last_time = time.time() - last_time
        last_time = time.time()
        add_rest_note_num = int(from_last_time / second_per_chord) 
        print(add_rest_note_num)
        for j in range(add_rest_note_num):
          note_series.append(config.CLASS_NUM - 1)

        note_input = np.array([note_series[-config.SEQ_LEN:]])
        chord_input = np.array([chord_id_series[current_chord_idx-config.SEQ_LEN:current_chord_idx]])
        print(note_input)
        
        feed_dict = {
          input_note_pl : note_input,
          input_chord_pl: chord_input,
          target_pl     : [0],
        }
        output = sess.run([pred], feed_dict)

        output = np.array(output).flatten()
        pred_note = np.argsort(output)[-1]
        """
        if pred_note == config.CLASS_NUM - 1:  # rest note class
          pred_note = np.argsort(output)[-2]
        """
        top_random = [1, 2]
        shuffle(top_random)
        pred_note = np.argsort(output)[-top_random[0]]
        if pred_note == config.CLASS_NUM - 1:  # rest note class
          pred_note = np.argsort(output)[-top_random[1]]
        note_series.append(pred_note)
        print(pred_note, chord_series[current_chord_idx - config.SEQ_LEN])
        note_on(midiOutput, pred_note, volume)

      del midiOutput
      pygame.midi.quit()
    


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
