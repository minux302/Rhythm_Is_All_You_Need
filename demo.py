import click
import numpy as np
import random
from random import shuffle
import tensorflow as tf

import config
from model import Model
from loader import Chord2Id

import os
import sys
import time
import pretty_midi
import pygame
import pygame.midi
import readchar
from multiprocessing import Process
from getch import getch

def generate_from_random(class_num, seq_len=50):
  generate = np.random.randint(0, class_num - 1, seq_len).tolist()
  return generate


def song_factory(song_name):

  if song_name == 'autumn_leaves':
    chord_list_parts = ['Cm7' , 'F7'  , 'BbM7', 'EbM7',
                        'Aø'  , 'D7'  , 'Gm7' , 'Gm7' ,
                        'Cm7' , 'F7'  , 'BbM7', 'EbM7',
                        'Aø'  , 'D7'  , 'Gm7' , 'Gm7' ,
                        'Aø'  , 'D7'  , 'Gm7' , 'Gm7' ,
                        'Cm7' , 'F7'  , 'BbM7', 'EbM7',
                        'Aø'  , 'D7'  , 'Gm7' , 'Fm7' ,
                        'Aø'  , 'D7'  , 'Gm7' , 'Gm7' ]
    tempo      = 120
    # tempo      = 131
    repeat_num = 3
  elif song_name == 'deacon_blues':
    chord_list_parts = ['CM7' , 'Em7' , 'A'   , 'D7'  ,
                        'G7'  , 'B7'  , 'Em7' , 'A7'  ,
                        'FM7' , 'B7'  ,
                        'CM7' , 'Em7' , 'A'   , 'D7'  ,
                        'G7'  , 'B7'  , 'Em7' , 'A7'  ,
                        'FM7' , 'B7'  ,
                        'CM7' , 'BbM7',
                        'DM7' , 'CM7' , 'EbM7', 'E7'  , 
                        'G7'  , 'F7'  , 'G'   , 'F7'  ,
                        'G7'  , 'F7'  , 'A'   , 'A7'  ,
                        'FM7' , 'E7'  , 'Am7' , 'Gm7' ,
                        'FM7' , 'C'   , 'D'   , 'F'   ,
                        'Am7' , 'Em7' , 'Dm7' , 'CM7' ,
                        'BbM7', 'Am7' , 'Am'  , 'C'   ,
                        'Am7' , 'Em7' , 'Dm7' , 'CM7' ,
                        'BbM7', 'Am7' , 'Am'  , 'C'   ,
                        'Am7' , 'Em7' , 'DM7' , 'C'   ,
                        'EbM7', 'E7'  ,
                        'G'   , 'F7'  , 'G7'  , 'F7'  ,
                        'G'   , 'F7'  , 'G7'  , 'D7'  ,
                        'FM7' , 'E7'  , 'Am7' , 'Bb7' ,
                        'E7'  , 'B7'  , 'B7' ,  'B7'  ,
                        'G'   , 'F7'  , 'G7'  , 'F7'  ,
                        'G'   , 'F7'  , 'G7'  , 'D7'  ,
                        'FM7' , 'E7'  , 'Am7' , 'Bb7' ,
                        'E7'  , 'B7'  , 'B7' ,  'B7' ,
                        ]
    tempo      = 140
    # tempo      = 151
    repeat_num = 1
  else:
    print("There is no midi for " + song_name)
    sys.exit()

  chord_list = []
  for i in range(repeat_num):
    chord_list += chord_list_parts
  return chord_list, tempo


class Demo:
  def __init__(self,
               song_name,
               ckpt_path,
               backing_volume_ratio=0.1):

    self.chord_list, tempo = song_factory(song_name)
    self.second_per_chord  = (60*4) / tempo
    self.volume            = 300
    self.start_time        = 0
    self.last_time         = 0

    # init for pygame
    pygame.init()
    pygame.midi.init()
    self.midiOutput = pygame.midi.Output(pygame.midi.get_default_output_id())
    self.midiOutput.set_instrument(0)  # 0: piano

    # subprossed for backing and sound melody note
    backing_path = song_name + '.mid'
    self.note_on_process = Process(target=self._note_on,    args=())
    self.backing_process = Process(target=self._backing_on, args=(backing_path,
                                                                  backing_volume_ratio*100))

    # init for session
    with tf.Graph().as_default():
      model = Model(seq_len=config.SEQ_LEN,
                    class_num=config.CLASS_NUM,
                    chord_class_num=config.CHORD_CLASS_NUM)
      self.input_note_pl, self.input_chord_pl, self.target_pl = model.placeholders()
      self.is_training_pl = tf.constant(False, name="is_training")
      self.pred           = model.infer(self.input_note_pl,
                                        self.input_chord_pl,
                                        self.is_training_pl)
      self.saver = tf.train.Saver()
      self.sess  = tf.Session()
      self.saver.restore(self.sess, ckpt_path)

    # init for input
    self.note_series     = generate_from_random(config.CLASS_NUM,       seq_len=config.SEQ_LEN)
    self.chord_id_series = generate_from_random(config.CHORD_CLASS_NUM, seq_len=config.SEQ_LEN - 1)
    self.chord2id        = Chord2Id(demo=True)

  def close(self):
    self.note_on_process.terminate()
    self.backing_process.terminate()
    del self.midiOutput
    pygame.midi.quit()

  def _note_on(self,pred_note, note_on_time=0.12):
    self.midiOutput.note_on(pred_note,  self.volume)
    time.sleep(note_on_time)
    self.midiOutput.note_off(pred_note, self.volume)

  def _backing_on(self,backing_path, volume_ratio):
    os.system('timidity {} --volume={}'.format(backing_path, volume_ratio))

  def run_bgm(self):
    self.note_on_process.start()
    self.backing_process.start()
    self.start_time = time.time() + self.second_per_chord  # tmp
    self.last_time = time.time()

  def _post_process(self, output):
    top_random = [1, 2]
    shuffle(top_random)
    pred_note = np.argsort(output)[-top_random[0]]
    if pred_note == config.CLASS_NUM - 1:  # rest note class
      pred_note = np.argsort(output)[-top_random[1]]

    self.note_series.append(pred_note)
    self._note_on(pred_note)

    return pred_note

  def run_melody(self):

    from_start_time = time.time() - self.start_time
    from_last_time  = time.time() - self.last_time
    self.last_time  = time.time()

    current_chord = self.chord_list[int(from_start_time // self.second_per_chord)]
    rest_note_num   = int(from_last_time / self.second_per_chord) 
    for i in range(rest_note_num):
      self.note_series.append(config.CLASS_NUM - 1)
      self.chord_id_series.append(self.chord2id.get_id(current_chord))  # Todo, Consideration chord for rest note
    self.chord_id_series.append(self.chord2id.get_id(current_chord))  # add next note chord

    note_input  = np.array([self.note_series[-config.SEQ_LEN:]])
    chord_input = np.array([self.chord_id_series[-config.SEQ_LEN:]])
        
    feed_dict = {
      self.input_note_pl : note_input,
      self.input_chord_pl: chord_input,
      self.target_pl     : [0],
    }
    output = self.sess.run([self.pred], feed_dict)
    output = np.array(output).flatten()

    pred_note = self._post_process(output)
    pred_note_name = pretty_midi.note_number_to_name(pred_note)
    print("chord: {:4}, note: {:4}".format(current_chord, pred_note_name))


def demo_run(song_name, ckpt_path):
  demo = Demo(song_name=song_name, ckpt_path=ckpt_path)

  demo.run_bgm()
  while True:
    key = ord(getch())
    if key == 27: # esc
      break
    start = time.time()
    demo.run_melody()
    print(time.time() - start)
  demo.close()

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
  "-s",
  "--song_name",
  help="song_name for backing",
  default="autumn_leaves",
  required=True
)
@click.option(
  "-c",
  "--ckpt_path",
  help="path to ckpt",
  default="save/v0/0_125",
  required=True
)
def main(song_name, ckpt_path):
  demo_run(song_name, ckpt_path)

if __name__ == '__main__':
  main()
