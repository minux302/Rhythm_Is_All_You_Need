import os
import click
import json
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy

import config
from model import Model
from loader import MelodyandChordLoader, get_p_extension_list


def train(id, reset):

  # set log and ckpt dir
  log_dir = config.LOG_DIR
  save_dir = config.SAVE_DIR
  if not(os.path.exists(log_dir)):
    os.system('mkdir ' + log_dir)
  if not(os.path.exists(save_dir)):
    os.system('mkdir ' + save_dir)

  log_id_dir = os.path.join(log_dir, id)
  save_id_dir = os.path.join(save_dir, id)
  if reset:
    if os.path.exists(log_id_dir):
      os.system('rm -rf ' + log_id_dir)
    if not(os.path.exists(save_id_dir)):
      os.system('rm -rf ' + save_id_dir)

  # set train and valid loader
  p_midi_list_train = get_p_extension_list(os.path.join(config.DATA_DIR, 'train'), 'mid')
  p_midi_list_valid = get_p_extension_list(os.path.join(config.DATA_DIR, 'valid'), 'mid')
  train_loader = MelodyandChordLoader(p_midi_list=p_midi_list_train,
                                      seq_len=config.SEQ_LEN,
                                      class_num=config.CLASS_NUM,
                                      fs=config.FRAME_PER_SECOND,
                                      batch_song_size=config.BATCH_SONG_SIZE,
                                      batch_size=config.BATCH_SIZE)
  valid_loader = MelodyandChordLoader(p_midi_list=p_midi_list_valid,
                                      seq_len=config.SEQ_LEN,
                                      class_num=config.CLASS_NUM,
                                      fs=config.FRAME_PER_SECOND,
                                      batch_song_size=config.BATCH_SONG_SIZE,
                                      batch_size=config.BATCH_SIZE)

  with tf.Graph().as_default():

    # model build
    model = Model(config.SEQ_LEN, config.CLASS_NUM)
    input_pl, label_pl = model.placeholders()
    pred = model.infer(input_pl)
    loss = model.loss(pred, label_pl)
    opt = model.optimizer(loss)

    config_gpu = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config_gpu) as sess:

      # init for train
      init_op = tf.global_variables_initializer()
      sess.run([init_op])
      batch_song_num = train_loader.get_batch_song_num()
      iter_num = 0
      tqdm_bar = tqdm(total=train_loader.get_total_songs())

      for epoch in range(config.EPOCHS):

        train_loader.shuffle_midi_list()

        for i in range(0, batch_song_num):

          tqdm_bar.update(config.BATCH_SONG_SIZE)
          train_loader.generate_batch_buffer(i)
          batch_num = train_loader.get_batch_num()

          for j in range(0, batch_num):
            batch_input, batch_target = train_loader.get_batch(j)

            feed_dict = {
                input_pl: batch_input,
                label_pl: batch_target,
            }
            _, _loss = sess.run([opt, loss], feed_dict)

            iter_num += 1
            # print("epochs {} | Steps {} | total loss : {}".format(epoch + 1, iter_num, _loss))

        """
        if config.VALIDATION_INTERVAL
          for i in range(0, batch_song_num):

            valid_loader.generate_batch_buffer(i)
            batch_num = valid_loader.get_batch_num()

            for j in range(0, batch_num):
              batch_input, batch_target = valid_loader.get_batch(j)

              feed_dict = {
                  input_pl: batch_input,
                  label_pl: batch_target,
              }
            _, _loss = sess.run([opt, loss], feed_dict)
        """



@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-i",
    "--id",
    help="training id",
    default="0",
    required=False
)
@click.option(
    "-r",
    "--reset",
    help="remove directory for ckpt and tensorboard",
    default="True",
    required=False
)
def main(id, reset):
    train(id, reset)


if __name__ == '__main__':
    main()
