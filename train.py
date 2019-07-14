import os
import click
import json
import time
import datetime
import tensorflow as tf

import config
from model import Model
from loader import MelodyandChordLoader, get_p_extension_list


def train(id, reset):

  # set log and ckpt dir
  log_dir  = config.LOG_DIR
  save_dir = config.SAVE_DIR
  if not(os.path.exists(log_dir)):
    os.system('mkdir ' + log_dir)
  if not(os.path.exists(save_dir)):
    os.system('mkdir ' + save_dir)

  log_id_dir  = os.path.join(log_dir, id)
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

  start_time = time.time()
  with tf.Graph().as_default():

    # model build
    model = Model(config.SEQ_LEN, config.CLASS_NUM)
    input_pl, label_pl = model.placeholders()
    pred  = model.infer(input_pl)
    loss  = model.loss(pred, label_pl)
    opt   = model.optimizer(loss)

    saver = tf.train.Saver()
    config_gpu = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

    with tf.Session(config=config_gpu) as sess:

      # init
      batch_song_iter_num = 0
      init_op = tf.global_variables_initializer()
      sess.run([init_op])

      merged       = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter(os.path.join(log_dir, id + '/train'), sess.graph)
      valid_writer = tf.summary.FileWriter(os.path.join(log_dir, id + '/valid'))

      train_batch_song_num = train_loader.get_batch_song_num()
      # valid_batch_song_num = valid_loader.get_batch_song_num()

      for epoch in range(config.EPOCHS):

        # train
        train_loader.shuffle_midi_list()
        for batch_song_idx in range(0, train_batch_song_num):

          # select 'batch_song_num' songs from dataset
          train_loader.generate_batch_buffer(batch_song_idx)
          batch_num = train_loader.get_batch_num()

          for batch_idx in range(0, batch_num):

            # create input data from selected songs
            batch_input, batch_target = train_loader.get_batch(batch_idx) 

            feed_dict = {
                input_pl: batch_input,
                label_pl: batch_target,
            }
            _, _loss = sess.run([opt, loss], feed_dict)

          batch_song_iter_num += 1
          loss_summary, summary = sess.run([loss, merged], feed_dict)  # summary for last batch
          train_writer.add_summary(summary, batch_song_iter_num)
          print("epoch: {}, song: {}/{}, Loss: {}".format(epoch + 1,
                                                          batch_song_idx * config.BATCH_SONG_SIZE, 
                                                          train_loader.get_total_songs(),
                                                          loss_summary))

          # valid
          if batch_song_iter_num % config.VALIDATION_INTERVAL == 0:

            # validate one batch only for time saving 
            valid_loader.shuffle_midi_list()
            valid_loader.generate_batch_buffer(0)
            batch_input, batch_target = valid_loader.get_batch(0)

            feed_dict = {
                input_pl: batch_input,
                label_pl: batch_target,
            }

            _, summary = sess.run([loss, merged], feed_dict)
            valid_writer.add_summary(summary, batch_song_iter_num)

        # save
        save_path = os.path.join(save_id_dir, id + '_' + str(epoch))
        saver.save(sess, save_path)

  train_writer.close()
  valid_writer.close()

  print('train is finished !!')
  td = datetime.timedelta(seconds=time.time() - start_time)
  print("time: ", td)


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
