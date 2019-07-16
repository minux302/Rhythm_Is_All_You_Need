import click
import numpy as np
import pretty_midi
import tensorflow as tf
import time

import config
from model import Model


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
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

def generate_from_random(unique_notes, seq_len=50):
  generate = np.random.randint(0,unique_notes,seq_len).tolist()
  return generate
    

def create_sample():

  generate  = generate_from_random(config.CLASS_NUM, seq_len=config.SEQ_LEN) 
  # generate =  [67, 67, 128, 128,  71,  67,  67,  71,  67,  67]
  generate = [62, 74, 74, 128, 91, 72, 128, 128, 71, 57, 128, 87, 84, 69, 69, 68, 128, 68, 68, 68]

  with tf.Graph().as_default():

    # model build
    model              = Model(config.SEQ_LEN, config.CLASS_NUM)
    input_pl, label_pl = model.placeholders()
    is_training_pl = tf.constant(False, name="is_training")
    pred               = model.infer(input_pl, is_training_pl)

    saver = tf.train.Saver()
    config_gpu = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

    with tf.Session(config=config_gpu) as sess:

      saver.restore(sess, 'save/0/0_99')

      max_generated=100
      for i in range(max_generated):
        test_input = np.array([generate[i:i+config.SEQ_LEN]])
        print(test_input)
        feed_dict = {
          input_pl: test_input,
          label_pl: [1]
        }
        output = sess.run([pred], feed_dict)
        output = np.array(output).flatten()
        output_argsort = np.argsort(output)

        # idx = np.random.randint(2)
        # pred_note = output_argsort[-idx]
        pred_note = output_argsort[-1]
        generate.append(pred_note)


    file_name = "result.mid"
    start_index=49
    fs=config.FRAME_PER_SECOND
    print(generate)

    array_piano_roll = np.zeros((128,max_generated+1), dtype=np.int16)
    for index, note in enumerate(generate[start_index:]):
      if note == 128:
        pass
      else:
        array_piano_roll[note,index] = 1

    generate_to_midi = piano_roll_to_pretty_midi(array_piano_roll, fs=fs)
    for note in generate_to_midi.instruments[0].notes:
      note.velocity = 100
    generate_to_midi.write(file_name)




if __name__ == '__main__':
  create_sample()
