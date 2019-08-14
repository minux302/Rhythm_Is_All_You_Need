import click
import tensorflow as tf

import config
from model import Model
from loader import Chord2Id


def ckpt2pb(ckpt_path):

  output_node_name = "output"
  pb_name = "saved_model.pb"
  pb_path = "pb"

  # init for session
  with tf.Graph().as_default():
    model = Model(seq_len=config.SEQ_LEN,
                  class_num=config.CLASS_NUM,
                  chord_class_num=config.CHORD_CLASS_NUM)
    input_note_pl, input_chord_pl, target_pl = model.placeholders()
    is_training_pl = tf.constant(False, name="is_training")
    pred           = model.infer(input_note_pl,
                                 input_chord_pl,
                                 is_training_pl)
    saver = tf.train.Saver()

    with tf.Session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      saver.restore(sess, ckpt_path)

      # print(pred.name)
      # for v in tf.global_variables():
      #   print(v.name)
      # for n in tf.get_default_graph().as_graph_def().node:
      #   print(n.name)
      minimal_graph = tf.graph_util.convert_variables_to_constants(
      # minimal_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        # [output_node_name],
        ["output"],
      )

      tf.train.write_graph(minimal_graph, pb_path, pb_name, as_text=False)

      """
      # graph = tf.get_default_graph() 
      tf.saved_model.simple_save(sess,
                                 pb_path,
                                 inputs={"input/input_note": input_note_pl, "input/input_chord": input_chord_pl},
                                 outputs={"output": pred})
      """

      # tensorflowjs_converter --input=tf_frozen_model --output_node_names='output' --output_json='model.json' pb/saved_model.pb web_model/
      # tensorflowjs_converter --input=tf_saved_model pb/ web_model/

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
  "-c",
  "--ckpt_path",
  help="path to ckpt",
  default="save/0/0_1",
  required=True
)
def main(ckpt_path):
  ckpt2pb(ckpt_path)

if __name__ == '__main__':
  main()