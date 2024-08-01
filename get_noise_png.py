from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from svg2png import exportsvg
# from cStringIO import StringIO
from io import StringIO
import json
import os
import time
import urllib
import zipfile
import numpy as np
import tensorflow as tf
import scipy.misc
from model import Model
import utils
import matplotlib.pyplot as plt
import random
import matplotlib
from seq2png import draw_strokes
import utils
width = 48
half_width = width / 2
seed=9001
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

plt.switch_backend('agg')

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

# Dataset directory
tf.app.flags.DEFINE_string(
    'data_dir',
    './',
    'The directory in which to find the dataset specified in model hparams. '
)

# Checkpoint directory
tf.app.flags.DEFINE_string(
    'log_root', './ckpt_mask',
    'Directory to store model checkpoints, tensorboard.')

# Resume training or not
tf.app.flags.DEFINE_boolean(
    'resume_training',True,
    'Set to true to load previous checkpoint')

# Model parameters (user defined)
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Pass in comma-separated key=value pairs such as '
    '\'save_every=40,decay_rate=0.99\' '
    '(no whitespace) to be read into the HParams object defined below')


def get_default_hparams():
    """ Return default and initial HParams """
    hparams = tf.contrib.training.HParams(
        categories=['pig'],  # Sketch categories
        num_steps=1000001,  # Number of total steps (the process will stop automatically if the loss is not improved)
        save_every=1,  # Number of epochs before saving model
        dec_rnn_size=512,  # Size of decoder
        enc_model='lstm',
        enc_rnn_size=512,
        dec_model='hyper',  # Decoder: lstm, layer_norm or hyper
        max_seq_len=-1,  # Max sequence length. Computed by DataLoader
        z_size=256,  # Size of latent variable
        batch_size=200,  # Minibatch size
        num_mixture=5,  # Recommend to set to the number of categories
        learning_rate=1e-3,  # Learning rate
        decay_rate=0.99999,   # Learning rate decay per minibatch.
        min_learning_rate=1e-5,  # Minimum learning rate
        grad_clip=1.,  # Gradient clipping
        de_weight=10.,  # Weight for deconv loss
        use_recurrent_dropout=True,  # Dropout with memory loss
        recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep
        use_input_dropout=False,  # Input dropout
        input_dropout_prob=0.9,  # Probability of input dropout keep
        use_output_dropout=False,  # Output droput
        output_dropout_prob=0.9,  # Probability of output dropout keep
        random_scale_factor=0.1,  # Random scaling data augmention proportion
        augment_stroke_prob=0.1,  # Point dropping augmentation proportion
        png_scale_ratio=1,  # Min scaling ratio
        png_rotate_angle=0,  # Max rotating angle (abs value)
        png_translate_dist=0,  # Max translating distance (abs value)
        is_training=True,  # Training mode or not
        png_width=48,  # Width of input images
        num_sub=2,  # Number of components for each category
        num_per_category=70000  # Training samples from each category
    )
    return hparams

def prepare(model_params):
    """ Prepare data and model for training """
    raw_data = utils.load_data(FLAGS.data_dir, model_params.categories, model_params.num_per_category)
    train_set, valid_set, test_set, max_seq_len = utils.preprocess_data(raw_data,
                                                                        model_params.batch_size,
                                                                        model_params.random_scale_factor,
                                                                        model_params.augment_stroke_prob,
                                                                        model_params.png_scale_ratio,
                                                                        model_params.png_rotate_angle,
                                                                        model_params.png_translate_dist)
    model_params.max_seq_len = max_seq_len
    #model_params.abs_norm = train_set.calc_abs_seq_norm()
    # Evaluating model params
    eval_model_params = utils.copy_hparams(model_params)
    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = False

    # Reset computation graph and build model
    utils.reset_graph()
    train_model = Model(model_params)
    eval_model = Model(eval_model_params, reuse=True)

    # Create new session
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())


    best_valid_cost = 1e20  # set a large init value
    best_valid_de_cost = 1e20
    epoch = 0
    count = 0
    #count = 3000
    # Save model params to a json file
    tf.gfile.MakeDirs(FLAGS.log_root)
    with tf.gfile.Open(os.path.join(FLAGS.log_root, 'model_config.json'), 'w') as f:
        json.dump(model_params.values(), f, indent=True)

    return sess, train_model, eval_model, train_set, valid_set, test_set, best_valid_cost,best_valid_de_cost, epoch, count


if __name__ == '__main__':
    model_params = get_default_hparams()
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)
    sess, model, eval_model, train_set, valid_set, test_set, best_valid_cost,best_valid_de_cost, epoch, count = prepare(model_params)
    probs = [0,0.1,0.3,0.5,0.7,0.9]
    for prob in probs:
        for batch in range(test_set.num_batches):
            seqs, pngs, labels, seq_len,seqs_bias,pngs_bias,mask = test_set.get_batch(batch,prob)
            for i in range(len(seqs_bias)):
                pre_draw = seqs_bias[i,:,:]
                pre_draw = utils.seq_5d_to_3d(pre_draw)
                filepath1 = os.path.join( './noised_%d/%d.svg' % (prob*10,i))
                draw_strokes(pre_draw, filepath1, 48, margin=1.5, color='black')
        exportsvg('./noised_%d'%(prob*10), './noised_%d_png'%(prob*10), 'png')
    