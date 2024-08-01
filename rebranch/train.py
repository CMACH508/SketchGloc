from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
import sample_abs
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Dataset directory
tf.app.flags.DEFINE_string(
    'data_dir',
    './',
    'The directory in which to find the dataset specified in model hparams. '
)

# Checkpoint directory
tf.app.flags.DEFINE_string(
    'log_root', './ckpt_ds3_999_stnum',
    'Directory to store model checkpoints, tensorboard.')

# Resume training or not
tf.app.flags.DEFINE_boolean(
    'resume_training',False,
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
        categories=['pig','bee','flower','bus','giraffe','car', 'cat' , 'horse'],  
        # Sketch categories 'pig','bee','flower','bus','giraffe'
        #ds2: 'airplane', 'angel', 'apple', 'butterfly', 'bus', 'cake','fish', 'spider', 'The Great Wall','umbrella'
        #ds3:'pig','bee','flower','bus','giraffe','car', 'cat' , 'horse'
        num_steps=1000001,  # Number of total steps (the process will stop automatically if the loss is not improved)
        re_scale=0, #Reorganization task setting
        save_every=1,  # Number of epochs before saving model
        dec_rnn_size=512,  # Size of decoder
        enc_model='lstm',
        enc_rnn_size=512,
        dec_model='hyper',  # Decoder: lstm, layer_norm or hyper
        max_seq_len=-1,  # Max sequence length. Computed by DataLoader
        max_stroke_len = -1,
        max_strokes_num = -1,
        z_size=128,  # Size of latent variable
        batch_size=100,  # Minibatch size
        num_mixture=5,  # Recommend to set to the number of categories
        learning_rate=1e-3,  # Learning rate
        decay_rate=0.99999,   # Learning rate decay per minibatch.
        min_learning_rate=1e-5,  # Minimum learning rate
        grad_clip=1.,  # Gradient clipping
        de_weight=1.,  # Weight for deconv loss
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
def get_batch_abs(seqs,set):
    seqs1 = seqs.copy()
    #seqs1[:,:,:2] *= set.scale_factor
    for i in range(len(seqs1)):
        for j in range(len(seqs1[i,:,:])-1):
            if seqs1[i,j,-1] == 1:
                break
            seqs1[i,j+1,:2]+=seqs1[i,j,:2]
    return seqs1

def get_batch_rel(seqs,set):
    seqs1 = seqs.copy()
    #seqs1[:,:,:2] *= set.scale_factor
    for i in range(len(seqs1)):
        absx=seqs1[i,0,0]
        absy=seqs1[i,0,1]
        n=1
        for j in range(len(seqs1[i,:,:])-1):
            seqs1[i,j+1,0] =  seqs1[i,j+1,0] - absx
            seqs1[i,j+1,1] =  seqs1[i,j+1,1] - absy
            absx+=seqs1[i,j+1,0]
            absy+=seqs1[i,j+1,1]
            
    return seqs1

def get_part_abs_bias(seqs,set):
    seqs1 = seqs.copy()
    #seqs1[:,:,:2] *= set.scale_factor
    for i in range(len(seqs1)):
        absx=0
        absy=0
        n=1
        for j in range(len(seqs1[i,:,:])-1):
            if seqs1[i,j,-1] == 1:
                break
            if n%10==0 or (j!=0  and seqs1[i,j-1,-2] == 1):
                #if seqs1[i,j-1,-2]==1:
                n=0
                absx += seqs1[i,j,0]
                absy += seqs1[i,j,1]
                biasx = 0
                biasy = 0
                if seqs1[i,j-1,-2] == 1:
                    biasx = np.random.randn(1) *seqs1[i,j,0]
                    biasy = np.random.randn(1)*seqs1[i,j,1]
                seqs1[i,j,0] = absx + biasx 
                seqs1[i,j,1] = absy + biasy
            else:
                absx += seqs1[i,j,0]
                absy += seqs1[i,j,1]
            #seqs1[i,j+1,:2]+=seqs1[i,j,:2]
            n+=1
    return seqs1

def get_part_abs(seqs,set):
    seqs1 = seqs.copy()
    #seqs1[:,:,:2] *= set.scale_factor
    for i in range(len(seqs1)):
        absx=0
        absy=0
        n=1
        for j in range(len(seqs1[i,:,:])-1):
            if seqs1[i,j,-1] == 1:
                break
            if (j!=0  and seqs1[i,j-1,-2] == 1):
                #if seqs1[i,j-1,-2]==1:
                n=0
                absx += seqs1[i,j,0]
                absy += seqs1[i,j,1]
                seqs1[i,j,0] = absx
                seqs1[i,j,1] = absy
            else:
                absx += seqs1[i,j,0]
                absy += seqs1[i,j,1]
            #seqs1[i,j+1,:2]+=seqs1[i,j,:2]
            n+=1
    return seqs1

def get_part_rel(seqs,set):
    seqs1 = seqs.copy()
    #seqs1[:,:,:2] *= set.scale_factor
    for i in range(len(seqs1)):
        absx=0
        absy=0
        n=1
        for j in range(len(seqs1[i,:,:])-1):
            if seqs1[i,j,-1] == 1:
                break
            if  (j!=0  and seqs1[i,j-1,-2]==1):
                #if seqs1[i,j-1,-2]==1:
                n=0
                seqs1[i,j,0] = seqs1[i,j,0] - absx
                seqs1[i,j,1] = seqs1[i,j,1] - absy
                absx += seqs1[i,j,0]
                absy += seqs1[i,j,1]
                
            else:
                absx += seqs1[i,j,0]
                absy += seqs1[i,j,1]
            #seqs1[i,j+1,:2]+=seqs1[i,j,:2]
            n+=1
    return seqs1

def evaluate_model(sess, model, data_set):
    """ Evaluating process """
    total_loss = 0.0
    #alpha_loss = 0.0
    #gaussian_loss = 0.0
    lil_loss = 0.0
    de_loss = 0.0

    for batch in range(data_set.num_batches):
        seqs, pngs, labels, seq_len,s_n,stroke_len,stroke_num,_,box,_  = data_set.get_batch(batch)
        #seqs[:,:2] /= data_set.scale_factor
        data_copy = seqs[0, :, :].copy()
        # data_copy = utils.seq_5d_to_3d(data_copy)
        # print(data_copy.shape)
        data_copy = np.split(data_copy[:, :], np.where(data_copy[:, 2])[0] + 1, axis=0)[:-1]
        data_bound = np.zeros([len(data_copy), 4])
        s_n, stroke_len = pad_stroke_batch(s_n, stroke_len, model.hps.max_strokes_num, model.hps.batch_size, )
        #abs_seqs[:,:2] += half_width
        #abs_seqs[:,:,:2] = abs_seqs[:,:,:2] / model.hps.abs_norm
        feed = {model.input_seqs: seqs, model.input_strokes: s_n, model.input_stroke_len: stroke_len,
                model.input_strokes_num: stroke_num, model.input_box: box}

        code = sess.run(model.p_mu, feed_dict=feed)

        feed = {model.batch_z:code,model.input_seqs: seqs}

        total_cost, lil_cost = \
            sess.run([model.loss,  model.lil_loss], feed)
        total_loss += total_cost
        #alpha_loss += alpha_cost
        #gaussian_loss += gaussian_cost
        lil_loss += lil_cost

    total_loss /= (data_set.num_batches)
    #alpha_loss /= (data_set.num_batches)
    #gaussian_loss /= (data_set.num_batches)
    lil_loss /= (data_set.num_batches)

    return total_loss,  lil_loss
def pad_stroke_batch(stroke_set,stroke_len,max_num,bs):
    l = len(stroke_set)
    stroke_set = np.concatenate((stroke_set,np.zeros([max_num*bs-l,stroke_set.shape[1],2])),axis=0)
    stroke_len = np.concatenate((stroke_len, np.zeros([max_num * bs - l])), axis=0)
    return stroke_set,stroke_len

def _train(sess, model, train_set, epoch, sum):
    """ Training process """
    start = time.time()

    index = np.arange(len(train_set.strokes))
    np.random.shuffle(index)
    count = 0

    for begin, end in zip(range(0, len(index), model.hps.batch_size),
                          range(model.hps.batch_size, len(index), model.hps.batch_size)):
        batch_index = index[begin:end]
        seqs, pngs, labels, seq_len,s_n,stroke_len,stroke_num,_,box,_  = train_set._get_batch_from_indices(batch_index,model.hps.re_scale)
        #seqs[:,:2] /= train_set.scale_factor
        data_copy = seqs[0, :, :].copy()
        # data_copy = utils.seq_5d_to_3d(data_copy)
        # print(data_copy.shape)
        data_copy = np.split(data_copy[:, :], np.where(data_copy[:, 2])[0] + 1, axis=0)[:-1]
        data_bound = np.zeros([len(data_copy), 4])
        s_n, stroke_len = pad_stroke_batch(s_n, stroke_len, model.hps.max_strokes_num, model.hps.batch_size, )
        #abs_seqs[:,:2] += half_width
        #abs_seqs[:,:,:2] = abs_seqs[:,:,:2] / model.hps.abs_norm
        feed = {model.input_seqs: seqs, model.input_strokes: s_n, model.input_stroke_len: stroke_len,
                model.input_strokes_num: stroke_num, model.input_box: box}

        total_cost,  lil_cost,batch_z, _= \
            sess.run([model.loss, model.lil_loss,model.batch_z, model.train_op
                      ], feed)
    


            

        count += 1
        sum += 1

        # Record the value of losses
        if count%20 == 0:
            end = time.time()
            time_taken = end - start
            start = time.time()


            print('Epoch: %d, Step: %d,  Lil: %.4f , Time: %.4f,'
                  % (epoch, count, lil_cost, time_taken))
    epoch += 1
    return epoch, sum


def _validate(sess, eval_model, valid_set):
    """ Validating process """
    start = time.time()
    valid_loss,  valid_lil_loss = \
        evaluate_model(sess, eval_model, valid_set)
    end = time.time()
    time_taken_valid = end - start

    print('Valid_cost: %.4f, Lil: %.4f, Time_taken: %.4f' %
          (valid_loss,  valid_lil_loss, time_taken_valid))
    return valid_lil_loss


def _test(sess, eval_model, test_set):
    """ Testing process """
    start = time.time()
    test_loss,  test_lil_loss = \
        evaluate_model(sess, eval_model, test_set)
    end = time.time()
    time_taken_test = end - start

    print('Test_cost: %.4f,  Lil: %.4f,  Time_taken: %.4f' %
          (test_loss,  test_lil_loss, time_taken_test))


def prepare(model_params):
    """ Prepare data and model for training """
    raw_data = utils.load_data(FLAGS.data_dir, model_params.categories, model_params.num_per_category)
    train_set, valid_set, test_set, max_seq_len,max_stroke_len,max_strokes_num = utils.preprocess_data(raw_data,
                                                                        model_params.batch_size,
                                                                        model_params.random_scale_factor,
                                                                        model_params.augment_stroke_prob,
                                                                        model_params.png_scale_ratio,
                                                                        model_params.png_rotate_angle,
                                                                        model_params.png_translate_dist)
    model_params.max_seq_len = max_seq_len
    model_params.max_strokes_num = max_strokes_num
    model_params.max_stroke_len = max_stroke_len
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

    s_model,s_draw_model = sample_abs.get_sam_p(0,test_set,model_params,0)

    # Create new session
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())

    # Load checkpoint if resume training
    if FLAGS.resume_training:
        sess, epoch, count, best_valid_cost = load_checkpoint(sess, FLAGS.log_root)

    else:
        best_valid_cost = 1e20  # set a large init value
        #best_valid_de_cost = 1e20
        epoch = 0
        count = 0
    #count = 3000
    # Save model params to a json file
    tf.gfile.MakeDirs(FLAGS.log_root)
    with tf.gfile.Open(os.path.join(FLAGS.log_root, 'model_config.json'), 'w') as f:
        json.dump(model_params.values(), f, indent=True)

    return sess, train_model, eval_model, train_set, valid_set, test_set, best_valid_cost, epoch, count ,s_model,s_draw_model


def load_checkpoint(sess, log_root):
    """ Load checkpoints"""
    utils.load_checkpoint(sess, log_root)
    file = np.load(FLAGS.log_root + "/para.npz")
    best_valid_cost = float(file['best_valid_loss'])

    epoch = int(file['epoch'])  # Last epoch during training
    count = int(file['count'])  # Previous accumulated steps for training
    return sess, epoch, count, best_valid_cost


def train_model(model_params):
    """ Main branch for RPCLVQ """
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)
    #print(model_params)
    sess, model, eval_model, train_set, valid_set, test_set, best_valid_cost, epoch, count ,s_model,s_draw_model= prepare(model_params)
    #sess.run(tf.variables_initializer(model.optimizer.variables()))
    #s_model,s_draw_model = sample_abs.get_sam_p(sess,test_set,model_params,epoch)


    cnt = 0  # Number of invalid training epoch
    for epo in range(100000):
        epoch, count = _train(sess, model, train_set, epoch, count)

        if (epoch % model_params.save_every) == 0 :
            print('Best_valid_loss: %4.4f  ' % (best_valid_cost))
            valid_cost = _validate(sess, eval_model, valid_set)

            if  best_valid_cost> valid_cost or epo==0:
                best_valid_cost = valid_cost
#                best_valid_de_cost = valid_de_loss
                # Save model to checkpoint path
                start = time.time()
                utils.save_model(sess, FLAGS.log_root, epoch)

                np.savez(FLAGS.log_root + "/para", best_valid_loss=best_valid_cost, epoch=epoch, count=count)
                end = time.time()
                time_taken_save = end - start
                print('time_taken_save %4.4f.' % time_taken_save)
                print('sampling...')
                sample_abs.sample_def(sess,test_set,model_params,epoch,s_model,s_draw_model)
                _test(sess, eval_model, test_set)
                cnt = 0
            elif (cnt+1) %20==0:  # Reload the last checkpoint
                sess, epoch, count, best_valid_cost = load_checkpoint(sess, FLAGS.log_root)
                cnt += 1
            else:
                 cnt +=1

            if cnt >= 5:  # No improvement on validation cost for five validation steps
                print("===================================")
                print("           No Improvement          ")
                print("===================================")
                break
    


def main(unused_argv):
    """Load model params, save config file and start trainer."""
    model_params = get_default_hparams()
    if FLAGS.hparams:
        model_params.parse(FLAGS.hparams)
    train_model(model_params)


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()