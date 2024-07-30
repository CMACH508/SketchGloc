
import random
import sys
import os
import json
import numpy as np
import tensorflow as tf
import utils
import glob
from PIL import Image
from seq2png import draw_strokes
from model import Model
from sample import sample
import scipy.misc
import re
import utils


def load_model_params(model_dir):
    model_params = utils.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_config = json.dumps(json.load(f))
        model_params.parse_json(model_config)
    return model_params


def modify_model_params(model_params):
    model_params.use_input_dropout = 0
    model_params.use_recurrent_dropout = 0
    model_params.use_output_dropout = 0
    model_params.is_training = False
    model_params.batch_size = 1
    model_params.max_seq_len = 1

    return model_params

def sort_paths(paths):
    idxs = []
    for path in paths:
        idxs.append(int(re.findall(r'\d+', path)[-1]))

    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            if idxs[i] > idxs[j]:
                tmp = idxs[i]
                idxs[i] = idxs[j]
                idxs[j] = tmp

                tmp = paths[i]
                paths[i] = paths[j]
                paths[j] = tmp
    return paths

def get_batch_abs(seqs,set):
    seqs1 = seqs.copy()
    #seqs1[:,:,:2] *= set.scale_factor
    for i in range(len(seqs1)):
        for j in range(len(seqs1[i,:,:])-1):
            if seqs1[i,j,-1] == 1:
                break
            seqs1[i,j+1,:2]+=seqs1[i,j,:2]
    return seqs1
def get_part_abs(seqs,set=None):
    seqs1 = seqs.copy()
    #seqs1[:,:,:2] *= set.scale_factor
    for i in range(len(seqs1)):
        absx=0
        absy=0
        n=1
        for j in range(len(seqs1[i,:,:])-1):
            if seqs1[i,j,-1] == 1:
                break
            if n%10==0 or (j!=0  and seqs1[i,j-1,-2]) == 1:
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
def pad_stroke_batch(stroke_set,stroke_len,max_num,bs):
    l = len(stroke_set)
    #print(l)
    stroke_set = np.concatenate((stroke_set,np.zeros([max_num*bs-l,stroke_set.shape[1],2])),axis=0)
    stroke_len = np.concatenate((stroke_len, np.zeros([max_num * bs - l])), axis=0)
    return stroke_set,stroke_len
def main():
    FLAGS = tf.app.flags.FLAGS
    args = sys.argv
    sample_path = args[1]
    model_ckpt = args[2]
    # Checkpoint directory
    tf.app.flags.DEFINE_string(
        'model_dir', './ckpt_ds1',
        'Directory to store the model checkpoints.'
    )
    # Sample directory
    tf.app.flags.DEFINE_string(
        'output_dir', './sample_ds1/sample_ds1_17',       #'./sample_ds1_34_test'
        'Directory to store the generated sketches.'
    )
    tf.app.flags.DEFINE_string(
        'data_dir',
        './',
        'The directory in which to find the dataset specified in model hparams. '
    )
    model_dir = FLAGS.model_dir
    model_dir =  model_ckpt
    SVG_DIR = FLAGS.output_dir
    SVG_DIR = sample_path
    

    model_params = load_model_params(model_dir)
    model_params = modify_model_params(model_params)
    raw_data = utils.load_data(FLAGS.data_dir, model_params.categories, model_params.num_per_category)
    train_set, valid_set, test_set, max_seq_len,max_stroke_len,max_strokes_num = utils.preprocess_data(raw_data,
                                                                        model_params.batch_size,
                                                                        model_params.random_scale_factor,
                                                                        model_params.augment_stroke_prob,
                                                                        model_params.png_scale_ratio,
                                                                        model_params.png_rotate_angle,
                                                                        model_params.png_translate_dist)
    model_params.max_seq_len  = max_seq_len
    model = Model(model_params)

    for label in range(len(model_params.categories)):
        img_paths = glob.glob(SVG_DIR + '/%d_*.png' % label)
        code_paths = glob.glob(SVG_DIR + '/code_%d_*.npy' % label)
        seq_paths = glob.glob(SVG_DIR + '/s_%d_*.npy' % label)
        img_paths = sort_paths(img_paths)
        code_paths = sort_paths(code_paths)
        seq_paths = sort_paths(seq_paths)
        if label == 0:
            img = np.array(img_paths)
            code = np.array(code_paths)
            seq = np.array(seq_paths)
        else:
            img = np.hstack((img, np.array(img_paths)))
            code = np.hstack((code, np.array(code_paths)))
            seq = np.hstack((seq, np.array(seq_paths)))

    seq_data = []

    for path in seq:
        seq = np.load(path, encoding='latin1', allow_pickle=True)
        #print(seq.shape)
        #abs part
        for ii in range(seq.shape[1]):
            seq[0,seq.shape[1]-ii-1,:2] -= seq[0,seq.shape[1]-ii-2,:2]

        seq = test_set.pad_seq_batch(seq,max_seq_len)
        seq = np.reshape(seq, [1, -1, 5])
        
        if seq_data == []:
            seq_data = seq
        else:
            seq_data = np.concatenate([seq_data, seq], axis=0)

    code_data = []
    for path in code:
        code_data.append(np.load(path))
    code_data = np.reshape(code_data, [-1, model_params.z_size])  # Real codes for original sketches

    sample_size = len(code_data)  # Number of samples for retrieval

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())
    utils.load_checkpoint(sess, model_dir)
    #seq_abs_data = get_part_abs(seq_data,train_set)
    #print(seq_data)
    for i in range(len(seq_data[:, 0, 0])):
        seqs = np.reshape(seq_data[i, :, :], [1, -1, 5])
        seqs, pngs, labels, seq_len,s_n,stroke_len,stroke_num,_,box  = test_set.get_box_input(seqs)
        s_n, stroke_len = pad_stroke_batch(s_n, stroke_len, model.hps.max_strokes_num, model.hps.batch_size )
        #seq_abs_data[:, :2] += half_width
        #seq_abs_data[:,:,:2] /= model.hps.abs_norm
        seq_len = np.reshape(len(seq_data[i, :, 0]),[1])
        feed = {model.input_seqs: seqs, model.input_strokes: s_n, model.input_stroke_len: stroke_len,
                model.input_strokes_num: stroke_num, model.input_box: box}
        z = sess.run(model.p_mu, feed)  # Codes of the samples

        if i == 0:
            batch_z = z
        else:
            batch_z = np.concatenate([batch_z, z], axis=0)  # Codes for generations

    # Begin retrieval
    top_1 = 0.
    top_10 = 0.
    top_50 = 0.

    temp_sample_size = int(sample_size / 10)
    for ii in range(10):  # reduce the risk of memory out
        real_code = np.tile(np.reshape(code_data, [sample_size, 1, model_params.z_size]), [1, temp_sample_size, 1])
        fake_code = np.tile(np.reshape(batch_z[temp_sample_size * ii:temp_sample_size * (ii + 1), :],
                                       [1, temp_sample_size, model_params.z_size]), [sample_size, 1, 1])
        distances = np.average((real_code[:,:,:] - fake_code[:,:,:]) ** 2,
                               axis=2)  # Distances between each two codes, sample_size * sample_size

        for n in range(50):
            temp_index = np.argmin(distances, axis=0)
            for i in range(temp_sample_size):
                distances[temp_index[i], i] = 1e10
            if n == 0:
                top_n_index = np.reshape(temp_index, [1, -1])
            else:
                top_n_index = np.concatenate([top_n_index, np.reshape(temp_index, [1, -1])], axis=0)

        for i in range(temp_sample_size):
            if (i + temp_sample_size * ii) != top_n_index[0, i]:
                print(i + temp_sample_size * ii,' to ',top_n_index[0, i])
            if top_n_index[0, i] == i + temp_sample_size * ii:
                top_1 += 1.
            for k in range(10):
                if top_n_index[k, i] == i + temp_sample_size * ii:
                    top_10 += 1.
                    break
            for k in range(50):
                if top_n_index[k, i] == i + temp_sample_size * ii:
                    top_50 += 1.
                    break

    print("Top 1 Ret: " + str(float(top_1 / sample_size)))
    print("Top 10 Ret: " + str(float(top_10 / sample_size)))
    print("Top 50 Ret: " + str(float(top_50 / sample_size)))

if __name__ == '__main__':
    main()