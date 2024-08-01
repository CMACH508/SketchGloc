from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import tensorflow as tf
import numpy as np
import os
import math
import cv2
import six
from bresenham import bresenham
import scipy.ndimage
import seq2png
from PIL import Image



width = 48
half_width = width / 2
num_sub = 30
num_ss = 5
max_part_len = 5
def get_default_hparams():
    """ Return default and initial HParams """
    hparams = tf.contrib.training.HParams(
        categories=['bee'],  # Sketch categories
        num_steps=1000001,  # Number of total steps (the process will stop automatically if the loss is not improved)
        save_every=1,  # Number of epochs before saving model
        dec_rnn_size=512,  # Size of decoder
        enc_model='lstm',
        enc_rnn_size=512,
        dec_model='hyper',  # Decoder: lstm, layer_norm or hyper
        max_seq_len=-1,  # Max sequence length. Computed by DataLoader
        max_stroke_len=-1,
        max_strokes_num=-1,
        z_size=256,  # Size of latent variable
        batch_size=200,  # Minibatch size
        num_mixture=5,  # Recommend to set to the number of categories
        learning_rate=0.001,  # Learning rate
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        min_learning_rate=0.00001,  # Minimum learning rate
        grad_clip=1.,  # Gradient clipping
        de_weight=0.5,  # Weight for deconv loss
        use_recurrent_dropout=True,  # Dropout with memory loss
        recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep
        use_input_dropout=False,  # Input dropout
        input_dropout_prob=0.90,  # Probability of input dropout keep
        use_output_dropout=False,  # Output droput
        output_dropout_prob=0.9,  # Probability of output dropout keep
        random_scale_factor=0.10,  # Random scaling data augmention proportion
        augment_stroke_prob=0.10,  # Point dropping augmentation proportion
        png_scale_ratio=0.98,  # Min scaling ratio
        png_rotate_angle=0,  # Max rotating angle (abs value)
        png_translate_dist=0,  # Max translating distance (abs value)
        is_training=True,  # Training mode or not
        png_width=96,  # Width of input images
        num_sub=2,  # Number of components for each category
        num_per_category=70000,  # Training samples from each category
        abs_norm = 1.0
    )
    return hparams


def copy_hparams(hparams):
    """ Return a copy of an HParams instance """
    return tf.contrib.training.HParams(**hparams.values())


def reset_graph():
    """ Close the current default session and resets the graph """
    sess = tf.get_default_session()
    if sess:
        sess.close()
    tf.reset_default_graph()


def load_seqs(data_dir, categories):
    """ Load sequence raw data """
    if not isinstance(categories, list):
        categories = [categories]

    train_seqs = None
    valid_seqs = None
    test_seqs = None

    for ctg in categories:
        # load sequence data
        seq_path = os.path.join(data_dir, ctg + '.npz')
        if six.PY3:
            seq_data = np.load(seq_path, encoding='latin1', allow_pickle=True)
        else:
            seq_data = np.load(seq_path, allow_pickle=True)
        tf.logging.info('Loaded sequences {}/{}/{} from {}'.format(
            len(seq_data['train']), len(seq_data['valid']), len(seq_data['test']),
            ctg + '.npz'))

        if train_seqs is None:
            train_seqs = seq_data['train']
            valid_seqs = seq_data['valid']
            test_seqs = seq_data['test']
        else:
            train_seqs = np.concatenate((train_seqs, seq_data['train']))
            valid_seqs = np.concatenate((valid_seqs, seq_data['valid']))
            test_seqs = np.concatenate((test_seqs, seq_data['test']))

    return train_seqs, valid_seqs, test_seqs


def load_data(data_dir, categories, num_per_category):
    """ Load sequence and image raw data """
    if not isinstance(categories, list):
        categories = [categories]

    train_seqs = None
    train_pngs = None
    valid_seqs = None
    valid_pngs = None
    test_seqs = None
    test_pngs = None
    train_labels = None
    valid_labels = None
    test_labels = None

    i = 0
    for ctg in categories:
        # load sequence data
        seq_path = os.path.join(data_dir, ctg + '.npz')
        if six.PY3:
            seq_data = np.load(seq_path, encoding='latin1', allow_pickle=True)
        else:
            seq_data = np.load(seq_path, allow_pickle=True)
        tf.logging.info('Loaded sequences {}/{}/{} from {}'.format(
            len(seq_data['train']), len(seq_data['valid']), len(seq_data['test']),
            ctg + '.npz'))

        if train_seqs is None:
            train_seqs = seq_data['train'][0:num_per_category]
            valid_seqs = seq_data['valid'][0:num_per_category]
            test_seqs = seq_data['test'][0:num_per_category]
        else:
            train_seqs = np.concatenate((train_seqs, seq_data['train'][0:num_per_category]))
            valid_seqs = np.concatenate((valid_seqs, seq_data['valid'][0:num_per_category]))
            test_seqs = np.concatenate((test_seqs, seq_data['test'][0:num_per_category]))

        # load png data
        
        # create labels
        if train_labels is None:
            train_labels = i * np.ones([num_per_category], dtype=np.int)
            valid_labels = i * np.ones([num_per_category], dtype=np.int)
            test_labels = i * np.ones([num_per_category], dtype=np.int)
        else:
            train_labels = np.concatenate([train_labels, i * np.ones([num_per_category], dtype=np.int)])
            valid_labels = np.concatenate([valid_labels, i * np.ones([num_per_category], dtype=np.int)])
            test_labels = np.concatenate([test_labels, i * np.ones([num_per_category], dtype=np.int)])
        i += 1

    return [train_seqs, valid_seqs, test_seqs,
            train_pngs, valid_pngs, test_pngs,
            train_labels, valid_labels, test_labels]


def preprocess_data(raw_data, batch_size, random_scale_factor, augment_stroke_prob, png_scale_ratio, png_rotate_angle,
                    png_translate_dist):
    """ Convert raw data to suitable model inputs """
    train_seqs, valid_seqs, test_seqs, train_pngs, valid_pngs, test_pngs, train_labels, valid_labels, test_labels = raw_data
    all_strokes = np.concatenate((train_seqs, valid_seqs, test_seqs))
    max_seq_len ,max_strokes_len,max_strokes_num= get_max_len(all_strokes)
    #lll max rev
    #max_strokes_len = max_part_len #5
    #max_strokes_num = max_strokes_num*4
    ###
    #max_strokes_len = 83
    #max_strokes_num = 23
    # set1 params eg.

    #45 18
    #set2

    ###
    print('max_seq',max_seq_len)
    train_set = DataLoader(
        train_seqs,
        train_pngs,
        train_labels,
        batch_size,
        max_seq_length=max_seq_len,
        max_strokes_len = max_strokes_len,
        max_strokes_num=max_strokes_num,
        random_scale_factor=random_scale_factor,
        augment_stroke_prob=augment_stroke_prob,
       png_scale_ratio=png_scale_ratio,
       png_rotate_angle=png_rotate_angle,
       png_translate_dist=png_translate_dist   )
        
#random_scale_factor=random_scale_factor,
#        augment_stroke_prob=augment_stroke_prob,
 #       png_scale_ratio=png_scale_ratio,
 #       png_rotate_angle=png_rotate_angle,
 #       png_translate_dist=png_translate_dist        
        
    seq_norm = train_set.calc_seq_norm()
    train_set.normalize_seq(seq_norm)

    valid_set = DataLoader(
        valid_seqs,
        valid_pngs,
        valid_labels,
        batch_size,
        max_seq_length=max_seq_len,
        max_strokes_len=max_strokes_len,
        max_strokes_num=max_strokes_num
    )
    valid_set.normalize_seq(seq_norm)

    test_set = DataLoader(
        test_seqs,
        test_pngs,
        test_labels,
        batch_size,
        max_seq_length=max_seq_len,
        max_strokes_len=max_strokes_len,
        max_strokes_num=max_strokes_num
    )
    test_set.normalize_seq(seq_norm)

    tf.logging.info('normalizing_scale_factor %4.4f.', seq_norm)
    return train_set, valid_set, test_set, max_seq_len,max_strokes_len,max_strokes_num


def load_checkpoint(sess, checkpoint_path):
    """ Load checkpoint of saved model """
    saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    print('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)


def save_model(sess, model_save_path, global_step):
    """ Save model """
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    checkpoint_path = os.path.join(model_save_path, 'vector')
    tf.logging.info('saving model %s.', checkpoint_path)
    tf.logging.info('global_step %i.', global_step)
    saver.save(sess, checkpoint_path, global_step=global_step)


def summ_content(tag, val):
    """ Construct summary content """
    summ = tf.summary.Summary()
    summ.value.add(tag=tag, simple_value=float(val))
    return summ


def write_summary(summ_writer, summ_dict, step):
    """ Write summary """
    for key, val in summ_dict.iteritems():
        summ_writer.add_summary(summ_content(key, val), step)
    summ_writer.flush()


def augment_strokes(strokes, prob=0.0):
    """ Perform data augmentation by randomly dropping out strokes """
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = list(candidate)
            prev_stroke = list(stroke)
            result.append(stroke)
    return np.array(result)


def seq_3d_to_5d(stroke, max_len=250):
    """ Convert from 3D format (npz file) to 5D (sketch-rnn paper) """
    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result


def seq_5d_to_3d(big_stroke):
    """ Convert from 5D format (sketch-rnn paper) back to 3D (npz file) """
    l = 0  # the total length of the drawing
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
        if l == 0:
            l = len(big_stroke)  # restrict the max total length of drawing to be the length of big_stroke
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result  # stroke-3


def get_max_len(strokes):
    """ Return the maximum length of an array of strokes """
    max_len = 0
    max_stroke_len=0
    max_strokes_num = 0
    maxset = []
    max_sn = []
    
    for stroke in strokes:
        ml = len(stroke)
        if ml > max_len:
            max_len = ml
        st = np.split(stroke[:, :2], np.where(stroke[:, 2])[0] + 1, axis=0)[:-1]
        msn = len(st)
        if msn > max_strokes_num:
            max_strokes_num = msn
        for ss in st:
            msl = len(ss)
            if msl > max_stroke_len:
                max_stroke_len = msl
            maxset.append(msl)
        max_sn.append(len(st))
    s= np.sort(maxset)
    sn = np.sort(max_sn)
    print(s[int(0.999 * len(s))])
    print(sn[int(0.999 * len(sn))])
    max_stroke_len = int(s[int(0.999 * len(s))])
    max_strokes_num = int(sn[int(0.999 * len(sn))])
    #ll rev 999
    #exit()
    return max_len,max_stroke_len,max_strokes_num


def rescale(X, ratio=0.85):
    """ Rescale the image to a smaller size """
    h, w = X.shape

    h2 = int(h * ratio)
    w2 = int(w * ratio)

    X2 = cv2.resize(X, (w2, h2), interpolation=cv2.INTER_AREA)

    dh = int((h - h2) / 2)
    dw = int((w - w2) / 2)

    res = np.copy(X)
    res[:, :] = 1
    res[dh:(dh + h2), dw:(dw + w2)] = X2

    return res


def rotate(X, angle=15):
    """ Rotate the image """
    h, w = X.shape
    rad = np.deg2rad(angle)

    nw = ((abs(np.sin(rad) * h)) + (abs(np.cos(rad) * w)))
    nh = ((abs(np.cos(rad) * h)) + (abs(np.sin(rad) * w)))

    rot_mat = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1)
    rot_move = np.dot(rot_mat, np.array([(nw - w) / 2, (nh - h) / 2, 0]))

    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

    res_w = int(math.ceil(nw))
    res_h = int(math.ceil(nh))

    res = cv2.warpAffine(X, rot_mat, (res_w, res_h), flags=cv2.INTER_LANCZOS4, borderValue=1)
    res = cv2.resize(res, (w, h), interpolation=cv2.INTER_AREA)

    return res


def translate(X, dx=5, dy=5):
    """ Translate the image """
    h, w = X.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    res = cv2.warpAffine(X, M, (w, h), borderValue=1)

    return res



class DataLoader(object):
    """ Class for loading data from raw data (sequence and image) """

    def __init__(self,
                 strokes,
                 images,
                 labels,
                 batch_size=100,
                 max_seq_length=250,
                 max_strokes_len=250,
                 max_strokes_num=250,
                 scale_factor=1.0,
                 random_scale_factor=0.0,
                 augment_stroke_prob=0.0,
                 png_scale_ratio=1,
                 png_rotate_angle=0,
                 png_translate_dist=0,
                 limit=1000):
        self.batch_size = batch_size  # minibatch size
        self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
        self.scale_factor = scale_factor  # divide data by this factor
        self.random_scale_factor = random_scale_factor  # data augmentation method
        self.limit = limit  # removes large gaps in the data
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
        self.png_scale_ratio = png_scale_ratio  # min randomly scaled ratio
        self.png_rotate_angle = png_rotate_angle  # max randomly rotate angle (in absolute value)
        self.png_translate_dist = png_translate_dist  # max randomly translate distance (in absolute value)
        self.start_stroke_token = [0, 0, 1, 0, 0]  # S_0 in sketch-rnn paper
        self.preprocess(strokes, images)
        self.labels = labels
        self.max_strokes_len = max_strokes_len
        self.max_strokes_num = max_strokes_num
        

    def preprocess(self, strokes, images):
        # preprocess stroke data
        self.strokes = []
        count_data = 0  # the number of drawing with length less than N_max

        for i in range(len(strokes)):
            data = np.copy(strokes[i])
            if len(data) <= self.max_seq_length:  # keep data with length less than N_max
                count_data += 1
                # removes large gaps from the data
                data = np.minimum(data, self.limit)  # prevent large values
                data = np.maximum(data, -self.limit)  # prevent small values
                data = np.array(data, dtype=np.float32)  # change data type
                data[:, 0:2] /= self.scale_factor  # scale the first two dims of data
                self.strokes.append(data)

        print("total sequences <= max_seq_len is %d" % count_data)
        self.num_batches = int(count_data / self.batch_size)

        # preprocess image data
        self.images = []

        '''for i in range(len(self.strokes)):
            mul_p = self.re_th_png(self.strokes[i])
            self.strokes[i][:,:2] *= mul_p
        print("total png images %d" % len(self.images))'''





    def random_sample(self):
        """ Return a random sample (3D stroke, png image) """
        l = len(self.strokes)
        idx = np.random.randint(0, l)
        seq = self.strokes[idx]
        png = self.images[idx]
        label = self.labels[idx]
        png = png.reshape((1, png.shape[0], png.shape[1]))
        return seq, png, label

    def idx_sample(self, idx):
        """ Return one sample by idx """
        data = self.random_scale_seq(self.strokes[idx])
        if self.augment_stroke_prob > 0:
            data = augment_strokes(data, self.augment_stroke_prob)
        strokes_3d = data
        strokes_5d = seq_3d_to_5d(strokes_3d, self.max_seq_length)

        data = np.copy(self.images[idx])
        png = np.reshape(data, [1, data.shape[0], data.shape[1]])
        png = self.random_scale_png(png)
        png = self.random_rotate_png(png)
        png = self.random_translate_png(png)
        label = self.labels[idx]
        return strokes_5d, png, label

    def random_scale_seq(self, data):
        """ Augment data by stretching x and y axis randomly [1-e, 1+e] """
        x_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        #result = data.copy()
        result = np.copy(data)
        #
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result

    def random_scale_png(self, data):
        """ Randomly scale image """
        out_pngs = np.copy(data)
        for i in range(data.shape[0]):
            in_png = data[i]
            ratio = random.uniform(self.png_scale_ratio, 1)
            out_png = rescale(in_png, ratio)
            out_pngs[i] = out_png
        return out_pngs

    def random_rotate_png(self, data):
        """ Randomly rotate image """
        out_pngs = np.copy(data)
        for i in range(data.shape[0]):
            in_png = data[i]
            angle = random.uniform(-self.png_rotate_angle, self.png_rotate_angle)
            out_png = rotate(in_png, angle)
            out_pngs[i] = out_png
        return out_pngs

    def random_translate_png(self, data):
        """ Randomly translate image """
        out_pngs = np.copy(data)
        for i in range(data.shape[0]):
            in_png = data[i]
            dx = random.uniform(-self.png_translate_dist, self.png_translate_dist)
            dy = random.uniform(-self.png_translate_dist, self.png_translate_dist)
            out_png = translate(in_png, dx, dy)
            out_pngs[i] = out_png
        return out_pngs

    def calc_seq_norm(self):
        """ Calculate the normalizing factor explained in appendix of sketch-rnn """
        data = []
        for i in range(len(self.strokes)):
            if len(self.strokes[i]) > self.max_seq_length:
                continue
            for j in range(len(self.strokes[i])):
                data.append(self.strokes[i][j, 0])
                data.append(self.strokes[i][j, 1])
        data = np.array(data)
        return np.std(data)  # standard dev of all the delta x and delta y in the datasets
    def calc_abs_seq_norm(self):
        """ Calculate the normalizing factor explained in appendix of sketch-rnn """
        data = []
        dt1,dt2 = 0,0
        for i in range(len(self.strokes)):
            for j in range(len(self.strokes[i])):
                data.append(self.strokes[i][j, 0]+dt1)
                data.append(self.strokes[i][j, 1]+dt2)
                dt1 += self.strokes[i][j, 0]
                dt2 += self.strokes[i][j, 1]
        data = np.array(data)
        #print(data)
        return np.std(data)  # standard dev of all the delta x and delta y in the datasets
    def normalize_seq(self, scale_factor=None):
        """ Normalize entire sequence dataset (delta_x, delta_y) by the scaling factor."""
        if scale_factor is None:
            scale_factor = self.calc_seq_norm()
        self.scale_factor = scale_factor
        for i in range(len(self.strokes)):
            self.strokes[i][:, 0:2] /= (self.scale_factor+1e-10)

    def re_th_png(self,seq):
        min_x, max_x, min_y, max_y = seq2png.get_bounds(seq)
        #print(min_x, max_x, min_y, max_y)
        #min_x, max_x, min_y, max_y = min_x *self.scale_factor, max_x*self.scale_factor, min_y*self.scale_factor, max_y*self.scale_factor
        min_x, max_x, min_y, max_y = min_x +half_width, max_x +half_width, min_y +half_width, max_y +half_width
        min_mul = 10.
        min_mul = min(min_mul,min(half_width/(max_x+1e-10),half_width/(max_y+1e-10)))
        min_mul = min(min_mul,min(half_width/(half_width-min_x+1e-10),half_width/(half_width-min_y+1e-10)))
        return min_mul

    def mydrawPNG_from_list(self,vector_image, Side=width):
        #whole_sub = []

        raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)
        #whole_ra = np.zeros(int(Side), int(Side)), dtype=np.float32)
        np.insert(vector_image[0],0,[0,0],0)
        for stroke in vector_image:

            initX, initY = int(stroke[0, 0]), int(stroke[0, 1])

            for i_pos in range(1, len(stroke)):


                cordList = list(bresenham(initX, initY, int(stroke[i_pos, 0]), int(stroke[i_pos, 1])))
                for cord in cordList:
                    if (cord[0] > 0 and cord[1] > 0) and (cord[0] < Side and cord[1] < Side):
                        raster_image[cord[1], cord[0]] = 255.0

                initX, initY = int(stroke[i_pos, 0]), int(stroke[i_pos, 1])
        #raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0


        return np.reshape(raster_image,[Side,Side])

    def stroke_pbias(self,seq_t):
        seq = seq_t.copy()
        min_x, max_x, min_y, max_y = seq2png.get_bounds(seq)
        min_x, max_x, min_y, max_y = min_x*self.scale_factor, max_x*self.scale_factor, min_y*self.scale_factor, max_y*self.scale_factor
        min_x, max_x, min_y, max_y = min_x+half_width, max_x+half_width, min_y+half_width, max_y+half_width
        x_bound_n = -min_x*0.8
        x_bound_p = (width-max_x)*0.8
        y_bound_n = -min_y*0.8
        y_bound_p = (width-max_y)*0.8
        #y_bound = min(width-max_y,min_y)*0.8
        x_bias = np.random.uniform(x_bound_n,x_bound_p)
        y_bias = np.random.uniform(y_bound_n,y_bound_p)

        return (x_bias)/self.scale_factor,(y_bias)/self.scale_factor

    def flush_strokes(self,seq_t):
        seq = seq_t.copy()
        seq = np.split(seq[:, :], np.where(seq[:, 2])[0] + 1, axis=0)[:-1]
        r_s = []
        p = np.random.rand(len(seq))
        for i in range(len(seq)):
            index  = np.argmax(p)
            p[index] = -1000
            if r_s == []:
                r_s = seq[index]
            else:
                r_s = np.concatenate([r_s,seq[index]],axis=0)
        return r_s

    def get_relgrid(self,min_x,max_x,min_y,max_y,xt,yt):
        x, y = 0, 0
        if xt < min_x:
            x = 0
        elif xt > max_x:
            x = 2
        else:
            x = 1
        if yt < min_y:
            y = 0
        elif yt > max_y:
            y = 2
        else:
            y = 1
        return x,y

    def get_box(self,data):
        data = data.copy()
        box = np.zeros([self.max_strokes_num,self.max_strokes_num,3,3])
        for i in range(min(self.max_strokes_num,len(data))):
            ds = data[i]
            
            min_x, max_x, min_y, max_y = seq2png.get_bounds_whole(ds)
            for j in range(min(self.max_strokes_num,len(data))):
                dt = data[j]
                min_x2, max_x2, min_y2, max_y2 = seq2png.get_bounds_whole(dt)
                #left down
                x_ld,y_ld = self.get_relgrid(min_x, max_x, min_y, max_y,min_x2,min_y2)

                x_rd, y_rd = self.get_relgrid(min_x, max_x, min_y, max_y, max_x2, min_y2)
                x_lu, y_lu = self.get_relgrid(min_x, max_x, min_y, max_y, min_x2, max_y2)
                x_ru, y_ru = self.get_relgrid(min_x, max_x, min_y, max_y, max_x2, max_y2)
                box[i,j,x_ld,y_ld:y_lu+1] = 1
                box[i, j, x_ld:x_rd+1, y_lu] = 1
                box[i, j, x_ru , y_ld:y_lu+1] = 1
                box[i, j, x_ld:x_rd + 1, y_rd] = 1
        return box


    def _get_batch_from_indices(self, indices,PROB=0.):
        """Given a list of indices, return the potentially augmented batch."""
        seq_batch = []
        png_batch = []
        label_batch = []
        seq_len = []
        seq_bias_batch = []
        png_bias_batch = []
        mask_batch = []
        stroke_num = []
        stroke_len = []
        box_batch = []
        seq_ori_batch = []
        box_ori_batch = []
        #PROB = np.random.randint(0,10)/10
        for idx in range(len(indices)):
            mask = np.zeros([self.max_seq_length+1,1])
            
            i = indices[idx]
            if self.augment_stroke_prob > 0:
                data = self.random_scale_seq(self.strokes[i])
            else:
                data = self.strokes[i]
            min_x, max_x, min_y, max_y = seq2png.get_bounds(data)
            bound = max(max_x-min_x,max_y-min_y)
            data_copy = data.copy()
            if self.augment_stroke_prob > 0:
                data_copy = augment_strokes(data_copy, self.augment_stroke_prob)
            if self.augment_stroke_prob > 0:
                x_bias,y_bias = self.stroke_pbias(data_copy)
                data_copy[0,0] += x_bias
                data_copy[0,1] += y_bias
            #data_copy[:,:2] *= self.scale_factor
            #打乱笔画
            absx , absy =0,0
            for i in range(len(data_copy) - 1):
                absx +=data_copy[i,0]
                absy +=data_copy[i,1]
                if i!=0 and data_copy[i-1,-1]==1 :
                    data_copy[i,0] =absx
                    data_copy[i,1] = absy
            rnd = random.uniform(0,10)/10
            #if rnd< 0.:
            #data_copy = self.flush_strokes(data_copy)
            absx , absy =0,0
            for i in range(len(data_copy) - 1):
                absx +=data_copy[i,0]
                absy +=data_copy[i,1]
                if i!=0 and data_copy[i,-1]==1:
                    data_copy[i+1,0] -=absx
                    data_copy[i+1,1] -= absy

            #abs part
            data_ori = data_copy.copy()
            for i in range(len(data_ori) - 1):
                data_ori[i+1,0] +=data_ori[i,0]
                data_ori[i+1,1] +=data_ori[i,1]

            data_abs = data_copy.copy()
            abs_x , abs_y = data_abs[0,0],data_abs[0,1]
            for i in range(len(data_abs) - 1):
                abs_x +=data_abs[i+1,0]
                abs_y +=data_abs[i+1,1]
                if  data_copy[i,-1]==1 :
                    data_abs[i+1,0] = abs_x+np.random.randn(1) *bound*PROB
                    data_abs[i+1,1] = abs_y+np.random.randn(1) *bound*PROB
            
            '''abs_x,abs_y = 0,0
            for i in range(len(data_abs) - 1):
                
                if  i!=0 and data_abs[i-1,-1]==1:
                    abs_x += data_abs[i,0]
                    abs_y += data_abs[i,1]
                else:
                    abs_x  = data_abs[i,0]
                    abs_y  = data_abs[i,1]
                
                data_abs[i,0] = abs_x
                data_abs[i,1] = abs_y'''

            seq_ori_batch.append(data_ori)
            seq_batch.append(data_ori)
            length = len(data_copy)
            seq_len.append(length)
            data_copy2 = data_copy.copy()
            data_copy2[:, :2] *= self.scale_factor
            
            for i in range(len(data_copy2) - 1):
                data_copy2[i + 1, :2] += data_copy2[i, :2]
            
            data_copy2[:, :2] += half_width
            #data_copy2 = self.stroke_pbias(data_copy2)
            #data_copy3 = np.split(data_copy2[:, :2], np.where(data_copy2[:, 2])[0] + 1, axis=0)[:-1]
            #png_mod = self.mydrawPNG_from_list(data_copy3) / 255.0 * 2. -1.
            #png_batch.append(png_mod)
            #bias part
            #i = indices[idx]
            #data = self.random_scale_seq(self.strokes[i])
            '''data_copy = data_copy.copy()
            absx , absy =0,0
            for i in range(len(data_copy) - 1):
                absx +=data_copy[i,0]
                absy +=data_copy[i,1]
                p = np.random.rand(1)
                #print(p)
                data_copy[i,0] =absx
                data_copy[i,1] = absy'''
            '''absx , absy =0,0
            for i in range(len(data_copy) - 1):
                absx +=data_copy[i,0]
                absy +=data_copy[i,1]
                if i!=0 and data_copy[i,-1]==1:
                    data_copy[i+1,0] -=absx
                    data_copy[i+1,1] -= absy'''
            #seq_bias_batch.append(data_copy)
            data_copy = data_abs.copy()

            #abs part train
            absx , absy =0,0
            for i in range(len(data_copy)):
                if i!=0 and data_copy[i-1,-1]==1:
                    absx,absy = data_copy[i,0],data_copy[i,1]
                else:
                    absx += data_copy[i,0]
                    absy += data_copy[i,1]
                data_copy[i,0] = absx
                data_copy[i,1] = absy
            '''absx , absy =0,0
            for i in range(len(data_copy) - 1):
                data_copy[i+1,0] -= data_copy[i,0]
                data_copy[i+1,1] -= data_copy[i,1]
            for i in range(len(data_copy) - 1):
                absx +=data_copy[i,0]
                absy +=data_copy[i,1]
                if i!=0 and data_copy[i,-1]==1:
                    data_copy[i+1,0] =absx
                    data_copy[i+1,1] = absy'''

            #new version
            cnt = 0
            stroke_set = []
            stroke=[]
            for ii in range(len(data_copy)):
                stroke.append(data_copy[ii,:2])
        
                if data_copy[ii,-1]==1:
                    stroke = np.array(stroke)
                    for iii in range(len(stroke)-1):
                        stroke[len(stroke)-iii-1,:] = stroke[len(stroke)-iii-1,:] - stroke[len(stroke)-iii-2,:]
                    stroke_set.append(stroke)
                    stroke=[]
                    cnt=0
                elif cnt==self.max_strokes_len:
                    stroke = np.array(stroke)
                    for iii in range(len(stroke)-1):
                        stroke[len(stroke)-iii-1,:] = stroke[len(stroke)-iii-1,:] - stroke[len(stroke)-iii-2,:]
                    stroke_set.append(stroke)
                    stroke=[]
                    cnt=0
                else:
                    cnt+=1
            box = self.get_box(stroke_set)
            #需要rel混abs格式
            box_batch.append(box)
            stroke_num.append(min(self.max_strokes_num,len(stroke_set)))
            if len(stroke_set) < self.max_strokes_num:
                for si in stroke_set:
                    for iii in range(len(si)-1):
                        si[iii+1,:] = si[iii+1,:] + si[iii,:]

                    #si[:,0] =  si[:,0] - si[0,0]
                    #si[:,1] =  si[:,1] - si[0,1]
                    s = self.pad_stroke(si,len(si),self.max_strokes_len)
                    #s[0,:] = 0
                    seq_bias_batch.append(s)
                    stroke_len.append((len(si)))
            else:
                for si in range(self.max_strokes_num):
                    for iii in range(len(stroke_set[si])-1):
                        stroke_set[si][iii+1,:] = stroke_set[si][iii+1,:] + stroke_set[si][iii,:]

                    #data_copy[si][:,0] =  data_copy[si][:,0] - data_copy[si][0,0]
                    #data_copy[si][:,1] =  data_copy[si][:,1] - data_copy[si][0,1]
                    s = self.pad_stroke(stroke_set[si],len(stroke_set[si]),self.max_strokes_len)
                    #s[0,:] = 0
                    seq_bias_batch.append(s)
                    stroke_len.append(min(self.max_strokes_len,len(stroke_set[si])))



            '''data_copy = np.split(data_copy[:, :2], np.where(data_copy[:, 2])[0] + 1, axis=0)[:-1]
            #if self.augment_stroke_prob > 0:
            #    np.random.shuffle(data_copy)
            #打乱动作
            box = self.get_box(data_copy)
            #需要rel-abs格式
            box_batch.append(box)
            stroke_num.append(min(self.max_strokes_num,len(data_copy)))
            if len(data_copy) < self.max_strokes_num:
                for si in data_copy:

                    #si[:,0] =  si[:,0] - si[0,0]
                    #si[:,1] =  si[:,1] - si[0,1]
                    s = self.pad_stroke(si,len(si),self.max_strokes_len)
                    #s[0,:] = 0
                    seq_bias_batch.append(s)
                    stroke_len.append((len(si)))
            else:
                for si in range(self.max_strokes_num):

                    #data_copy[si][:,0] =  data_copy[si][:,0] - data_copy[si][0,0]
                    #data_copy[si][:,1] =  data_copy[si][:,1] - data_copy[si][0,1]
                    s = self.pad_stroke(data_copy[si],len(data_copy[si]),self.max_strokes_len)
                    #s[0,:] = 0
                    seq_bias_batch.append(s)
                    stroke_len.append(min(self.max_strokes_len,len(data_copy[si])))'''

            #length = len(data_copy)
            #seq_len.append(length)

            mask_batch.append(mask)
            #label_batch.append(self.labels[i])

            #new
            data_ori_copy = data_ori.copy()
            cnt = 0
            stroke_set = []
            stroke=[]
            for ii in range(len(data_ori_copy)):
                stroke.append(data_ori_copy[ii,:2])
        
                if data_ori_copy[ii,-1]==1:
                    stroke = np.array(stroke)
                    for iii in range(len(stroke)-1):
                        stroke[len(stroke)-iii-1,:] = stroke[len(stroke)-iii-1,:] - stroke[len(stroke)-iii-2,:]
                    stroke_set.append(stroke)
                    stroke=[]
                    cnt=0
                elif cnt==self.max_strokes_len:
                    stroke = np.array(stroke)
                    for iii in range(len(stroke)-1):
                        stroke[len(stroke)-iii-1,:] = stroke[len(stroke)-iii-1,:] - stroke[len(stroke)-iii-2,:]
                    stroke_set.append(stroke)
                    stroke=[]
                    cnt=0
                else:
                    cnt+=1

            box_ori = self.get_box(stroke_set)
            box_ori_batch.append(box_ori)

            '''#box_ori设置
            data_ori_copy = data_ori.copy()
            absx , absy =data_ori_copy[0,0],data_ori_copy[0,1]
            for i in range(len(data_ori_copy) - 1):
                data_ori_copy[len(data_ori_copy) - 1 - i,0] -= data_ori_copy[len(data_ori_copy) - 2 - i,0]
                data_ori_copy[len(data_ori_copy) - 1 - i,1] -= data_ori_copy[len(data_ori_copy) - 2 - i,1]
            for i in range(len(data_ori_copy) - 1):
                absx +=data_ori_copy[i+1,0]
                absy +=data_ori_copy[i+1,1]
                if i!=0 and data_ori_copy[i,-1]==1:
                    data_ori_copy[i+1,0] =absx
                    data_ori_copy[i+1,1] = absy
            data_ori_copy = np.split(data_ori_copy[:, :2], np.where(data_ori_copy[:, 2])[0] + 1, axis=0)[:-1]
            #np.random.shuffle(data_copy)
            #打乱动作
            box_ori = self.get_box(data_ori_copy)'''
            #需要rel格式
            #box_ori_batch.append(box_ori)
        
        seq_len = np.array(seq_len, dtype=int)

        #png_bias_batch = np.array(png_bias_batch)
        mask_batch = np.array(mask_batch)
        stroke_len = np.array(stroke_len)
        stroke_num = np.array(stroke_num)
        box_batch = np.reshape(box_batch,[self.batch_size,self.max_strokes_num,-1,9])
        box_ori_batch = np.reshape(box_ori_batch,[self.batch_size,self.max_strokes_num,-1,9])
        seq_bias_batch = np.array(seq_bias_batch)
        seq_len = np.array(seq_len, dtype=int)
        return self.pad_seq_batch(seq_batch, self.max_seq_length), png_batch, label_batch, seq_len,seq_bias_batch,stroke_len,stroke_num,mask_batch,box_batch,box_ori_batch
        #seq_batch 是真实abs格式，seq_bias 是输入rel storke格式， box是打乱格式，box_ori是原始格式ori，使用同一个seq_bias

    def get_box_input(self,seqs):
        
        seq_batch = []
        png_batch = []
        label_batch = []
        seq_len = []
        seq_bias_batch = []
        png_bias_batch = []
        mask_batch = []
        stroke_num = []
        stroke_len = []
        box_batch = []
        #PROB = np.random.randint(0,10)/10
        PROB = 0
        for idx in range(len(seqs)):
            mask = np.zeros([self.max_seq_length+1,1])
            
            
            data = seqs[idx,:,:]
            data = seq_5d_to_3d(data)
            data = data[1:,:]
            #print(data.shape)
            min_x, max_x, min_y, max_y = seq2png.get_bounds(data)
            bound = max(max_x-min_x,max_y-min_y)
            data_copy = data.copy()
            
            #data_copy[:,:2] *= self.scale_factor
            #打乱笔画
            absx , absy =0,0
            for i in range(len(data_copy) - 1):
                absx +=data_copy[i,0]
                absy +=data_copy[i,1]
                if i!=0 and data_copy[i-1,-1]==1 :
                    data_copy[i,0] =absx
                    data_copy[i,1] = absy
            rnd = random.uniform(0,10)/10
            #if rnd< 0.:
            #data_copy = self.flush_strokes(data_copy)
            absx , absy =0,0
            for i in range(len(data_copy) - 1):
                absx +=data_copy[i,0]
                absy +=data_copy[i,1]
                if i!=0 and data_copy[i,-1]==1:
                    data_copy[i+1,0] -=absx
                    data_copy[i+1,1] -= absy

            #abs part
            data_ori = data_copy.copy()
            for i in range(len(data_ori) - 1):
                data_ori[i+1,0] +=data_ori[i,0]
                data_ori[i+1,1] +=data_ori[i,1]

            data_abs = data_copy.copy()
            abs_x , abs_y = data_abs[0,0],data_abs[0,1]
            for i in range(len(data_abs) - 1):
                abs_x +=data_abs[i+1,0]
                abs_y +=data_abs[i+1,1]
                if  data_copy[i,-1]==1 :
                    data_abs[i+1,0] = abs_x+np.random.randn(1) *bound*PROB
                    data_abs[i+1,1] = abs_y+np.random.randn(1) *bound*PROB
            
            '''abs_x,abs_y = 0,0
            for i in range(len(data_abs) - 1):
                
                if  i!=0 and data_abs[i-1,-1]==1:
                    abs_x += data_abs[i,0]
                    abs_y += data_abs[i,1]
                else:
                    abs_x  = data_abs[i,0]
                    abs_y  = data_abs[i,1]
                
                data_abs[i,0] = abs_x
                data_abs[i,1] = abs_y'''

#            seq_ori_batch.append(data_ori)
            seq_batch.append(data_ori)
            length = len(data_copy)
            seq_len.append(length)
            data_copy2 = data_copy.copy()
            data_copy2[:, :2] *= self.scale_factor
            
            for i in range(len(data_copy2) - 1):
                data_copy2[i + 1, :2] += data_copy2[i, :2]
            
            data_copy2[:, :2] += half_width
            #data_copy2 = self.stroke_pbias(data_copy2)
            #data_copy3 = np.split(data_copy2[:, :2], np.where(data_copy2[:, 2])[0] + 1, axis=0)[:-1]
            #png_mod = self.mydrawPNG_from_list(data_copy3) / 255.0 * 2. -1.
            #png_batch.append(png_mod)
            #bias part
            #i = indices[idx]
            #data = self.random_scale_seq(self.strokes[i])
            '''data_copy = data_copy.copy()
            absx , absy =0,0
            for i in range(len(data_copy) - 1):
                absx +=data_copy[i,0]
                absy +=data_copy[i,1]
                p = np.random.rand(1)
                #print(p)
                data_copy[i,0] =absx
                data_copy[i,1] = absy'''
            '''absx , absy =0,0
            for i in range(len(data_copy) - 1):
                absx +=data_copy[i,0]
                absy +=data_copy[i,1]
                if i!=0 and data_copy[i,-1]==1:
                    data_copy[i+1,0] -=absx
                    data_copy[i+1,1] -= absy'''
            #seq_bias_batch.append(data_copy)
            data_copy = data_abs.copy()

            #abs part train
            absx , absy =0,0
            for i in range(len(data_copy)):
                if i!=0 and data_copy[i-1,-1]==1:
                    absx,absy = data_copy[i,0],data_copy[i,1]
                else:
                    absx += data_copy[i,0]
                    absy += data_copy[i,1]
                data_copy[i,0] = absx
                data_copy[i,1] = absy
            '''absx , absy =0,0
            for i in range(len(data_copy) - 1):
                data_copy[i+1,0] -= data_copy[i,0]
                data_copy[i+1,1] -= data_copy[i,1]
            for i in range(len(data_copy) - 1):
                absx +=data_copy[i,0]
                absy +=data_copy[i,1]
                if i!=0 and data_copy[i,-1]==1:
                    data_copy[i+1,0] =absx
                    data_copy[i+1,1] = absy'''

            #new version
            cnt = 0
            stroke_set = []
            stroke=[]
            for ii in range(len(data_copy)):
                stroke.append(data_copy[ii,:2])
        
                if data_copy[ii,-1]==1:
                    stroke = np.array(stroke)
                    for iii in range(len(stroke)-1):
                        stroke[len(stroke)-iii-1,:] = stroke[len(stroke)-iii-1,:] - stroke[len(stroke)-iii-2,:]
                    stroke_set.append(stroke)
                    stroke=[]
                    cnt=0
                elif cnt==self.max_strokes_len:
                    stroke = np.array(stroke)
                    for iii in range(len(stroke)-1):
                        stroke[len(stroke)-iii-1,:] = stroke[len(stroke)-iii-1,:] - stroke[len(stroke)-iii-2,:]
                    stroke_set.append(stroke)
                    stroke=[]
                    cnt=0
                else:
                    cnt+=1
            box = self.get_box(stroke_set)
            #需要rel混abs格式
            box_batch.append(box)
            stroke_num.append(min(self.max_strokes_num,len(stroke_set)))
            if len(stroke_set) < self.max_strokes_num:
                for si in stroke_set:
                    for iii in range(len(si)-1):
                        si[iii+1,:] = si[iii+1,:] + si[iii,:]
                    #si[:,0] =  si[:,0] - si[0,0]
                    #si[:,1] =  si[:,1] - si[0,1]
                    s = self.pad_stroke(si,len(si),self.max_strokes_len)
                    #s[0,:] = 0
                    seq_bias_batch.append(s)
                    stroke_len.append((len(si)))
            else:
                for si in range(self.max_strokes_num):
                    for iii in range(len(stroke_set[si])-1):
                        stroke_set[si][iii+1,:] = stroke_set[si][iii+1,:] + stroke_set[si][iii,:]

                    #data_copy[si][:,0] =  data_copy[si][:,0] - data_copy[si][0,0]
                    #data_copy[si][:,1] =  data_copy[si][:,1] - data_copy[si][0,1]
                    s = self.pad_stroke(stroke_set[si],len(stroke_set[si]),self.max_strokes_len)
                    #s[0,:] = 0
                    seq_bias_batch.append(s)
                    stroke_len.append(min(self.max_strokes_len,len(stroke_set[si])))

            '''data_copy[-1,-1]=1
            data_copy = np.split(data_copy[:, :2], np.where(data_copy[:, 2])[0] + 1, axis=0)[:-1]
            #if self.augment_stroke_prob > 0:
            #    np.random.shuffle(data_copy)
            #打乱动作
            box = self.get_box(data_copy)
            #需要rel混abs格式
            box_batch.append(box)
            stroke_num.append(min(self.max_strokes_num,len(data_copy)))
            if len(data_copy) < self.max_strokes_num:
                for si in data_copy:

                    #si[:,0] =  si[:,0] - si[0,0]
                    #si[:,1] =  si[:,1] - si[0,1]
                    s = self.pad_stroke(si,len(si),self.max_strokes_len)
                    #s[0,:] = 0
                    seq_bias_batch.append(s)
                    stroke_len.append((len(si)))
            else:
                for si in range(self.max_strokes_num):

                    #data_copy[si][:,0] =  data_copy[si][:,0] - data_copy[si][0,0]
                    #data_copy[si][:,1] =  data_copy[si][:,1] - data_copy[si][0,1]
                    s = self.pad_stroke(data_copy[si],len(data_copy[si]),self.max_strokes_len)
                    #s[0,:] = 0
                    seq_bias_batch.append(s)
                    stroke_len.append(min(self.max_strokes_len,len(data_copy[si])))'''
            


            #length = len(data_copy)
            #seq_len.append(length)

            mask_batch.append(mask)
            #label_batch.append(self.labels[i])
        seq_len = np.array(seq_len, dtype=int)

        #png_bias_batch = np.array(png_bias_batch)
        mask_batch = np.array(mask_batch)
        stroke_len = np.array(stroke_len)
        stroke_num = np.array(stroke_num)
        box_batch = np.reshape(box_batch,[self.batch_size,self.max_strokes_num,-1,9])
        seq_bias_batch = np.array(seq_bias_batch)
        seq_len = np.array(seq_len, dtype=int)
        return self.pad_seq_batch(seq_batch, self.max_seq_length), png_batch, label_batch, seq_len,seq_bias_batch,stroke_len,stroke_num,mask_batch,box_batch

    def _get_batch_from_indices_extra(self, indices,PROB=0.):
        """Given a list of indices, return the potentially augmented batch."""
        seq_batch = []
        png_batch = []
        label_batch = []
        seq_len = []
        seq_bias_batch = []
        seq_bias_batch_2 = []
        png_bias_batch = []
        mask_batch = []
        stroke_num = []
        stroke_len = []
        box_batch = []
        #PROB = np.random.randint(0,10)/10
        for idx in range(len(indices)):
            mask = np.zeros([self.max_seq_length+1,1])
            
            i = indices[idx]
            data = self.random_scale_seq(self.strokes[i])
            min_x, max_x, min_y, max_y = seq2png.get_bounds(data)
            bound = max(max_x-min_x,max_y-min_y)
            data_copy = data.copy()
            if self.augment_stroke_prob > 0:
                data_copy = augment_strokes(data_copy, self.augment_stroke_prob)
            if self.augment_stroke_prob > 0:
                x_bias,y_bias = self.stroke_pbias(data_copy)
                data_copy[0,0] += x_bias
                data_copy[0,1] += y_bias
            #data_copy[:,:2] *= self.scale_factor
            absx , absy =0,0
            for i in range(len(data_copy) - 1):
                absx +=data_copy[i,0]
                absy +=data_copy[i,1]
                if i!=0 and data_copy[i-1,-1]==1 :
                    data_copy[i,0] =absx
                    data_copy[i,1] = absy
            #data_copy = self.flush_strokes(data_copy)
            absx , absy =0,0
            for i in range(len(data_copy) - 1):
                absx +=data_copy[i,0]
                absy +=data_copy[i,1]
                if i!=0 and data_copy[i,-1]==1:
                    data_copy[i+1,0] -=absx
                    data_copy[i+1,1] -= absy

            #abs part
            data_abs = data_copy.copy()
            for i in range(len(data_abs) - 1):
                data_abs[i+1,0] +=data_abs[i,0]
                data_abs[i+1,1] +=data_abs[i,1]
            seq_batch.append(data_copy)
            length = len(data_copy)
            seq_len.append(length)
            data_copy2 = data_copy.copy()
            data_copy2[:, :2] *= self.scale_factor
            
            for i in range(len(data_copy2) - 1):
                data_copy2[i + 1, :2] += data_copy2[i, :2]
            
            data_copy2[:, :2] += half_width
            #data_copy2 = self.stroke_pbias(data_copy2)
            #data_copy3 = np.split(data_copy2[:, :2], np.where(data_copy2[:, 2])[0] + 1, axis=0)[:-1]
            #png_mod = self.mydrawPNG_from_list(data_copy3) / 255.0 * 2. -1.
            #png_batch.append(png_mod)
            #bias part
            #i = indices[idx]
            #data = self.random_scale_seq(self.strokes[i])
            data_copy = data_copy.copy()
            absx , absy =0,0
            for i in range(len(data_copy) - 1):
                absx +=data_copy[i,0]
                absy +=data_copy[i,1]
                p = np.random.rand(1)
                #print(p)
                if i!=0 and data_copy[i-1,-1]==1 :
                    if p<1:
                    #print(p)
                        data_copy[i,0] =absx+ np.random.randn(1) *bound*0.3
                        data_copy[i,1] = absy+np.random.randn(1) *bound*0.3
                    else:
                        data_copy[i,0] =absx
                        data_copy[i,1] = absy
                    mask[i+1,0] = 1
                else:
                    mask[i+1,0] = 0
            absx , absy =0,0
            for i in range(len(data_copy) - 1):
                absx +=data_copy[i,0]
                absy +=data_copy[i,1]
                if i!=0 and data_copy[i,-1]==1:
                    data_copy[i+1,0] -=absx
                    data_copy[i+1,1] -= absy
            seq_bias_batch_2.append(data_copy)
            
            data_copy = np.split(data_copy[:, :2], np.where(data_copy[:, 2])[0] + 1, axis=0)[:-1]
            np.random.shuffle(data_copy)
            box = self.get_box(data_copy)
            box_batch.append(box)
            stroke_num.append(min(self.max_strokes_num,len(data_copy)))
            if len(data_copy) < self.max_strokes_num:
                for si in data_copy:
                    s = self.pad_stroke(si,len(si),self.max_strokes_len)
                    s[0,:] = 0
                    seq_bias_batch.append(s)
                    stroke_len.append((len(si)))
            else:
                for si in range(self.max_strokes_num):
                    s = self.pad_stroke(data_copy[si],len(data_copy[si]),self.max_strokes_len)
                    s[0,:] = 0
                    seq_bias_batch.append(s)
                    stroke_len.append(min(self.max_strokes_len,len(data_copy[si])))

            #length = len(data_copy)
            #seq_len.append(length)

            mask_batch.append(mask)
            #label_batch.append(self.labels[i])
        seq_len = np.array(seq_len, dtype=int)

        #png_bias_batch = np.array(png_bias_batch)
        mask_batch = np.array(mask_batch)
        stroke_len = np.array(stroke_len)
        stroke_num = np.array(stroke_num)
        box_batch = np.reshape(box_batch,[self.batch_size,self.max_strokes_num,-1,9])
        seq_bias_batch = np.array(seq_bias_batch)
        seq_len = np.array(seq_len, dtype=int)
        return seq_bias_batch_2,self.pad_seq_batch(seq_batch, self.max_seq_length), png_batch, label_batch, seq_len,seq_bias_batch,stroke_len,stroke_num,mask_batch,box_batch

        

    def pad_stroke(self,stroke,len,max_len):
        stroke = stroke[:,:2]
        if len < max_len:
            stroke = np.concatenate((stroke,np.zeros([max_len-len,2])),axis=0)
        else:
            stroke = stroke[:max_len,:]

        return stroke
    def random_batch(self):
        """Return a randomised portion of the training data."""
        idxs = np.random.permutation(list(range(0, len(self.strokes))))[0:self.batch_size]
        return self._get_batch_from_indices(idxs)

    def get_batch(self, idx,PROB=0.):
        """Get the idx'th batch from the dataset."""
        assert idx >= 0, "idx must be non negative"
        assert idx < self.num_batches, "idx must be less than the number of batches"
        start_idx = idx * self.batch_size
        indices = list(range(start_idx, start_idx + self.batch_size))
        return self._get_batch_from_indices(indices,PROB)

    def pad_seq_batch(self, batch, max_len):
        """ Pad the batch to be 5D format, and fill the sequence to reach max_len """
        result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
        assert len(batch) == self.batch_size
        for i in range(self.batch_size):
            l = len(batch[i])
            assert l <= max_len
            result[i, 0:l, 0:2] = batch[i][:, 0:2]
            result[i, 0:l, 3] = batch[i][:, 2]
            result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
            result[i, l:, 4] = 1
            # put in the first token, as described in sketch-rnn methodology
            result[i, 1:, :] = result[i, :-1, :]
            result[i, 0, :] = 0
            result[i, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
            result[i, 0, 3] = self.start_stroke_token[3]
            result[i, 0, 4] = self.start_stroke_token[4]
        return result