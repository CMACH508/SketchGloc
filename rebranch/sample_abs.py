import sys
import random
import os
import json
import numpy as np
import tensorflow as tf
import utils
import glob
from PIL import Image
from seq2png import draw_strokes
from model import Model
import scipy.misc
import re
from svg2png import exportsvg
import matplotlib.image


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def sample(sess, sample_model, z, gen_size=1, seq_len=250, temperature=0.24, greedy_mode=False):
    """ Sample a sequence of strokes """

    def adjust_pdf(pi_pdf, temp):
        """ Adjust the pdf of pi according to temperature """
        pi_pdf = np.log(pi_pdf) / temp
        pi_pdf -= pi_pdf.max()
        pi_pdf = np.exp(pi_pdf)
        pi_pdf /= pi_pdf.sum()
        return pi_pdf


    def get_pi_idx(x, pdf, temp=1.0, greedy=False):
        """ Sample from a pdf, optionally greedily """
        if greedy:
            return np.argmax(pdf)
        pdf = adjust_pdf(np.copy(pdf), temp)
        accumulate = 0
        for i in range(0, pdf.size):
            accumulate += pdf[i]
            if accumulate >= x:
                return i
        tf.logging.info('Error with sampling ensemble.')
        return -1


    def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
        """ Sample from a 2D Gaussian """
        if greedy:
            return mu1, mu2
        mean = [mu1, mu2]
        s1 *= temp * temp
        s2 *= temp * temp
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]


    def get_seqs(z, seq_len, greedy, temp):
        """ Generate sequences according to latent vector """
        feed = {sample_model.batch_z: z}
        input_state = sess.run(sample_model.initial_state, feed)

        strokes = np.zeros((seq_len, len(z), 5), dtype=np.float32)
        factors = np.zeros((seq_len, 1), dtype=np.float32)
        input_x = np.zeros((len(z), 1, 5), dtype=np.float32)
        input_x[:, 0, 2] = 1  # Initially, we want to see beginning of new stroke
        #abs_seqs = input_x[:,:,:2].copy()
        for seq_i in range(seq_len):
            #abs_seqs = input_x[:,:,:]
            feed = {sample_model.initial_state: input_state,
                    sample_model.input_x: input_x,
                    #sample_model.input_abs_x:abs_seqs.copy()/sample_model.hps.abs_norm,
                    sample_model.batch_z: z
                    }

            dec_out, out_state = sess.run([sample_model.dec_out, sample_model.final_state], feed)

            pi, mux, muy, sigmax, sigmay, corr, pen, pen_logits = dec_out
            input_state = out_state

            # Generate stroke position from Gaussian mixtures
            idx = get_pi_idx(random.random(), pi[0], temp, greedy)
            next_factor = np.max(pi[0])
            next_x1, next_x2 = sample_gaussian_2d(mux[0][idx], muy[0][idx],
                                                  sigmax[0][idx], sigmay[0][idx],
                                                  corr[0][idx], np.sqrt(temp), greedy)
            # Generate stroke pen status
            idx_eos = get_pi_idx(random.random(), pen[0], temp, greedy)

            eos = np.zeros(3)
            eos[idx_eos] = 1

            strokes[seq_i, :, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]
            factors[seq_i,0] = next_factor
            input_x = np.array([next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
            input_x = input_x.reshape([1, 1, 5])
            #abs_seqs[:,:,:2] += input_x[:,:,:2]
        

        strokes = np.reshape(strokes,[1,-1,5])
        stroke_del = del_stroke(strokes,factors)
        #strokes = get_batch_rel(strokes,None)
        #strokes = np.reshape(strokes,[-1,5])
    
        stroke_del = np.reshape(stroke_del,[1,-1,5])
        stroke_del = get_batch_rel(stroke_del,None)
        stroke_del = np.reshape(stroke_del,[-1,5])
        return utils.seq_5d_to_3d(np.reshape(strokes, [seq_len, 5])),utils.seq_5d_to_3d(np.reshape(stroke_del, [seq_len, 5]))


    # Generate a batch of sketches based on one latent vector
    gen_strokes = []
    re_factor = []
    for i in range(gen_size):
        sketch,factor = get_seqs(z, seq_len, greedy_mode, temperature)
        gen_strokes.append(sketch)
        re_factor.append(factor)
    return gen_strokes


def load_model_params(model_dir):
    model_params = utils.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_config = json.dumps(json.load(f))
        model_params.parse_json(model_config)
    return model_params


def modify_model_params(model_params):
    """ Adjust to the generating mode """
    model_params.use_input_dropout = 0
    model_params.use_recurrent_dropout = 0
    model_params.use_output_dropout = 0
    model_params.is_training = False
    model_params.batch_size = 1
    model_params.max_seq_len = 1

    return model_params

def sort_paths(paths):
    """ Order the loaded images """
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

def get_part_rel(seqs,set=None):
    seqs1 = seqs.copy()
    #seqs1[:,:,:2] *= set.scale_factor
    for i in range(len(seqs1)):
        absx=0
        absy=0
        n=2
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
def del_stroke(stroke,factor):
    st = stroke[0].copy()
    fa = factor.copy()
    #print(fa)
    #print(st)
    mask = 0
    for i in range(len(st)):
        
        if i==0 or st[i-1,-2]==1:
            if fa[i] > 0.5:
                mask = 1
            else:
                mask = 0
        #if i !=0:
            #print(mask,st[i-1,-1],fa[i])
        st[i,:2] = mask * st[i,:2]
    return st
def pad_stroke_batch(stroke_set,stroke_len,max_num,bs):
    l = len(stroke_set)
    #print(max_num,bs,l,stroke_set)
    stroke_set = np.concatenate((stroke_set,np.zeros([max_num*bs-l,stroke_set.shape[1],2])),axis=0)
    stroke_len = np.concatenate((stroke_len, np.zeros([max_num * bs - l])), axis=0)
    return stroke_set,stroke_len

def get_sam_p(sess,test_set,t_model_params,index):
    model_params = utils.copy_hparams(t_model_params)
    
    FLAGS = tf.app.flags.FLAGS
    # Dataset directory

    # Output directory

    output_dir = './sample_ds1/sample_ds1'

    # Number of generated samples per category
    num_per_category= 100
    conditional = True


    samples_per_category = num_per_category

    model_params.batch_size = samples_per_category
    model_params.is_training = False
    model_params.use_input_dropout = 0
    model_params.use_recurrent_dropout = 0
    model_params.use_output_dropout = 0
 #   model_params.max_seq_len = max_seq_len
    model = Model(model_params,reuse=True)
    model_params.batch_size = 1
#    model_params.max_seq_len = max_seq_len
    model_params = modify_model_params(model_params)
    draw_model = Model(model_params,reuse=True)
    model_params.max_seq_len = t_model_params.max_seq_len

    return model,draw_model

def sample_def(sess,test_set,t_model_params,index,model,draw_model):
    model_params = utils.copy_hparams(t_model_params)
    FLAGS = tf.app.flags.FLAGS
    # Dataset directory

    # Output directory

    output_dir = './sample_ds33/sample_ds33'

    # Number of generated samples per category
    num_per_category= 100
    conditional = True


    samples_per_category = num_per_category

    

    #utils.load_checkpoint(sess, FLAGS.log_root)

    color = ['black', 'red', 'blue', 'green', 'orange', 'cyan', 'tomato', 'magenta', 'purple', 'brown']
 #   model_dir = FLAGS.model_dir
 #   data_dir = FLAGS.data_dir
    SVG_DIR = output_dir+'_'+str(index)
    
    # Temperature for synthesis, details can be found in aforementioned reference [1]
    temperature = 0.24


    #al, si = sess.run([model.de_alpha, model.de_sigma2])
    #model_params.categories = [['airplane', 'angel', 'apple', 'butterfly', 'bus', 'cake','fish', 'spider', 'The Great Wall','umbrella']]#,'bee','flower','bus','giraffe'
    if not os.path.exists(SVG_DIR):
        os.makedirs(SVG_DIR)
    #raw_data = utils.load_data(data_dir, model_params.categories[0], model_params.num_per_category)
    #print(model_params.categories[0])

    cnt_name = -1
    for category in range(len(model_params.categories)):
        
        #model_params.abs_norm  = float(train_set.calc_abs_seq_norm())
        
        
        index = np.arange(2500 * len(model_params.categories))
        #np.random.shuffle(index)

        # Map the input images to the latent variables
        seqs, pngs, labels, seq_len,s_n,stroke_len,stroke_num,_,box,box_ori = test_set._get_batch_from_indices(index[category*2500:(category)*2500+samples_per_category])
        s_n, stroke_len = pad_stroke_batch(s_n, stroke_len, model.hps.max_strokes_num, samples_per_category )
        #abs_seqs[:,:2] += half_width
        #abs_seqs[:,:,:2] = abs_seqs[:,:,:2] / model.hps.abs_norm
        
        #abs_seqs[:, :2] += half_width
        #abs_seqs[:,:,:2] /= model.hps.abs_norm
        #print(seqs)
        if conditional is True:  # Conditional sampling
            feed = {model.input_seqs: seqs, model.input_strokes: s_n, model.input_stroke_len: stroke_len,
                model.input_strokes_num: stroke_num, model.input_box: box}
            z = sess.run(model.p_mu, feed)
            feed = {model.input_seqs: seqs, model.input_strokes: s_n, model.input_stroke_len: stroke_len,
                model.input_strokes_num: stroke_num, model.input_box: box_ori}
            z_ori = sess.run(model.p_mu, feed)
        else:  # Without references input
            z = np.random.normal(0., 1., [samples_per_category, model_params.z_size])  # Latent codes of sketches you expected to generate

        feed = {
            model.batch_z: z
            #model.input_pngs: pngs_bias
        }
        #gau_label = sess.run(model.gau_label, feed)

        # Generate strokes
        
        name = "none"
        for cnt in range(samples_per_category):
            if cnt == 0:
                cnt_name +=1
                name = model_params.categories[cnt_name]
            # Generated sketches
            os.makedirs(f"{SVG_DIR}/{name}", exist_ok=True)
            path = os.path.join(f"{SVG_DIR}/{name}" ,'%d_%d.svg' % (category, cnt))
            stroke = sample(sess, draw_model, np.reshape(z[cnt, :], [1, -1]), 1, model_params.max_seq_len, temperature)
            #print(factor)
            
            filepath1 = os.path.join(SVG_DIR, '%d_%d.svg' % (category, cnt))
            #draw_strokes(stroke_del[0], filepath1, 48, margin=1.5, color=color[category])

            
            pre_draw = stroke[0].copy()
            #l=0
            for ii in range(len(pre_draw)-1):
                pre_draw[len(pre_draw)-ii-1,:2] -= pre_draw[len(pre_draw)-ii-2,:2]
            #print(pre_draw,color[category])
            #for i in range(len(pre_draw)-1):
            #    pre_draw[len(pre_draw)-i-1,:2] = pre_draw[len(pre_draw)-i-1,:2] - pre_draw[len(pre_draw)-i-2,:2]
            draw_strokes(pre_draw, filepath1, 48, margin=1.5, color=color[category])
            draw_strokes(pre_draw, path, 48, margin=1.5, color=color[category])

            pre_draw = utils.seq_5d_to_3d(seqs[cnt,:,:])
            #path = os.path.join(f"./sample/{name}" ,'%d_%d_ori.svg' % (category, cnt))
            #draw_strokes(pre_draw, path, 48, margin=1.5, color=color[category])
            # Corresponding latent codes
            filepath2 = os.path.join(SVG_DIR, 'code_%d_%d.npy' % (category, cnt))
            np.save(filepath2, np.reshape(z_ori[cnt, :], [1, -1]))
            # Corresponding indexes of the Gaussian components
            filepath4 = os.path.join(SVG_DIR, 's_%d_%d.npy' % (category, cnt))
            np.save(filepath4, stroke)
    for i in range(len(model_params.categories)):
        name = model_params.categories[i]
        path = os.path.join(f"{SVG_DIR}" ,name)
        exportsvg(path, path, 'png')
        file = os.listdir(path)
        for j in file:
            if j.endswith('svg'):
                os.remove(os.path.join(path, j))
    exportsvg(SVG_DIR, SVG_DIR, 'png')
def get_ds(k):
    if k==1:
        return [['pig' , 'bee','flower','bus','giraffe']]
    if k==2:
        return [['airplane', 'angel', 'apple', 'butterfly', 'bus', 'cake','fish', 'spider', 'The Great Wall','umbrella']]
    if k==3:
        return [['pig','bee','flower','bus','giraffe','car', 'cat' , 'horse']]

def main():
    #
    args = sys.argv
    sample_num = int(args[1])
    sample_ds = int(args[2])
    sample_path = args[3]
    sample_prob = float(args[4])
    model_ckpt = args[5]
    print(sample_num,sample_ds,sample_path,sample_prob)
    #

    FLAGS = tf.app.flags.FLAGS
    # Dataset directory
    tf.app.flags.DEFINE_string(
        'data_dir',
        './',
        'The directory in which to find the dataset specified in model hparams. '
    )
    # Checkpoint directory
    tf.app.flags.DEFINE_string(
        'model_dir', './ckpt_ds1_l',
        'Directory to store the model checkpoints.'
    )
    # Output directory
    tf.app.flags.DEFINE_string( 
        'output_dir', './sample_ds1_34_test',
        'Directory to store the generated sketches.'
    )
    # Number of generated samples per category
    tf.app.flags.DEFINE_integer(
        'num_per_category', 100 ,
        'Number of generated samples per category.'
    )
    # Whether the sampling needs the sketch images input as references
    tf.app.flags.DEFINE_boolean(
        'conditional', True,
        'Whether the sampling is with conditions.'
    )

    color = ['black', 'red', 'blue', 'green', 'orange', 'cyan', 'tomato', 'magenta', 'purple', 'brown']
    model_dir = FLAGS.model_dir
    data_dir = FLAGS.data_dir
    SVG_DIR = FLAGS.output_dir
    samples_per_category = FLAGS.num_per_category
    # Temperature for synthesis, details can be found in aforementioned reference [1]
    temperature = 0.24
    model_dir =  model_ckpt
    model_params = load_model_params(model_dir)

    #al, si = sess.run([model.de_alpha, model.de_sigma2])
    model_params.categories = [['pig' , 'bee','flower','bus','giraffe']]#,'bee','flower','bus','giraffe'
    #'airplane', 'angel', 'apple', 'butterfly', 'bus', 'cake','fish', 'spider', 'The Great Wall','umbrella'
    
    ###
    samples_per_category = sample_num 
    model_params.categories=get_ds(sample_ds) 
    #print(sample_ds.type)
    #print(get_ds(int(sample_ds)) )
    SVG_DIR=sample_path 
    model_dir =  model_ckpt

    ###

    if not os.path.exists(SVG_DIR):
        os.makedirs(SVG_DIR)
    raw_data = utils.load_data(data_dir, model_params.categories[0], model_params.num_per_category)
    print(model_params.categories[0])
    model_params.batch_size = samples_per_category
    train_set, valid_set, test_set, max_seq_len,max_stroke_len,max_strokes_num = utils.preprocess_data(raw_data,
                                                  model_params.batch_size,
                                                  model_params.random_scale_factor,
                                                  model_params.augment_stroke_prob,
                                                  model_params.png_scale_ratio,
                                                  model_params.png_rotate_angle,
                                                  model_params.png_translate_dist)
    model_params.batch_size = 1
    model_params.max_seq_len = max_seq_len
    model_params = modify_model_params(model_params)
    draw_model = Model(model_params)
    model_params.batch_size = samples_per_category
    model_params.max_seq_len = max_seq_len
    model = Model(model_params)
        
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())

    utils.load_checkpoint(sess, model_dir)
    cnt_name = -1
    for category in range(len(model_params.categories[0])):
        
        #model_params.abs_norm  = float(train_set.calc_abs_seq_norm())
        

        index = np.arange(2500 * len(model_params.categories[0]))
        #np.random.shuffle(index)

        # Map the input images to the latent variables
        seqs, pngs, labels, seq_len,s_n,stroke_len,stroke_num,_,box,box_ori = test_set._get_batch_from_indices(index[category*2500:(category)*2500+samples_per_category],PROB=sample_prob)
        s_n, stroke_len = pad_stroke_batch(s_n, stroke_len, model.hps.max_strokes_num, model.hps.batch_size, )
        #abs_seqs[:,:2] += half_width
        #abs_seqs[:,:,:2] = abs_seqs[:,:,:2] / model.hps.abs_norm
        
        #abs_seqs[:, :2] += half_width
        #abs_seqs[:,:,:2] /= model.hps.abs_norm
        
        if FLAGS.conditional is True:  # Conditional sampling
            feed = {model.input_seqs: seqs, model.input_strokes: s_n, model.input_stroke_len: stroke_len,
                model.input_strokes_num: stroke_num, model.input_box: box}
            z = sess.run(model.p_mu, feed)
            feed = {model.input_seqs: seqs, model.input_strokes: s_n, model.input_stroke_len: stroke_len,
                model.input_strokes_num: stroke_num, model.input_box: box_ori}
            z_ori = sess.run(model.p_mu, feed)
        else:  # Without references input
            z = np.random.normal(0., 1., [samples_per_category, model_params.z_size])  # Latent codes of sketches you expected to generate

        feed = {
            model.batch_z: z
            #model.input_pngs: pngs_bias
        }
        #gau_label = sess.run(model.gau_label, feed)

        # Generate strokes
        
        name = "none"
        for cnt in range(samples_per_category):
            if cnt == 0:
                cnt_name +=1
                name = model_params.categories[0][cnt_name]
            # Generated sketches
            os.makedirs(f"{SVG_DIR}/{name}", exist_ok=True)
            path = os.path.join(f"{SVG_DIR}/{name}" ,'%d_%d.svg' % (category, cnt))
            stroke = sample(sess, draw_model, np.reshape(z[cnt, :], [1, -1]), 1, max_seq_len, temperature)
            #print(factor)
            
            filepath1 = os.path.join(SVG_DIR, '%d_%d.svg' % (category, cnt))
            #draw_strokes(stroke_del[0], filepath1, 48, margin=1.5, color=color[category])

            '''pre_draw1 = seqs[cnt,:,:]
            #l=0
            for ii in range(len(pre_draw1)-1):
                pre_draw1[len(pre_draw1)-ii-1,:2] -= pre_draw1[len(pre_draw1)-ii-2,:2]'''
            
            pre_draw = stroke[0].copy()
            #l=0
            for ii in range(len(pre_draw)-1):
                pre_draw[len(pre_draw)-ii-1,:2] -= pre_draw[len(pre_draw)-ii-2,:2]
            #for i in range(len(pre_draw)-1):
            #    pre_draw[len(pre_draw)-i-1,:2] = pre_draw[len(pre_draw)-i-1,:2] - pre_draw[len(pre_draw)-i-2,:2]
            draw_strokes(pre_draw, filepath1, 48, margin=1.5, color=color[category])
            draw_strokes(pre_draw, path, 48, margin=1.5, color=color[category])

            #pre_draw = utils.seq_5d_to_3d(seqs[cnt,:,:])
            #path = os.path.join(f"./sample/{name}" ,'%d_%d_ori.svg' % (category, cnt))
            #draw_strokes(pre_draw, path, 48, margin=1.5, color=color[category])
            # Corresponding latent codes
            filepath2 = os.path.join(SVG_DIR, 'code_%d_%d.npy' % (category, cnt))
            np.save(filepath2, np.reshape(z_ori[cnt, :], [1, -1]))
            # Corresponding indexes of the Gaussian components
            filepath4 = os.path.join(SVG_DIR, 's_%d_%d.npy' % (category, cnt))
            np.save(filepath4, stroke)
            #print(s_n[:10,:,:],seqs[0,:,:],stroke)
            #break
    for i in range(len(model_params.categories[0])):
        name = model_params.categories[0][i]
        path = os.path.join(f"{SVG_DIR}" ,name)
        exportsvg(path, path, 'png')
        file = os.listdir(path)
        for j in file:
            if j.endswith('svg'):
                os.remove(os.path.join(path, j))
    exportsvg(SVG_DIR, SVG_DIR, 'png')

if __name__ == '__main__':
    main()