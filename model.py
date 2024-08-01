from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import rnn
import Cnn

class Model(object):
    def __init__(self, hps, reuse=tf.AUTO_REUSE):
        self.hps = hps
        with tf.variable_scope('sketchBoard', reuse=reuse):
            self.config_model()
            self.build_sketchBoard()

    def config_model(self):
        """ Model configuration """
        self.k = self.hps.num_mixture * self.hps.num_sub  # Gaussian number
        self.global_ = tf.get_variable(name='num_of_steps', shape=[], initializer=tf.ones_initializer(dtype=tf.float32),
                                       trainable=False)

        self.sequence_lengths = tf.placeholder(
            dtype=tf.int32, shape=[self.hps.batch_size])
        self.input_seqs = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.max_seq_len + 1, 5],
                                         name="input_seqs")
        self.mask = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.max_seq_len + 1,1],
                                         name="mask")
        self.input_abs_seqs = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.max_seq_len + 1, 5],
                                         name="input_abs_seqs")
        self.input_pngs = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.png_width, self.hps.png_width],
                                         name="input_pngs")
        self.input_strokes = tf.placeholder(tf.float32, [self.hps.batch_size * self.hps.max_strokes_num, self.hps.max_stroke_len,2],
                                         name="input_strokes")
        self.input_stroke_len = tf.placeholder(tf.int32, [self.hps.batch_size * self.hps.max_strokes_num],
                                         name="input_stroke_len")
        self.input_strokes_num = tf.placeholder(tf.int32, [self.hps.batch_size ],
                                               name="input_strokes_num")
        self.input_box = tf.placeholder(tf.float32, [self.hps.batch_size ,self.hps.max_strokes_num,self.hps.max_strokes_num,9],
                                         name="input_stroke_len")

        '''input = tf.reshape(self.input_box,[-1,9])
        fc_spec = [('no', 9, 1, 'fc_embed')]
        fc_net = Cnn.FcNet(fc_spec, input)
        self.rel_mat = fc_net.fc_layers[-1]
        self.rel_matrix = tf.reshape(self.rel_mat,[-1,self.hps.max_strokes_num,self.hps.max_strokes_num])
        self.rel_matrix = tf.ones_like(self.rel_matrix)'''
        self.input_x = tf.identity(self.input_seqs[:, :self.hps.max_seq_len, :], name='input_x')
        self.input_abs_seqs_masked = tf.concat([self.input_abs_seqs,self.mask],axis=2)
        self.input_abs_x = tf.identity(self.input_abs_seqs_masked[:, :self.hps.max_seq_len, :], name='input_abs_x')
        self.output_x = self.input_seqs[:, 1:self.hps.max_seq_len + 1, :]
        self.output_abs_x = self.input_abs_seqs_masked[:, 1:self.hps.max_seq_len + 1, :]
        # Decoder cell configuration
        if self.hps.dec_model == 'lstm':
            cell_fn = rnn.LSTMCell
        elif self.hps.dec_model == 'layer_norm':
            cell_fn = rnn.LayerNormLSTMCell
        elif self.hps.dec_model == 'hyper':
            cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        if self.hps.enc_model == 'lstm':
            enc_cell_fn = rnn.LSTMCell
        elif self.hps.enc_model == 'layer_norm':
            enc_cell_fn = rnn.LayerNormLSTMCell
        elif self.hps.enc_model == 'hyper':
            enc_cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        # Dropout configuration
        use_input_dropout = False if self.hps.use_input_dropout == 0 else True
        use_output_dropout = False if self.hps.use_output_dropout == 0 else True
        use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True

        cell = cell_fn(self.hps.dec_rnn_size,
                       use_recurrent_dropout=use_recurrent_dropout,
                       dropout_keep_prob=self.hps.recurrent_dropout_prob)
        if self.hps.enc_model == 'hyper':
            self.enc_cell_fw = enc_cell_fn(
                self.hps.enc_rnn_size,
                use_recurrent_dropout=use_recurrent_dropout,
                dropout_keep_prob=self.hps.recurrent_dropout_prob)
            self.enc_cell_bw = enc_cell_fn(
                self.hps.enc_rnn_size,
                use_recurrent_dropout=use_recurrent_dropout,
                dropout_keep_prob=self.hps.recurrent_dropout_prob)
        else:
            self.enc_cell_fw = enc_cell_fn(
                self.hps.enc_rnn_size,
                use_recurrent_dropout=use_recurrent_dropout,
                dropout_keep_prob=self.hps.recurrent_dropout_prob)
            self.enc_cell_bw = enc_cell_fn(
                self.hps.enc_rnn_size,
                use_recurrent_dropout=use_recurrent_dropout,
                dropout_keep_prob=self.hps.recurrent_dropout_prob)


        if use_input_dropout:
            tf.logging.info('Dropout to input w/ keep_prob = %4.4f.', self.hps.input_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)

        if use_output_dropout:
            tf.logging.info('Dropout to output w/ keep_prob = %4.4f.', self.hps.output_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.hps.output_dropout_prob)

        self.cell = cell

    def build_sketchBoard(self):
        self.ol, self.last_state = tf.nn.bidirectional_dynamic_rnn(
            self.enc_cell_fw,
            self.enc_cell_bw,
            self.input_strokes,
            sequence_length=self.input_stroke_len,
            time_major=False,
            swap_memory=True,
            dtype=tf.float32,
            scope='ENC1_RNN')
        last_state_fw, last_state_bw = self.last_state
        last_h_fw = self.enc_cell_fw.get_output(last_state_fw)
        last_h_bw = self.enc_cell_bw.get_output(last_state_bw)
        last_h = tf.concat([last_h_fw, last_h_bw], 1)
        self.embed = self.reconstruct_batch(last_h)
        #ex process
        '''self.d_embed = tf.tile(tf.expand_dims(tf.reshape(self.embed, [-1, self.hps.max_strokes_num, self.hps.enc_rnn_size * 2]),axis = 1),[1,self.hps.max_strokes_num,1,1])
        #self.rel_matrix = tf.concat([self.d_embed,self.input_box],axis=3)
        fc_spec = [('no', self.hps.enc_rnn_size * 2 , 9, 'fc_embed')]
        fc_net = Cnn.FcNet(fc_spec, self.d_embed)
        self.rel_matrix_1 = fc_net.fc_layers[-1]
        self.rel_matrix = tf.concat([self.rel_matrix_1,self.input_box],axis=3)
        fc_spec = [('no', 18 , 1, 'fc_embed2')]
        fc_net = Cnn.FcNet(fc_spec, self.rel_matrix)
        self.rel_matrix = tf.squeeze(fc_net.fc_layers[-1])'''
        self.embed = tf.tile(tf.expand_dims(tf.reshape(self.embed, [-1, self.hps.max_strokes_num, self.hps.enc_rnn_size * 2]),axis = 3),[1,1,1,9])
        #self.rel_matrix = tf.transpose(self.input_box, [0,3,1,2])

        def kaiming_init(size, activate):
            in_dim = size[1]
            if activate == 'relu':
                kaiming_stddev = tf.sqrt(2. / in_dim)
            elif activate == 'tanh':
                kaiming_stddev = tf.sqrt(5. / (3. * in_dim))
            elif activate == 'sigmoid':
                kaiming_stddev = tf.sqrt(1. / in_dim)
            else:
                kaiming_stddev = tf.sqrt(1. / in_dim)
            return tf.random_normal(shape=size, stddev=kaiming_stddev)

        weight_1 = tf.get_variable(name="gcn_weight_1", dtype=tf.float32, initializer=kaiming_init([9,self.hps.enc_rnn_size * 2, self.hps.enc_rnn_size * 2], 'relu'))
        weight_2 = tf.get_variable(name="gcn_weight_2", dtype=tf.float32, initializer=kaiming_init([9,self.hps.enc_rnn_size * 2, self.hps.enc_rnn_size * 2], 'relu'))
        #weight_3 = tf.get_variable(name="gcn_weight_3", dtype=tf.float32, initializer=kaiming_init([9,1], 'none'))
        #weight_4 = tf.get_variable(name="gcn_weight_4", dtype=tf.float32, initializer=kaiming_init([9,1], 'none'))
        #weight_5 = tf.get_variable(name="gcn_weight_5", dtype=tf.float32, initializer=kaiming_init([1,self.hps.max_strokes_num], 'none'))
        gcn_out_0 = []
        #norm part
        times = self.input_box.shape[2]
        factor = tf.tile(tf.reduce_sum(self.input_box[:,:,:,:],axis=2,keep_dims=True),[1,1,times,1])
        #self.input_box = self.input_box/(factor+1e-3)
        
        for ii in range(9):
            gcn_out_0.append(tf.matmul(self.input_box[:,:,:,ii],self.embed[:,:,:,ii]))
        gcn_out_0 = tf.stack(gcn_out_0)
        gcn_out_0 = tf.transpose(gcn_out_0,[1,2,3,0])
        gcn_out_0 = tf.reshape(gcn_out_0,[-1,self.hps.enc_rnn_size * 2,9])
        #gcn_out_0 = tf.reshape(tf.matmul(self.rel_matrix, self.embed), [-1,9, self.hps.enc_rnn_size * 2])
        gcn_out_1 = []
        for ii in range(9):
            gcn_out_1.append(tf.nn.relu(tf.matmul(gcn_out_0[:,:,ii], weight_1[ii,:,:])))
        gcn_out_1 = tf.stack(gcn_out_1)
        gcn_out_1 = tf.transpose(gcn_out_1,[1,2,0])
        gcn_out_2 = []
        for ii in range(9):
            gcn_out_2.append(tf.nn.relu(tf.matmul(gcn_out_1[:,:,ii], weight_2[ii,:,:])))
        gcn_out_2 = tf.stack(gcn_out_2)
        gcn_out_2 = tf.transpose(gcn_out_2,[1,2,0])
        #gcn_out_0 = tf.matmul(gcn_out_0,weight_3)
        #gcn_out_2 = tf.matmul(gcn_out_2,weight_4)
        gcn_out_0 = tf.reduce_sum(gcn_out_0,axis=2) 
        gcn_out_2 = tf.reduce_sum(gcn_out_2,axis=2) 
        #gcn_out_1 = tf.nn.relu(tf.matmul(gcn_out_0, weight_1))
        #gcn_out_2 = tf.nn.relu(tf.matmul(gcn_out_1, weight_2))
        gcn_out = tf.reduce_sum(tf.reshape((gcn_out_0 + gcn_out_2)/2., [self.hps.batch_size, self.hps.max_strokes_num, self.hps.enc_rnn_size * 2 ]),
                                axis=1) #/ tf.cast(tf.tile(tf.expand_dims(self.input_strokes_num,axis=1),[1,self.hps.enc_rnn_size*2]), dtype=tf.float32)
        #gcn_out = tf.reshape(tf.matmul(weight_5,tf.reshape((gcn_out_0 + gcn_out_2)/2., [self.hps.batch_size, self.hps.max_strokes_num, self.hps.enc_rnn_size * 2 ])),[self.hps.batch_size,self.hps.enc_rnn_size * 2])
        #ll rev *9 1up 2down #2 sum
        #lll rev w5 reduce
        #llll erv red

        bn2_out = tf.nn.tanh(tf.contrib.layers.batch_norm(gcn_out, decay=0.9, epsilon=1e-05, center=True, scale=True,
                                                          updates_collections=None, is_training=self.hps.is_training))

        fc_spec_mu = [('no', self.hps.enc_rnn_size * 2 , self.hps.z_size, 'fc_mu')]
        fc_net_mu = Cnn.FcNet(fc_spec_mu, bn2_out)
        self.p_mu = fc_net_mu.fc_layers[-1]

        fc_spec_sigma2 = [('no', self.hps.enc_rnn_size * 2 , self.hps.z_size, 'fc_sigma2')]
        fc_net_sigma2 = Cnn.FcNet(fc_spec_sigma2, bn2_out)
        self.p_sigma2 = fc_net_sigma2.fc_layers[-1]
        self.batch_z = self.get_z(self.p_mu, self.p_sigma2)
        with tf.variable_scope('decoder') as dec_param_scope:
            fc_spec = [('tanh', self.hps.z_size, self.cell.state_size, 'init_state')]
            fc_net = Cnn.FcNet(fc_spec, self.batch_z)
            self.initial_state = fc_net.fc_layers[-1]

            dec_input = tf.concat(
                [self.input_x, tf.tile(tf.expand_dims(self.batch_z, axis=1), [1, self.hps.max_seq_len, 1])], axis=2)

            self.dec_out, self.final_state = self.rnn_decoder(dec_input, self.initial_state)
            self.pi, self.mux, self.muy, self.sigmax, self.sigmay, self.corr, self.pen, self.pen_logits = self.dec_out
        #self.gaussian_loss = self.calculate_gaussian_loss(self.p_alpha, self.component_z, tf.stop_gradient(self.q_mu),
        #                                                  tf.stop_gradient(self.hyper_mask))
        target = tf.reshape(self.output_x, [-1, 5])
        self.x1_data, self.x2_data, self.pen_data = tf.split(target, [1, 1, 3], 1)
        self.lil_loss = self.get_lil_loss(self.pi, self.mux, self.muy, self.sigmax, self.sigmay, self.corr,
                                          self.pen_logits, self.x1_data, self.x2_data, self.pen_data)
        self.loss = self.lil_loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.lr = (self.hps.learning_rate - self.hps.min_learning_rate) * \
                      (self.hps.decay_rate ** self.global_) + self.hps.min_learning_rate
            optimizer = tf.train.AdamOptimizer(self.lr)
            gvs = optimizer.compute_gradients(self.loss)

            g = self.hps.grad_clip
            for i, (grad, var) in enumerate(gvs):
                if grad is not None:
                    gvs[i] = (tf.clip_by_norm(grad, g), var)
            self.train_op = optimizer.apply_gradients(gvs)

        #self.rel_matrix  = tf.nn.embedding_lookup(self.pos_embeddings,self.input_box)
        '''target = tf.reshape(self.output_x, [-1, 5])
        self.x1_data, self.x2_data, self.pen_data = tf.split(target, [1, 1, 3], axis=1)
        #self.output_t = tf.concat([self.output_abs_x,self.output_x],axis=2)
        self.p_mu, self.p_sigma2 = self.encoder(self.output_abs_x,self.sequence_lengths)
        self.p_sigma2 = tf.exp(self.p_sigma2 / 2.0)
        self.batch_z = self.get_z(self.p_mu, self.p_sigma2)  # reparameterization
        
        fc_spec = [('tanh', self.hps.z_size, self.cell.state_size, 'init_state')]
        fc_net = Cnn.FcNet(fc_spec, self.batch_z)
        self.initial_state = fc_net.fc_layers[-1]
        pre_z = tf.tile(tf.reshape(self.batch_z, [self.hps.batch_size, 1, self.hps.z_size]),
                        [1, self.hps.max_seq_len, 1])
        dec_input = tf.concat([self.input_x, pre_z], axis=2)
        self.dec_out, self.final_state = self.rnn_decoder(dec_input, self.initial_state)
        self.pi, self.mux, self.muy, self.sigmax, self.sigmay, self.corr, self.pen, self.pen_logits = self.dec_out
        self.half_z_size = self.hps.z_size/2
        self.half_z_size = int(self.half_z_size)
        self.gen_img = self.cnn_decoder_ex(self.batch_z[:,0:self.half_z_size])
        #self.gen_img = tf.clip_by_value(self.gen_img*100,-1,1)
        self.lil_loss = self.get_lil_loss(self.pi, self.mux, self.muy, self.sigmax, self.sigmay, self.corr,
                                          self.pen_logits, self.x1_data, self.x2_data, self.pen_data)
        self.de_loss = self.calculate_deconv_loss(self.input_pngs, self.gen_img,'square')
        #self.loss = tf.cond( self.lil_loss * 10 < self.de_loss,lambda:self.hps.de_weight * self.de_loss ,lambda:self.lil_loss + self.hps.de_weight *self.de_loss)
        self.loss = self.lil_loss + self.hps.de_weight *self.de_loss
        
        self.lr = (self.hps.learning_rate - self.hps.min_learning_rate) * \
                  (self.hps.decay_rate ** (self.global_ / 3)) + self.hps.min_learning_rate
        optimizer = tf.train.AdamOptimizer(self.lr,beta1=0.9)
        self.optimizer = optimizer
        optimizer1 = tf.train.GradientDescentOptimizer(self.lr)
        #optimizer1 = tf.Print(self.gen_img,['grad_value: ',optimizer.variables()])
        gvs_t = optimizer.compute_gradients(self.loss)
        gvs = gvs_t
        #self.r_gvs = gvs
        #gvs1 = tf.Print(self.gen_img,['gvs_value: ',gvs])
        g = self.hps.grad_clip
        for i, (grad, var) in enumerate(gvs):
            #grad1 = tf.Print(self.gen_img,['grad_value: ',grad])
                #grad.eval()
                #grad1 = tf.Print(self.gen_img,['grad_value: ',grad])
            if grad is not None:

                gvs[i] = (tf.clip_by_value(grad, -g, g), var)
                

        self.train_op = optimizer.apply_gradients(gvs)
       
        gvs_t = optimizer1.compute_gradients(self.loss)
        gvs = gvs_t
        #self.r_gvs = gvs
        #gvs1 = tf.Print(self.gen_img,['gvs_value: ',gvs])
        g = self.hps.grad_clip
        for i, (grad, var) in enumerate(gvs):
            #grad1 = tf.Print(self.gen_img,['grad_value: ',grad])
                #grad.eval()
                #grad1 = tf.Print(self.gen_img,['grad_value: ',grad])
            if grad is not None:

                gvs[i] = (tf.clip_by_value(grad, -g, g), var)
        self.train_op1 = optimizer1.apply_gradients(gvs)
        


        #am 2
        self.z_in = tf.identity(self.batch_z[:,self.half_z_size:],name='z_in')

        self.z_in = tf.stop_gradient(self.z_in)
        self.gen_img2 = self.cnn_decoder_in(self.z_in)
        self.de_loss2 = self.calculate_deconv_loss(self.input_pngs, self.gen_img2,'square')
        self.loss2 =  self.hps.de_weight *self.de_loss2
        optimizer2 = tf.train.AdamOptimizer(self.lr)
        gvs = optimizer2.compute_gradients(self.loss2)
        g = self.hps.grad_clip
        for i, (grad, var) in enumerate(gvs):
            if grad is not None:

                gvs[i] = (tf.clip_by_value(grad, -g, g), var)
        self.train_op2 = optimizer2.apply_gradients(gvs)

        #am 3
        self.z_in2 =  tf.identity(self.batch_z[:,self.half_z_size:],name='z_in2')
        self.gen_img3 = self.cnn_decoder_in(self.z_in2)
        self.image_in = tf.zeros_like(self.input_pngs)
        self.de_loss3 = self.calculate_deconv_loss(self.image_in, self.gen_img3, 'square')
        self.loss3 = self.hps.de_weight *self.de_loss3
        optimizer3 = tf.train.AdamOptimizer(self.lr)
        gvs = optimizer.compute_gradients(self.loss3)
        g = self.hps.grad_clip
        for i, (grad, var) in enumerate(gvs): 
            if grad is not None :

                gvs[i] = (tf.clip_by_value(grad, -g, g), var)
        self.train_op3 = optimizer.apply_gradients(gvs)'''

    def cnn_decoder_ex(self, code):
        with tf.variable_scope('deconv_ex', reuse=tf.AUTO_REUSE):
            fc_spec = [
                ('leaky_relu', self.half_z_size, 3 * 3 * 256, 'fc1'),
            ]
            fc_net = Cnn.FcNet(fc_spec, code)
            fc1 = fc_net.fc_layers[-1]
            fc1 = tf.reshape(fc1, [-1, 3, 3, 256])

            de_conv_specs = [
                #('leaky_relu', (3, 3), [1, 2, 2, 1], 256),
                ('leaky_relu', (3, 3), [1, 2, 2, 1], 128),
                ('leaky_relu', (3, 3), [1, 2, 2, 1], 64),
                ('leaky_relu', (3, 3), [1, 2, 2, 1], 32),
                #('leaky_relu', (3, 3), [1, 2, 2, 1], 16),
                ('tanh', (3, 3), [1, 2, 2, 1], 1)
            ]
            conv_net = Cnn.ConvNet(de_conv_specs, fc1, self.hps.is_training, deconv=True)
        return conv_net.conv_layers[-1]

    def calculate_deconv_loss(self, img, gen_img, sign):
        img = tf.reshape(img, [self.hps.batch_size, self.hps.png_width ** 2])
        gen_img = tf.reshape(gen_img, [self.hps.batch_size, self.hps.png_width ** 2])
        if sign == 'square':
            return tf.reduce_mean(tf.reduce_sum(tf.square(img - gen_img), axis=1))
        elif sign == 'absolute':
            return tf.reduce_mean(tf.reduce_sum(tf.abs(img - gen_img), axis=1))
        else:
            assert False, 'please choose a respectable cell'

    def cnn_decoder_in(self, code):
        with tf.variable_scope('deconv_in', reuse=tf.AUTO_REUSE):
            fc_spec = [
                ('leaky_relu', self.half_z_size, 3 * 3 * 256, 'fc1'),
            ]
            fc_net = Cnn.FcNet(fc_spec, code)
            fc1 = fc_net.fc_layers[-1]
            fc1 = tf.reshape(fc1, [-1, 3, 3, 256])

            de_conv_specs = [
                #('leaky_relu', (3, 3), [1, 2, 2, 1], 256),
                ('leaky_relu', (3, 3), [1, 2, 2, 1], 128),
                ('leaky_relu', (3, 3), [1, 2, 2, 1], 64),
                ('leaky_relu', (3, 3), [1, 2, 2, 1], 32),
                #('leaky_relu', (3, 3), [1, 2, 2, 1], 16),
                ('tanh', (3, 3), [1, 2, 2, 1], 1)
            ]
            conv_net = Cnn.ConvNet(de_conv_specs, fc1, self.hps.is_training, deconv=True)
        return conv_net.conv_layers[-1]

    def get_lil_loss(self, pi, mu1, mu2, s1, s2, corr, pen_logits, x1_data, x2_data, pen_data):
        result0 = self.get_density(x1_data, x2_data, mu1, mu2, s1, s2, corr)
        epsilon = 1e-6
        result1 = tf.multiply(result0, pi)
        result1 = tf.reduce_sum(result1, axis=1, keep_dims=True)
        result1 = -tf.log(result1 + epsilon)  # Avoid log(0)

        masks = 1.0 - pen_data[:, 2]
        masks = tf.reshape(masks, [-1, 1])
        result1 = tf.multiply(result1, masks)

        result2 = tf.nn.softmax_cross_entropy_with_logits(logits=pen_logits, labels=pen_data)
        result2 = tf.reshape(result2, [-1, 1])

        if not self.hps.is_training:
            result2 = tf.multiply(result2, masks)
        return tf.reduce_mean(tf.reduce_sum(tf.reshape(result1 + result2, [self.hps.batch_size, -1]), axis=1))

    def reconstruct_batch(self,batch):
        #self.embed = tf.zeros([self.hps.batch_size, self.hps.max_strokes_num, 1024])
        sum = 0
        zero = tf.zeros([self.hps.max_strokes_num,self.hps.enc_rnn_size * 2])
        embed_list = []
        for i in range(self.hps.batch_size):
            l = self.input_strokes_num[i]
            tmp = batch[sum:sum+l,:]
            #zero = tf.zeros([self.hps.max_strokes_num-l,self.hps.enc_rnn_size * 2])
            tmp = tf.concat((tmp,zero[0:self.hps.max_strokes_num-l,:]),axis=0)
            #self.embed[i,:l,:] = batch[sum:sum+l,:]
            embed_list.append(tmp)
            sum += l
        self.embed = tf.stack(embed_list)
        return self.embed


    def encoder(self, batch, sequence_lengths):
        """Define the bi-directional encoder module of sketch-rnn."""
        unused_outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
            self.enc_cell_fw,
            self.enc_cell_bw,
            batch,
            sequence_length=sequence_lengths,
            time_major=False,
            swap_memory=True,
            dtype=tf.float32,
            scope='ENC_RNN')

        last_state_fw, last_state_bw = last_states
        last_h_fw = self.enc_cell_fw.get_output(last_state_fw)
        last_h_bw = self.enc_cell_bw.get_output(last_state_bw)
        last_h = tf.concat([last_h_fw, last_h_bw], 1)

        mu = rnn.super_linear(
        last_h,
        self.hps.z_size,
        input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
        scope='ENC_RNN_mu',
        init_w='gaussian',
        weight_start=0.001)
        presig = rnn.super_linear(
        last_h,
        self.hps.z_size,
        input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
        scope='ENC_RNN_sigma',
        init_w='gaussian',
        weight_start=0.001)
        
        
        return mu, presig

    def rnn_decoder(self, inputs, initial_state):
        # Number of outputs is end_of_stroke + prob + 2 * (mu + sig) + corr
        num_mixture = 20
        n_out = (3 + num_mixture * 6)

        with tf.variable_scope('decoder'):
            output, last_state = tf.nn.dynamic_rnn(
                self.cell,
                inputs,
                initial_state=initial_state,
                time_major=False,
                swap_memory=True,
                dtype=tf.float32)

            output = tf.reshape(output, [-1, self.hps.dec_rnn_size])
            fc_spec = [('no', self.hps.dec_rnn_size, n_out, 'fc')]
            fc_net = Cnn.FcNet(fc_spec, output)
            output = fc_net.fc_layers[-1]

            out = self.get_mixture_params(output)
            last_state = tf.identity(last_state, name='last_state')
            self.output = output
        return out, last_state

    def get_mixture_params(self, output):
        pen_logits = output[:, 0:3]
        pi, mu1, mu2, sigma1, sigma2, corr = tf.split(output[:, 3:], 6, 1)

        pi = tf.nn.softmax(pi)
        pen = tf.nn.softmax(pen_logits)

        sigma1 = tf.exp(sigma1)
        sigma2 = tf.exp(sigma2)
        corr = tf.tanh(corr)

        r = [pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits]
        return r

    def get_z(self, mu, sigma2):
        """ Reparameterization """
        #sigma = tf.sqrt(sigma2)
        sigma = tf.exp(sigma2 / 2)
        eps = tf.random_normal((self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)
        z = tf.add(mu, tf.multiply(sigma, eps), name='z_code')
        return z

    def get_density(self, x1, x2, mu1, mu2, s1, s2, rho):
        norm1 = tf.subtract(x1, mu1)
        norm2 = tf.subtract(x2, mu2)
        s1s2 = tf.multiply(s1, s2)
        z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
             2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
        neg_rho = 1 - tf.square(rho)
        result = tf.exp(tf.div(-z, 2 * neg_rho))
        denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
        result = tf.div(result, denom)
        return result