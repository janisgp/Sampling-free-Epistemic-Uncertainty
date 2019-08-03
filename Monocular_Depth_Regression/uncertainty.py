import time
import numpy as np
import tensorflow as tf

from monodepth_model import MonodepthModel
from utils.postprocessing import post_process_disparity, post_process_disparity_var
from utils.derivatives import dElu_tf, dSigmoid_tf


class monodepth_uncertainty():
    
    def __init__(self, params):
        self.init_graph(params)
        
    def init_graph(self, params):
        """
        init modified graph, hard coded
        """

        tf.reset_default_graph()

        self.image_path = tf.placeholder(tf.string)

        left_inp = tf.image.decode_png(tf.read_file(self.image_path))
        left_inp_norm = left_inp / 255

        self.left_resize_norm = tf.image.resize_images(left_inp_norm,  [params.height, params.width], tf.image.ResizeMethod.AREA)

        left = tf.stack([self.left_resize_norm,  tf.image.flip_left_right(self.left_resize_norm)],  0)
        left.set_shape( [2, None, None, 3])
        right = None

        model = MonodepthModel(params, params.mode, left, right, use_dropout=params.use_dropout)

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)

        # SAVER
        train_saver = tf.train.Saver()

        # INIT
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coordinator)

        # RESTORE
        restore_path = params.checkpoint_path.split(".")[0]
        train_saver.restore(self.sess, restore_path)

        ##########################
        ### Forward Prop MC

        print('MC forward...')

        weights_conv2 = tf.get_default_graph().get_tensor_by_name("model/decoder/Conv_17/weights:0")
        biases_conv2 = tf.get_default_graph().get_tensor_by_name("model/decoder/Conv_17/biases:0")

        # concat1 = model.concat1
        elu6 = tf.get_default_graph().get_tensor_by_name("model/decoder/Conv_16/Elu:0")

        outs = []
        conv2_list = []
        for i in range(params.mc_samples):


            dropped = tf.nn.dropout(elu6, params.drop_rate)
            kernel_size = 3
            p = np.floor((kernel_size - 1) / 2).astype(np.int32)
            p_dropped = tf.pad(dropped, [[0, 0], [p, p], [p, p], [0, 0]])
            conv2 = tf.nn.convolution(p_dropped, weights_conv2, 'VALID')
            conv2_b = tf.nn.bias_add(conv2, biases_conv2)
            conv2_list.append(tf.expand_dims(conv2_b, 0))
            conv2 = 0.3 * tf.nn.sigmoid(conv2_b)

            conv2 = tf.expand_dims(tf.expand_dims(conv2[:,:,:,0], 3), 0)

            outs.append(conv2)

        conv2_list = tf.concat(conv2_list, 0)
        self.outs = tf.concat(outs, 0)


        print('OUR forward...')

        # concat1 = model.concat1
        elu6 = tf.get_default_graph().get_tensor_by_name("model/decoder/Conv_16/Elu:0")
        pre_sigmoid = tf.get_default_graph().get_tensor_by_name("model/decoder/Conv_17/BiasAdd:0")

        var = elu6**2 * (1 - params.drop_rate) / params.drop_rate

        kernel_size = 3
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_var2 = tf.pad(var, [[0, 0], [p, p], [p, p], [0, 0]])
        conv2_var_p = tf.nn.convolution(p_var2, weights_conv2**2, 'VALID')
        conv2_var = 0.3**2 * conv2_var_p * dSigmoid_tf(pre_sigmoid)**2

        self.our_var = tf.expand_dims(conv2_var[:,:,:,0], 3)

        # our prediction
        kernel_size = 3
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_dropped_our = tf.pad(elu6*params.drop_rate, [[0, 0], [p, p], [p, p], [0, 0]])
        conv2_our = tf.nn.convolution(p_dropped_our, weights_conv2, 'VALID')
        conv2_b_our = tf.nn.bias_add(conv2_our, biases_conv2)
        conv2_our = 0.3 * tf.nn.sigmoid(conv2_b_our)

        self.our_mean = tf.expand_dims(tf.expand_dims(conv2_our[:,:,:,0], 3), 0)

    def forward_mc(self, img_path, pp: bool=True):
        """
        forward prop with mc sampling
        """

        # prediction
        start = time.time()
        result = self.sess.run(self.outs, feed_dict={self.image_path: img_path})
        rt = time.time() - start

        if pp:
            # postprocessing
            res_pp_list = []
            for i in range(result.shape[0]):
                res_pp = post_process_disparity(result[i].squeeze())
                res_pp_list.append(np.expand_dims(res_pp, 0))
            res_pp = np.concatenate(res_pp_list, 0)

            # get mean and var
            res_pp_mean = np.mean(res_pp, 0)
            res_pp_var = np.var(res_pp, 0)

            return res_pp_mean, res_pp_var, rt
        else:
            return result.mean(0)[0], result.var(0), rt

    def forward_our(self, img_path, pp: bool=True):
        """
        forward prop with mc sampling
        """

        # prediction
        start = time.time()
        pred_our, var_our = self.sess.run([self.our_mean, self.our_var], feed_dict={self.image_path: img_path})
        rt = time.time() - start

        if pp:
            # postprocessing
            res_pp_var = post_process_disparity_var(var_our.squeeze())
            res_pp_mean = post_process_disparity(pred_our.squeeze())

            return res_pp_mean, res_pp_var, rt
        else:
            return pred_our, var_our, rt