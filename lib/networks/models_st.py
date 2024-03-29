#coding:utf-8
#---------------------------------------------------
# STC-Graph generation
# Based on SGG Written by Danfei Xu
#---------------------------------------------------


import tensorflow as tf
from networks.network import Network
import losses
from fast_rcnn.config import cfg
import net_utils as utils


class basenet(Network):
    def __init__(self, data):
        self.inputs = []
        self.data = data
        self.ims = data['ims']
        self.rois = data['rois']
        self.iterable = False
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = {}
        self.batch = cfg.TRAIN.IMS_PER_BATCH 
        # self.batch = cfg.TEST.IMS_PER_BATCH 
    def _vgg16(self):
        self.layers = dict({'ims': self.ims, 'rois': self.rois}) # 
        self._vgg_conv()
        self._vgg_fc()

    def _vgg_conv(self):
        (self.feed('ims')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3')
             .stop_gradient(name='conv_out'))

    def _vgg_fc(self): 
        (self.feed('conv_out', 'rois') 
             .roi_pool(7, 7, 1.0/16, name='pool5')
             .fc(4096, name='fc6')
             .dropout(self.keep_prob, name='drop6')
             .fc(4096, name='fc7')
             .dropout(self.keep_prob, name='vgg_out'))

    def _union_rel_vgg_fc(self):
        (self.feed('conv_out', 'rel_rois')
             .roi_pool(7, 7, 1.0/16, name='rel_pool5')
             .fc(4096, name='rel_fc6')
             .dropout(self.keep_prob, name='rel_drop6')
             .fc(4096, name='rel_fc7')
             .dropout(self.keep_prob, name='rel_vgg_out'))

    def _cls_pred(self, input_layer, layer_suffix='', reuse=False, new_var=False):
        layer_name = 'cls_score'+layer_suffix if new_var else 'cls_score'
        print(layer_name)
        (self.feed(input_layer)
             .fc(self.data['num_classes'], relu=False, name=layer_name,
                 reuse=reuse)
             .softmax(name='cls_prob'+layer_suffix)
             .argmax(1, name='cls_pred'+layer_suffix))


    def _bbox_pred(self, input_layer, layer_suffix='', reuse=False, new_var=False):
        layer_name = 'bbox_pred'+layer_suffix if new_var else 'bbox_pred'
        (self.feed(input_layer)
             .fc(self.data['num_classes']*4, relu=False, name=layer_name,
                 reuse=reuse))

    def _rel_pred(self, input_layer, layer_suffix='', reuse=False, new_var=False):
        layer_name = 'rel_score'+layer_suffix if new_var else 'rel_score'
        (self.feed(input_layer)
             .fc(self.data['num_predicates'], relu=False, name=layer_name,
                 reuse=reuse)
             .softmax(name='rel_prob'+layer_suffix)
             .argmax(1, name='rel_pred'+layer_suffix))
    # Losses
    def _sg_losses(self, ops={}, suffix=''):
        ops = self._frc_losses(ops, suffix)
        rel_score = self.get_output('rel_score'+suffix)
        ops['loss_rel'+suffix] =  losses.loss_sigmoid(rel_score, self.data['predicates'],
                                                name='rel_loss'+suffix, ignore_bg=False)
        return ops

    def _frc_losses(self, ops={}, suffix=''):
        # classification loss
        cls_score = self.get_output('cls_score'+suffix)
        ops['loss_cls'+suffix] = losses.loss_sigmoid(cls_score, self.data['labels'], name='cls_loss'+suffix)

        # bounding box regression L1 loss
        if cfg.TRAIN.BBOX_REG:
            bbox_pred = self.get_output('bbox_pred'+suffix)
            ops['loss_box'+suffix]  = 0*losses.l1_loss(bbox_pred, self.data['bbox_targets'], 'reg_loss'+suffix,
                                              self.data['bbox_inside_weights'])
        else:
            print('NO BBOX REGRESSION!!!!!')
        return ops

    def cls_pred_output(self, iters=None):
        if iters is not None:
            op = {}
            for i in iters:
                if self.iterable and i != self.n_iter - 1:
                    op[i] = self.get_output('cls_prob_iter%i' % i)
                else:
                    op[i] = self.get_output('cls_prob')

        else:
            op = self.get_output('cls_prob')
        return op

    def bbox_pred_output(self, iters=None):
        if iters is not None:
            op = {}
            for i in iters:
                op[i] = self.get_output('bbox_pred')

        else:
            op = self.get_output('bbox_pred')
        return op

    def rel_pred_output(self, iters=None):
        if iters is not None:
            op = {}
            for i in iters:
                if self.iterable and i != self.n_iter - 1:
                    op[i] = self.get_output('rel_prob_iter%i' % i)
                else:
                    op[i] = self.get_output('rel_prob')

        else:
            op = self.get_output('rel_prob')
        return op


class dual_graph_vrd(basenet):
    def __init__(self, data):
        basenet.__init__(self, data)

        self.num_roi = data['num_roi']
        self.num_rel = data['num_rel']# 9*num_roi
        self.rel_rois = data['rel_rois']
        self.iterable = True

        self.edge_mask_inds = data['rel_mask_inds']
        self.edge_segment_inds = data['rel_segment_inds']

        self.edge_pair_mask_inds = data['rel_pair_mask_inds']
        self.edge_pair_segment_inds = data['rel_pair_segment_inds']

        # number of refine iterations
        self.n_iter = data['n_iter']
        self.relations = data['relations']

        self.vert_state_dim = 512
        self.edge_state_dim = 512

    def setup(self):
        self.layers = dict({'ims': self.ims, 'rois': self.rois, 'rel_rois': self.rel_rois})
        self._vgg_conv()
        self._vgg_fc()
        self._union_rel_vgg_fc()
        self._cells()
        self._iterate()

    def _cells(self):
        """
        construct RNN cells and states
        """
        # intiialize lstms
        self.vert_rnn = tf.nn.rnn_cell.GRUCell(self.vert_state_dim, activation=tf.tanh)
        self.edge_rnn = tf.nn.rnn_cell.GRUCell(self.edge_state_dim, activation=tf.tanh)

        # lstm states
        self.vert_state = self.vert_rnn.zero_state(self.num_roi, tf.float32)
        self.edge_state = self.edge_rnn.zero_state(self.num_rel, tf.float32)
    def testforme(self):
        return self.get_output('cls_pred')
    def getver(self):
        return self.get_output('vert_unary')
    def getedge(self):
        return self.get_output('edge_unary')
    def getcls(self):
        return self.get_output('cls_score'), self.get_output('rel_score')
    def _iterate(self): ## 
        (self.feed('vgg_out')
             .fc(self.vert_state_dim, relu=False, name='vert_unary'))

        (self.feed('rel_vgg_out')
             .fc(self.edge_state_dim, relu=True, name='edge_unary'))


        vert_unary = self.get_output('vert_unary')
        edge_unary = self.get_output('edge_unary') # concat with tem_rel_vgg_out
        
        #### input: vert_unary
        temp_edge_unary = self.get_temp_edge(vert_unary,self.data['img_to_roi_idx'],self.batch)
        edge_unary = tf.concat(0,[edge_unary,temp_edge_unary])
        ####
        vert_factor = self._vert_rnn_forward(vert_unary, reuse=False)
        edge_factor = self._edge_rnn_forward(edge_unary, reuse=False)


        for i in xrange(self.n_iter):
            reuse = i > 0
            # compute edge states
            edge_ctx = self._compute_edge_context(vert_factor, edge_factor, reuse=reuse)
            edge_factor = self._edge_rnn_forward(edge_ctx, reuse=True)

            # compute vert states
            vert_ctx = self._compute_vert_context(edge_factor, vert_factor, reuse=reuse)
            vert_factor = self._vert_rnn_forward(vert_ctx, reuse=True)
            vert_in = vert_factor
            edge_in = edge_factor

            self._update_inference(vert_in, edge_in, i)
    def get_temp_edge(self,vert_unary,idx,BATCH):
        initial_outputs = tf.TensorArray(dtype=tf.float32, size=BATCH, dynamic_size=True, infer_shape=False)

        def cond(size,i,outputs_):
            return tf.less(i,size)
        def body(size,i,outputs_):
            dims=tf.to_float(idx[i,1])
            slice_output1=tf.slice(vert_unary,[idx[i,0],0],[tf.to_int32(tf.ceil(dims*9/10)),512])
            slice_output2=tf.slice(vert_unary,[idx[i,0]+tf.to_int32(tf.ceil(dims*1/10)),0],[tf.to_int32(tf.ceil(dims*9/10)),512])
            slice_output_=tf.concat(1,[slice_output1,slice_output2])
            slice_output_=tf.reshape(slice_output_,[-1,1024])

            outputs_=outputs_.write(i,slice_output_)

            return size,i+1,outputs_
        i = tf.cast(tf.constant(0),tf.int32)
        size = tf.cast(tf.constant(BATCH),tf.int32)

        _,i,outputs_ = tf.while_loop(cond, body, [size,i,initial_outputs])
        
        concat_vert = tf.reshape(outputs_.concat(),[-1,1024])
        w1 = tf.Variable(tf.random_normal([1024, 512]))
        b1 = tf.Variable(tf.zeros([1, 512]) + 0.1)
        temp_edge_unary = tf.matmul(concat_vert,w1) + b1
        return temp_edge_unary
    def get_temp_edge_fialed(self,vert_unary,idx,BATCH):
        def cond(size,i):
            return tf.less(i,size)
        def body(size,i):
            slice_c=tf.slice(vert_unary,[idx[i,0],0],[idx[i,1],-1])
            dims=idx[i,1]
            slice_output1=tf.slice(slice_c,[0,0],[tf.to_int32(dims*4/5),-1])
            slice_output2=tf.slice(slice_c,[tf.to_int32(dims*1/5),0],[-1,-1])
            slice_output_=tf.concat(1,[slice_output1,slice_output2])
            
            indices=tf.range(idx[i,0],idx[i,0]+tf.to_int32((idx[i,1])*4/5))

            slice_output=tf.get_variable("slice_output",[tf.shape(vert_unary)[0],tf.shape(vert_unary)[1]*2], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            slice_output=tf.scatter_update(slice_output,indices,slice_output_)
            

            tf.get_variable_scope().reuse_variables() # Reuse variables
            with tf.control_dependencies([slice_output]):
                return (size,i+1)


        i = tf.cast(tf.constant(0),tf.int32)
        size = tf.cast(tf.constant(BATCH),tf.int32)

        _,i = tf.while_loop(cond, body, [size,i])
        concat_vert = tf.get_variable("slice_output",[vert_unary.get_shape()[0],vert_unary.get_shape()[1]*2], dtype=tf.float32)
        w1 = tf.Variable(tf.random.truncated_normal([vert_unary.get_shape()[1]*2,vert_unary.get_shape()[1]], stddev=0.1))
        b1 = tf.Variable(tf.zeros([vert_unary.get_shape()[1]*2]))
        temp_edge_unary = tf.matmul(x,w1) + b1
        return temp_edge_unary
    def _compute_edge_context_hard(self, vert_factor, reduction_mode='max'):
        """
        max or average message pooling
        """
        if reduction_mode=='max':
            return tf.reduce_max(tf.gather(vert_factor, self.relations), [1])
        elif reduction_mode=='mean':
            return tf.reduce_mean(tf.gather(vert_factor, self.relations), [1])

    def _compute_vert_context_hard(self, edge_factor, vert_factor, reduction_mode='max'):
        """
        max or average message pooling
        """
        edge_factor_gathered = utils.pad_and_gather(edge_factor, self.edge_mask_inds, None)

        vert_ctx = utils.padded_segment_reduce(edge_factor_gathered, self.edge_segment_inds,
                                               vert_factor.get_shape()[0], reduction_mode)

        return vert_ctx

    def _compute_edge_context_soft(self, vert_factor, edge_factor, reuse=False):
        """
        attention-based edge message pooling
        """
        vert_pairs = utils.gather_vec_pairs(vert_factor, self.relations)

        sub_vert, obj_vert = tf.split(split_dim=1, num_split=2, value=vert_pairs)
        sub_vert_w_input = tf.concat(concat_dim=1, values=[sub_vert, edge_factor])
        obj_vert_w_input = tf.concat(concat_dim=1, values=[obj_vert, edge_factor])

        # compute compatibility scores
        (self.feed(sub_vert_w_input)
             .fc(1, relu=False, reuse=reuse, name='sub_vert_w_fc')
             .sigmoid(name='sub_vert_score'))
        (self.feed(obj_vert_w_input)
             .fc(1, relu=False, reuse=reuse, name='obj_vert_w_fc')
             .sigmoid(name='obj_vert_score'))

        sub_vert_w = self.get_output('sub_vert_score')
        obj_vert_w = self.get_output('obj_vert_score')

        weighted_sub = tf.mul(sub_vert, sub_vert_w)
        weighted_obj = tf.mul(obj_vert, obj_vert_w)
        return weighted_sub + weighted_obj

    def _compute_vert_context_soft(self, edge_factor, vert_factor, reuse=False):
        """
        attention-based vertex(node) message pooling
        """

        out_edge = utils.pad_and_gather(edge_factor, self.edge_pair_mask_inds[:,0])
        in_edge = utils.pad_and_gather(edge_factor, self.edge_pair_mask_inds[:,1])
        # gather correspounding vert factors
        vert_factor_gathered = tf.gather(vert_factor, self.edge_pair_segment_inds)

        # concat outgoing edges and ingoing edges with gathered vert_factors
        out_edge_w_input = tf.concat(concat_dim=1, values=[out_edge, vert_factor_gathered])
        in_edge_w_input = tf.concat(concat_dim=1, values=[in_edge, vert_factor_gathered])

        # compute compatibility scores
        (self.feed(out_edge_w_input)
             .fc(1, relu=False, reuse=reuse, name='out_edge_w_fc')
             .sigmoid(name='out_edge_score'))
        (self.feed(in_edge_w_input)
             .fc(1, relu=False, reuse=reuse, name='in_edge_w_fc')
             .sigmoid(name='in_edge_score'))

        out_edge_w = self.get_output('out_edge_score')
        in_edge_w = self.get_output('in_edge_score')

        # weight the edge factors with computed weigths
        out_edge_weighted = tf.mul(out_edge, out_edge_w)
        in_edge_weighted = tf.mul(in_edge, in_edge_w)


        edge_sum = out_edge_weighted + in_edge_weighted
        vert_ctx = tf.segment_sum(edge_sum, self.edge_pair_segment_inds)
        return vert_ctx

    def _vert_rnn_forward(self, vert_in, reuse=False):
        with tf.variable_scope('vert_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()
            (vert_out, self.vert_state) = self.vert_rnn(vert_in, self.vert_state)
        return vert_out

    def _edge_rnn_forward(self, edge_in, reuse=False):
        with tf.variable_scope('edge_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()
            (edge_out, self.edge_state) = self.edge_rnn(edge_in, self.edge_state)
        return edge_out

    def _update_inference(self, vert_factor, edge_factor, iter_i):
        # make predictions
        reuse = iter_i > 0  # reuse variables

        iter_suffix = '_iter%i' % iter_i if iter_i < self.n_iter - 1 else ''
        self._cls_pred(vert_factor, layer_suffix=iter_suffix, reuse=reuse)
        self._bbox_pred(vert_factor, layer_suffix=iter_suffix, reuse=reuse)
        self._rel_pred(edge_factor, layer_suffix=iter_suffix, reuse=reuse)

    def losses(self):
        return self._sg_losses()


class vrd(basenet):
    """
    Baseline: the visual relation detection module proposed by
    Lu et al.
    """

    def __init__(self, data):
        basenet.__init__(self, data)
        self.rel_rois = data['rel_rois']

    def setup(self):
        self.layers = dict({'ims': self.ims, 'rois': self.rois, 'rel_rois': self.rel_rois})
        self._vgg_conv()
        self._vgg_fc()
        self._union_rel_vgg_fc()
        self._cls_pred('vgg_out')
        self._bbox_pred('vgg_out')
        self._rel_pred('rel_vgg_out')

    def losses(self):
        return self._sg_losses()


class dual_graph_vrd_maxpool(dual_graph_vrd):
    """
    Baseline: context-pooling by max pooling
    """
    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_hard(vert_factor, reduction_mode='max')

    def _compute_vert_context(self, edge_factor, vert_factor, reuse):
        return self._compute_vert_context_hard(edge_factor, vert_factor, reduction_mode='max')


class dual_graph_vrd_avgpool(dual_graph_vrd):
    """
    Baseline: context-pooling by avg. pooling
    """
    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_hard(vert_factor, reduction_mode='mean')

    def _compute_vert_context(self, edge_factor, vert_factor, reuse):
        return self._compute_vert_context_hard(edge_factor, vert_factor, reduction_mode='mean')


class dual_graph_vrd_final(dual_graph_vrd):
    """
    Our final model: context-pooling by attention
    """
    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_soft(vert_factor, edge_factor, reuse)

    def _compute_vert_context(self, edge_factor, vert_factor, reuse):
        return self._compute_vert_context_soft(edge_factor, vert_factor, reuse)
