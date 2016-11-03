import tensorflow as tf
import numpy as np


def build_activation_summary(x, Collection):
    tensor_name = x.op.name
    hs = tf.histogram_summary(tensor_name + '/activations', x)
    ss = tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    tf.add_to_collection(Collection + "_summaries", hs)
    tf.add_to_collection(Collection + "_summaries", ss)


def conv2d(x, W, stride, name):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID", name=name)


def xavier_std(in_size, out_size):
    return np.sqrt(2. / (in_size + out_size))


def get_var(name, size, initializer, Collection):
    w = tf.get_variable(name, size, initializer=initializer,
                        collections=[Collection + "_weights"])
    if tf.get_variable_scope().reuse == False:
        tf.add_to_collection(Collection + "_summaries",
                             tf.histogram_summary(w.op.name, w))
    return w


def add_conv_layer(head, channels, kernel_size, stride, Collection):
    assert len(head.get_shape()
               ) == 4, "can't add a conv layer to this input"
    layer_name = "conv" + \
        str(len(tf.get_collection(Collection + "_convolutions")))
    tf.add_to_collection(Collection + "_convolutions", layer_name)
    head_channels = head.get_shape().as_list()[3]
    w_size = [kernel_size, kernel_size, head_channels, channels]
    std = xavier_std(head_channels * kernel_size **
                     2, channels * kernel_size**2)

    w = get_var(layer_name + "_W", w_size, initializer=tf.truncated_normal_initializer(
        stddev=std), Collection=Collection)

    new_head = tf.nn.relu(
        conv2d(head, w, stride, name=layer_name), name=layer_name + "_relu")
    build_activation_summary(new_head, Collection + "_summaries")
    return new_head


def add_linear_layer(head, size, Collection, layer_name=None, weight_name=None):
    assert len(head.get_shape()
               ) == 2, "can't add a linear layer to this input"
    if layer_name == None:
        layer_name = "linear" + \
            str(len(tf.get_collection(Collection + "_linears")))
        tf.add_to_collection(Collection + "_linears", layer_name)
    if weight_name == None:
        weight_name = layer_name + "_W"
    head_size = head.get_shape().as_list()[1]
    w_size = [head_size, size]
    std = xavier_std(head_size, size)

    w = get_var(weight_name, w_size, initializer=tf.truncated_normal_initializer(
        stddev=std), Collection=Collection)

    new_head = tf.matmul(head, w, name=layer_name)
    build_activation_summary(new_head, Collection + "_summaries")
    return new_head


def add_relu_layer(head, size, Collection, layer_name=None, weight_name=None):
    if layer_name == None:
        layer_name = "relu" + \
            str(len(tf.get_collection(Collection + "_relus")))
        tf.add_to_collection(Collection + "_relus", layer_name)
    head = add_linear_layer(
        head, size, Collection, layer_name, weight_name)
    new_head = tf.nn.relu(head, name=layer_name + "_relu")
    build_activation_summary(new_head, Collection + "_summaries")
    return new_head

relu_layer_counter = [0]
conv_layer_counter = [0]
linear_layer_counter = [0]
conditional_linear_layer_counter = [0]

# for the multiple calls to share variable, all variable names must be the
# same evey call


def hidden_state_to_Q(hidden_state, _name, action_num, Collection):
    head = add_relu_layer(hidden_state, size=512, Collection=Collection,
                          layer_name="final_linear_" + _name, weight_name="final_linear_Q_W")
    # the last layer is linear without a relu
    head_size = head.get_shape().as_list()[1]

    Q_w = get_var("Q_W", [head_size, action_num], initializer=tf.truncated_normal_initializer(
        stddev=xavier_std(head_size, action_num)), Collection=Collection)

    Q = tf.matmul(head, Q_w, name=_name)
    tf.add_to_collection(Collection + "_summaries",
                         tf.histogram_summary(_name, Q))
    return Q


def createQNetwork(input_state, action, config, Collection=None):
    action_num = config.action_num
    normalized = input_state / 128. - 1.
    tf.add_to_collection(Collection + "_summaries", tf.histogram_summary(
        "normalized_input", normalized))

    head = add_conv_layer(normalized, channels=32,
                          kernel_size=8, stride=4, Collection=Collection)
    head = add_conv_layer(head, channels=64,
                          kernel_size=4, stride=2, Collection=Collection)
    head = add_conv_layer(head, channels=64,
                          kernel_size=3, stride=1, Collection=Collection)

    h_conv3_shape = head.get_shape().as_list()
    head = tf.reshape(
        head, [-1, h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3]], name="conv3_flat")

    hidden_state = head

    Q = hidden_state_to_Q(hidden_state, "Q", action_num, Collection)
    return Q


def clipped_l2(y, y_t, grad_clip=1):
    with tf.name_scope("clipped_l2"):
        batch_delta = y - y_t
        batch_delta_abs = tf.abs(batch_delta)
        batch_delta_quadratic = tf.minimum(batch_delta_abs, grad_clip)
        batch_delta_linear = (
            batch_delta_abs - batch_delta_quadratic) * grad_clip
        batch = batch_delta_linear + batch_delta_quadratic**2 / 2
    return batch


def build_train_op(Q, Y, action, config):
    action_num = config.action_num
    with tf.name_scope("loss"):
        # could be done more efficiently with gather_nd or transpose + gather
        action_one_hot = tf.one_hot(
            action, action_num, 1., 0., name='action_one_hot')
        DQN_acted = tf.reduce_sum(
            Q * action_one_hot, reduction_indices=1, name='DQN_acted')

        batch_loss = clipped_l2(Y, DQN_acted)
        loss = tf.reduce_sum(batch_loss, name="Q_loss")

        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "losses/Q_0", batch_loss[0]))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "losses/Q", loss))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "losses/Q_max", tf.reduce_max(batch_loss)))

        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "main/Y_0", Y[0]))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "main/acted_Q_0", DQN_acted[0]))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "main/acted_Q_max", tf.reduce_max(DQN_acted)))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "main/Y_max", tf.reduce_max(Y)))

    #opti = tf.train.RMSPropOptimizer(config.learning_rate, 0.95, 0.95, 0.01)
    #opti = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate, decay=0.95, momentum=0.0, epsilon=0.01)
    train_op, grads = build_rmsprop_optimizer(
        loss, config.learning_rate, 0.95, 0.01, 1, "graves_rmsprop")
    #grads = opti.compute_gradients(loss)

    #train_op = opti.apply_gradients(grads)

    for grad, var in grads:
        if grad is not None:
            tf.add_to_collection("DQN_summaries", tf.histogram_summary(
                var.op.name + '/gradients', grad, name=var.op.name + '/gradients'))

    return train_op


def build_rmsprop_optimizer(loss, learning_rate, rmsprop_decay, rmsprop_constant, gradient_clip, version):
    with tf.name_scope('rmsprop'):
        optimizer = None
        if version == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate, decay=rmsprop_decay, momentum=0.0, epsilon=rmsprop_constant)
        elif version == 'graves_rmsprop':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        grads_and_vars = optimizer.compute_gradients(loss)

        grads = []
        params = []
        for p in grads_and_vars:
            if p[0] == None:
                continue
            grads.append(p[0])
            params.append(p[1])
        #grads = [gv[0] for gv in grads_and_vars]
        #params = [gv[1] for gv in grads_and_vars]
        if gradient_clip > 0:
            grads = tf.clip_by_global_norm(grads, gradient_clip)[0]

        if version == 'rmsprop':
            return optimizer.apply_gradients(zip(grads, params))
        elif version == 'graves_rmsprop':
            square_grads = [tf.square(grad) for grad in grads]

            avg_grads = [tf.Variable(tf.zeros(var.get_shape()))
                         for var in params]
            avg_square_grads = [tf.Variable(
                tf.zeros(var.get_shape())) for var in params]

            update_avg_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * grad_pair[1]))
                                for grad_pair in zip(avg_grads, grads)]
            update_avg_square_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * tf.square(grad_pair[1])))
                                       for grad_pair in zip(avg_square_grads, grads)]
            avg_grad_updates = update_avg_grads + update_avg_square_grads

            rms = [tf.sqrt(avg_grad_pair[1] - tf.square(avg_grad_pair[0]) + rmsprop_constant)
                   for avg_grad_pair in zip(avg_grads, avg_square_grads)]

            rms_updates = [grad_rms_pair[0] / grad_rms_pair[1]
                           for grad_rms_pair in zip(grads, rms)]
            train = optimizer.apply_gradients(zip(rms_updates, params))

            return tf.group(train, tf.group(*avg_grad_updates)), grads_and_vars
