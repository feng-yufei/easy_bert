import os
import modeling
import optimization
import tensorflow as tf
from load_data import DataLoader
import numpy as np
from time import time
import datetime


# def mgpu_train(n_gpu, bert_config, train_flag, X, M, I, Y):
#    for i in range(n_gpu):
#        do_reuse = True if i >= 1 else False
#        with tf.device("/gpu:%d" % i), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
#            (total_loss, per_example_loss, logits) = create_model(bert_config, train_flag, X, M, I, Y, 3)


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels):
    """Creates a classification model."""
    # is training is now changed to tensor
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

    # If you want to use the token-level output, use model.get_sequence_output() instead.
    output_layer = model.get_pooled_output()  # modeling.py line222: [batch_size, hidden_size] of the first token
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        # if is_training:
        # i.e., 0.1 dropout
        output_layer = tf.layers.dropout(output_layer, 0.1, training=is_training)  # if train: dropout

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)  # softmax(output_layer*output_w+output_b)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)  # logits_predict

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)  # true label

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)  # *(-1)
        loss = tf.reduce_mean(per_example_loss)  # cal loss

        return (loss, per_example_loss, logits)


def run_epoch(sess, logits, clf_losses, data_iterator, phase, batch_size=16, train_op=tf.constant(0)):
    t_correct = 0
    t_loss = 0
    n_all = 0
    t0 = time()

    for x_batch, m_batch, i_batch, y_batch in data_iterator.sampled_batch(batch_size=batch_size, phase=phase):
        batch_pred, batch_loss, _ = sess.run([logits, clf_losses, train_op],
                                             feed_dict={X: x_batch,
                                                        M: m_batch, I: i_batch,
                                                        Y: y_batch, train_flag: phase == 'train'})

        n_sample = y_batch.shape[0]
        n_all += n_sample
        t_loss += batch_loss * n_sample

        t_correct += np.sum(np.argmax(batch_pred, axis=1) == y_batch)

    print("{} Loss: {:.4f},  Accuarcy: {:.2f}%, {:.2f} Seconds Used:".
          format(phase, t_loss / n_all, 100 * t_correct / n_all, time() - t0))


if __name__ == "__main__":

    # default for all
    do_lower_case = True
    warmup_proportion = 0.1

    # path to the pretrained bert model
    vocab_file = 'uncased_L-12_H-768_A-12/vocab.txt'
    bert_config_file = 'uncased_L-12_H-768_A-12/bert_config.json'
    init_checkpoint = 'uncased_L-12_H-768_A-12/bert_model.ckpt'

    max_word = 128
    train_batch_size = 32
    eval_batch_size = train_batch_size
    learning_rate = 2e-5
    label_size = 3
    n_epochs = 10

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    data_iterator = DataLoader()

    # traininig sample size and warming up
    x = data_iterator.n_train
    num_train_steps = int(x / train_batch_size * n_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)

    # placeholder
    X = tf.placeholder(tf.int32, [None, max_word])
    M = tf.placeholder(tf.int32, [None, max_word])
    I = tf.placeholder(tf.int32, [None, max_word])
    Y = tf.placeholder(tf.int32, [None])

    train_flag = tf.placeholder_with_default(True, shape=())

    # create model (graph)
    (total_loss, per_example_loss, logits) = create_model(bert_config, train_flag, X, M, I, Y, label_size)

    # init model with pre-training
    tvars = tf.trainable_variables()
    if init_checkpoint:
        (assignment_map, initialized_variable_names) = \
            modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # --------------  data_iterator -------------

    train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps,
                                             False)  # Adam

    # session & initialize
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # saving module
    saver = tf.train.Saver()

    print('Start Training ... ')
    for i in range(0, n_epochs):
        print('Epoch {}...'.format(i))
        run_epoch(sess, logits, total_loss, data_iterator, 'train', batch_size=train_batch_size, train_op=train_op)
        run_epoch(sess, logits, total_loss, data_iterator, 'validation', batch_size=eval_batch_size)
        run_epoch(sess, logits, total_loss, data_iterator, 'test', batch_size=eval_batch_size)

        if i in [0, 2, 4, 9]:
            c_time = str(datetime.datetime.now()).replace(' ', '-').split('.')[0]
            saver.save(sess, './saved_models/bert_model-{}.ckpt'.format(c_time), global_step=i + 1)
