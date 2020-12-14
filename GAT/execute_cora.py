import time
import numpy as np
import tensorflow as tf

from models import GAT
from utils import process
import argparse
import torch
import scipy.stats as sts
import random

def two_sample_test_batch(logits):
    prob = torch.softmax(logits, 1)
    probmean = torch.mean(prob,2)
    values, indices = torch.topk(probmean, 2, dim=1)
    aa = logits.gather(1, indices[:,0].unsqueeze(1).unsqueeze(1).repeat(1,1,args.sample_num))
    bb = logits.gather(1, indices[:,1].unsqueeze(1).unsqueeze(1).repeat(1,1,args.sample_num))
    pvalue = sts.ttest_rel(aa,bb, axis=2).pvalue
    index = (torch.abs(aa - bb)).sum(1).sum(1) == 0
    pvalue[index] = 1
    return pvalue


def setup_seed(seed):
    tf.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--dataset', type=str, default='cora', help='pubmed, cora, citeseer')
parser.add_argument('--id', type=str, default='default')
parser.add_argument('--seed', type=int, default=None)


# Variational attention
parser.add_argument('--att_type', type=str, default='soft_attention',
                    help='soft_attention, soft_weibull, soft_lognormal')
parser.add_argument('--k_weibull', type=float, default=1000.0,
                    help='initialization of k in weibull distribution.')
parser.add_argument('--att_kl', type=float, default=0.0,
                    help='weights for KL term in variational attention.')

parser.add_argument('--w_1', type=float, default=1.0,
                    help='weights for hid_k.')
parser.add_argument('--b_1', type=float, default=0.0,
                    help='bias for hid_k.')

parser.add_argument('--w_2', type=float, default=1.0,
                    help='weights for k_weibull.')

parser.add_argument('--b_2', type=float, default=0.0,
                    help='bias for k_weibull.')

parser.add_argument('--w_3', type=float, default=1.0,
                    help='weights for k_weibull.')

parser.add_argument('--b_3', type=float, default=0.0,
                    help='bias for k_weibull.')


parser.add_argument('--k_parameterization', type=str, default='blue',
                    help='parameterization type for k')


parser.add_argument('--kl_anneal_rate', type=float, default=1.0,
                    help='KL anneal rate.')

parser.add_argument('--patience', type=int, default=100, help='Number of epochs for patient.')

parser.add_argument('--att_prior_type', type=str, default='constant',
                    help='contextual, constant, parameter, which type of prior used in variational attention.')
parser.add_argument('--alpha_gamma', type=float, default=1.0,
                    help='initialization of alpha in gamma distribution.')
parser.add_argument('--beta_gamma', type=float, default=1.0,
                    help='initialization of beta in gamma distribution.')
parser.add_argument('--sigma_normal_prior', type=float, default=1.0,
                    help='initialization of sigma in prior normal distribution.')
parser.add_argument('--sigma_normal_posterior', type=float, default=1.0,
                    help='initialization of sigma in posterior normal distribution.')
parser.add_argument('--att_contextual_se', type=int, default=0,
                        help='whether to use squeeze and excite in prior computation.')
parser.add_argument('--att_se_hid_size', type=int, default=10,
                    help='squeeze and excite factor in attention prior.')
parser.add_argument('--att_se_nonlinear', type=str, default='relu',
                    help='which type nonlinearity in se unit.')
parser.add_argument('--sample_num', type=int, default=2,
                    help='sample number to obtain uncertainty.')
parser.add_argument('--eps_2', type=float, default=1.0,
                    help='weight for exp(-phi) to balance mean and variance.')
parser.add_argument('--eps_1', type=float, default=1.0,
                    help='weight for exp(phi) to balance mean and variance.')
parser.add_argument('--eps_b', type=float, default=1.0,
                    help='eps bias.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='dropout.')
parser.add_argument('--l2_coef', type=float, default=5e-4,
                    help='l2_coef.')

args = parser.parse_args()

checkpt_file = 'pre_trained/' + args.id + args.dataset + '.ckpt'

dataset = args.dataset

if args.seed is not None:
    setup_seed(args.seed)
# training params
batch_size = 1
nb_epochs = args.epochs
patience = args.patience
lr = args.lr # learning rate
l2_coef = args.l2_coef  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

adj = adj.todense()

features = features[np.newaxis]
adj = adj[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

with tf.Graph().as_default():
    if args.seed is not None:
        tf.set_random_seed(args.seed)
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.float32, shape=())
        epoch_num = tf.placeholder(dtype=tf.float32, shape=())

    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity, args=args)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    # loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    loss_origin = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    if args.att_type != 'soft_attention':
        KL_loss = tf.add_n(tf.get_collection('kl_list')) / len(tf.get_collection('kl_list'))
        KL_loss = tf.exp(epoch_num * args.kl_anneal_rate) / (1 + tf.exp(epoch_num * args.kl_anneal_rate)) * KL_loss
        # loss = loss + KL_loss
        loss = loss_origin + KL_loss * args.att_kl
    else:
        loss = loss_origin

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = features.shape[0]

            while tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                    feed_dict={
                        ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
                        lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: 1.0,
                        attn_drop: args.dropout, ffd_drop: args.dropout, epoch_num: epoch})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                loss_value_vl, acc_vl = sess.run([loss_origin, accuracy],
                # loss_value_vl, acc_vl = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: biases[vl_step*batch_size:(vl_step+1)*batch_size],
                        lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: 0.0,
                        attn_drop: 0.0, ffd_drop: 0.0, epoch_num: epoch})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        ts_size = features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            loss_value_ts, acc_ts, logits_pred = sess.run([loss, accuracy, log_resh],
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: 0.0,
                    attn_drop: 0.0, ffd_drop: 0.0, epoch_num: epoch})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        # uncertainty estimation
        output_sample = np.zeros(shape=[nb_nodes, nb_classes, args.sample_num])
        for i in range(args.sample_num):
            ts_size = features.shape[0]
            ts_step = 0
            ts_loss = 0.0
            ts_acc = 0.0

            while ts_step * batch_size < ts_size:
                logits_reshape = sess.run([log_resh],
                    feed_dict={
                        ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                        bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                        lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                        msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                        is_train: 1.0,
                        attn_drop: 0.6, ffd_drop: 0.6, epoch_num: epoch})
                ts_loss += loss_value_ts
                ts_acc += acc_ts
                ts_step += 1
            output_sample[:, :, i] = logits_reshape[0]


        test_mask = test_mask[0] # node_size
        y_test = y_test[0]
        y_test = torch.from_numpy(np.argmax(y_test, axis=1)).type(torch.float32) # node_size

        output_sample = output_sample[test_mask]
        output_sample = torch.from_numpy(output_sample)
        testresult = torch.from_numpy(two_sample_test_batch(output_sample))
        print('shape', output_sample.shape, test_mask.shape, y_test.shape, logits_pred.shape)
        logits_pred = torch.from_numpy(logits_pred).type(torch.float32)

        prediction = torch.argmax(logits_pred[test_mask], 1).type_as(logits_pred)  # TODO: use greedy or sample?
        # print(prediction == y_test[test_mask])
        accurate_pred = (prediction == y_test[test_mask]).type_as(logits_pred)
        uncertain = (testresult > 0.01).type_as(logits_pred)

        ac_1 = (accurate_pred * (1 - uncertain.squeeze())).sum()
        iu_1 = ((1 - accurate_pred) * uncertain.squeeze()).sum()

        uncertain = (testresult > 0.05).type_as(logits_pred)
        ac_2 = (accurate_pred * (1 - uncertain.squeeze())).sum()
        iu_2 = ((1 - accurate_pred) * uncertain.squeeze()).sum()

        uncertain = (testresult > 0.1).type_as(logits_pred)
        ac_3 = (accurate_pred * (1 - uncertain.squeeze())).sum()
        iu_3 = ((1 - accurate_pred) * uncertain.squeeze()).sum()

        base_aic_1 = (ac_1 + iu_1) / accurate_pred.size(0) * 100
        base_aic_2 = (ac_2 + iu_2) / accurate_pred.size(0) * 100
        base_aic_3 = (ac_3 + iu_3) / accurate_pred.size(0) * 100

        print("Test set uncertainty results:",
              "uncertainty= {:.4f}".format(base_aic_1.data.item()),
              "uncertainty= {:.4f}".format(base_aic_2.data.item()),
              "uncertainty= {:.4f}".format(base_aic_3.data.item()))
        print('epoch', epoch)


        print('id', args.id, 'seed', args.seed, 'Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        sess.close()
