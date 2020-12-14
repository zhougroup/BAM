import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d
eps = 1e-20
def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False, args=None, training=1.0):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        logprobs = tf.log(coefs + eps)
        if args.att_prior_type == 'contextual':
            kernel_initializer = tf.keras.initializers.he_normal() # glorot_normal
            if args.att_type == 'soft_weibull':
                if 0:
                    alpha_gamma = coefs
                else:
                    dot_gamma = tf.layers.dense(seq_fts, args.att_se_hid_size, activation=None, use_bias=True,
                                                kernel_initializer=kernel_initializer, bias_initializer=tf.zeros_initializer())
                    dot_gamma = tf.nn.relu(dot_gamma)
                    dot_gamma = tf.layers.dense(dot_gamma, 1, activation=None, use_bias=True,
                                                kernel_initializer=kernel_initializer, bias_initializer=tf.zeros_initializer())
                    dot_gamma = tf.transpose(dot_gamma, [0, 2, 1])
                    print('****************************dot_gamma shape', dot_gamma.get_shape().as_list())
                    alpha_gamma = tf.nn.softmax(dot_gamma, axis=-1) * args.beta_gamma
                prior_att_weights = alpha_gamma / tf.reduce_sum(alpha_gamma, axis=-1, keepdims=True)
                # TODO need a transpose here? get which one is key and which is query
            if args.att_type == 'soft_lognormal':
                dot_mu = tf.layers.dense(seq_fts, args.att_se_hid_size, activation=None, use_bias=True,
                                            kernel_initializer=kernel_initializer, bias_initializer=tf.zeros_initializer())
                dot_mu = tf.nn.relu(dot_mu)
                dot_mu = tf.layers.dense(dot_mu, 1, activation=None, use_bias=True,
                                            kernel_initializer=kernel_initializer, bias_initializer=tf.zeros_initializer())
                dot_mu = tf.transpose(dot_mu, [0, 2, 1])
                dot_mu = tf.nn.softmax(dot_mu)
                mean_normal_prior = tf.log(dot_mu + eps)
                prior_att_weights = dot_mu
        else:
            alpha_gamma = args.alpha_gamma
            mean_normal_prior = 0.0
        if args.att_type == 'soft_weibull':
            lambda_weibull = tf.exp(logprobs) / tf.exp(tf.lgamma(1 + 1.0 / args.k_weibull))
            u_weibull = tf.random_uniform(shape=logprobs.shape, dtype=logprobs.dtype)
            sample_weibull = lambda_weibull * (tf.exp(1.0 / args.k_weibull * tf.log(
                -tf.log(1.0 - u_weibull + eps) + eps)) * training + tf.exp(
                tf.lgamma(1.0 + 1.0 / args.k_weibull)) * (1 - training))
            sample_weibull = sample_weibull
            out_coefs = sample_weibull / tf.reduce_sum(sample_weibull, axis=-1, keepdims=True)
            KL = -(alpha_gamma * tf.log(lambda_weibull + eps) - np.euler_gamma * alpha_gamma / args.k_weibull \
                   - tf.log(args.k_weibull + eps) - args.beta_gamma * lambda_weibull * tf.exp(
                        tf.lgamma(1 + 1.0 / args.k_weibull)) + np.euler_gamma + 1.0 + \
                   alpha_gamma * tf.log(args.beta_gamma + eps) - tf.lgamma(alpha_gamma + eps))
            KL = KL * tf.cast(bias_mat > -1e7, KL.dtype)
            KL_backward = tf.reduce_sum(KL) / tf.reduce_sum(tf.cast(bias_mat > -1e7, KL.dtype))
            tf.add_to_collection('kl_list', KL_backward)

        elif args.att_type == 'soft_lognormal':
            mean_normal_posterior = logprobs - args.sigma_normal_posterior ** 2 / 2

            sample_normal = mean_normal_posterior + bias_mat + (args.sigma_normal_posterior * tf.random_normal(
                    shape=logprobs.shape, dtype=logprobs.dtype)) * training + (
                    args.sigma_normal_posterior ** 2 / 2) * (1-training)
            out_coefs = tf.nn.softmax(sample_normal)
            KL = tf.log(args.sigma_normal_prior / args.sigma_normal_posterior + eps) + (
                args.sigma_normal_posterior ** 2 + (mean_normal_prior - mean_normal_posterior) ** 2
            ) / (2 * args.sigma_normal_prior ** 2) - 0.5
            KL = KL * tf.cast(bias_mat > -1e7, KL.dtype)
            KL_backward = tf.reduce_sum(KL) / tf.reduce_sum(tf.cast(bias_mat > -1e7, KL.dtype))
            tf.add_to_collection('kl_list', KL_backward)
        else:
            out_coefs = coefs

        if coef_drop != 0.0:
            out_coefs = tf.nn.dropout(out_coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(out_coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
eps = 1e-20
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False, args=None,
                 training=1.0):
    use_bias = True
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)

        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat*f_1
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices,
                values=tf.nn.leaky_relu(logits.values),
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)
        eps_tensor = tf.SparseTensor(indices=logits.indices,
                                     values=tf.ones_like(logits.values, dtype=tf.float32) * eps,
                                     dense_shape=logits.dense_shape)
        one_tensor = tf.SparseTensor(indices=logits.indices,
                                     values=tf.ones_like(logits.values, dtype=tf.float32),
                                     dense_shape=logits.dense_shape)
        # eps_tensor = tf.cast(tf.constant([eps]), tf.float32)
        logprobs = tf.SparseTensor(indices=logits.indices,
                                   values=tf.log(tf.sparse_add(coefs, eps_tensor).values),
                                   dense_shape=logits.dense_shape)

        if args.att_prior_type == 'contextual':
            kernel_initializer = tf.keras.initializers.he_normal() # glorot_normal
            bias_initializer = tf.zeros_initializer()
            if args.att_type == 'soft_weibull':
                dot_gamma = tf.layers.dense(seq_fts, args.att_se_hid_size, activation=None, use_bias=use_bias,
                                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
                dot_gamma = tf.nn.relu(dot_gamma)
                dot_gamma = tf.layers.dense(dot_gamma, 1, activation=None, use_bias=use_bias,
                                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

                dot_gamma = tf.transpose(dot_gamma, [0, 2, 1])
                print('****************************dot_gamma shape', dot_gamma.get_shape().as_list())
                alpha_gamma = tf.nn.softmax(dot_gamma, axis=-1) * args.beta_gamma
                prior_att_weights = alpha_gamma / tf.reduce_sum(alpha_gamma, axis=-1, keepdims=True)
            if args.att_type == 'soft_lognormal':
                dot_mu = tf.layers.dense(seq_fts, args.att_se_hid_size, activation=None, use_bias=use_bias,
                                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
                dot_mu = tf.nn.relu(dot_mu)
                dot_mu = tf.layers.dense(dot_mu, 1, activation=None, use_bias=use_bias,
                                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
                dot_mu = tf.transpose(dot_mu, [0, 2, 1])
                dot_mu = tf.nn.softmax(dot_mu)
                mean_normal_prior = tf.log(dot_mu + eps)
                prior_att_weights = dot_mu
        else:
            alpha_gamma = args.alpha_gamma
            mean_normal_prior = 0.0

        if args.att_type == 'soft_weibull':
            rand_uniform = tf.random_uniform(shape=[108365])
            sample_weibull = tf.SparseTensor(indices=logits.indices,
                                             values=logprobs.values - tf.lgamma(1 + 1.0 / args.k_weibull) + tf.log(
                                                 -tf.log(tf.ones_like(logits.values) - rand_uniform + eps*tf.ones_like(logits.values)) + eps*tf.ones_like(logits.values)) / args.k_weibull,
                                             dense_shape=logits.dense_shape)
            lambda_weibull = tf.SparseTensor(indices=logits.indices,
                                             values=tf.exp(logprobs.values - tf.lgamma(1 + 1.0 / args.k_weibull)),
                                             dense_shape=logits.dense_shape)
            sample_weibull = tf.sparse_softmax(sample_weibull)
            mean_weibull = tf.sparse_softmax(logprobs)
            out_coefs = tf.SparseTensor(indices=logits.indices,
                                        values=sample_weibull.values * training + mean_weibull.values * (1 - training),
                                        dense_shape=logits.dense_shape)
            if args.att_prior_type == 'contextual':
                alpha_gamma_sparse = tf.SparseTensor(indices=logits.indices,
                                                     values=tf.gather_nd(tf.squeeze(tf.squeeze(alpha_gamma, 0), 0),
                                                                         tf.expand_dims(logits.indices[:, 1], axis=1)),
                                                     dense_shape=logits.dense_shape)

                KL_1 = tf.SparseTensor(indices=logits.indices,
                                       values=(logprobs.values - tf.lgamma(1 + 1.0 / args.k_weibull)) *
                                               alpha_gamma_sparse.values - args.beta_gamma * lambda_weibull.values *
                                               tf.exp(tf.lgamma(1 + 1.0 / args.k_weibull)),
                                       dense_shape=logits.dense_shape)
                KL_1_mean = tf.sparse_reduce_sum(KL_1) / tf.sparse_reduce_sum(one_tensor)
                KL_2_mean = tf.reduce_mean(- np.euler_gamma * alpha_gamma_sparse.values / args.k_weibull +
                        alpha_gamma_sparse.values * tf.log(args.beta_gamma + eps) -
                        tf.lgamma(alpha_gamma_sparse.values + eps))
                KL_backward = - (KL_1_mean + KL_2_mean)
            else:
                KL = tf.SparseTensor(indices=logits.indices,
                                     values=(logprobs.values - tf.lgamma(1 + 1.0 / args.k_weibull)) * alpha_gamma -
                                             args.beta_gamma * lambda_weibull.values *
                                             tf.exp(tf.lgamma(1 + 1.0 / args.k_weibull)),
                                     dense_shape=logits.dense_shape)
                KL_backward = - tf.sparse_reduce_sum(KL) / tf.sparse_reduce_sum(one_tensor)
            tf.add_to_collection('kl_list', KL_backward)

        elif args.att_type == 'soft_lognormal':
            mean_normal_posterior = logprobs.values - args.sigma_normal_posterior ** 2 / 2
            sample_normal_value = mean_normal_posterior + args.sigma_normal_posterior * tf.random_normal(
                shape=[108365], dtype=tf.float32
            )
            sample_normal = tf.SparseTensor(indices=logits.indices,
                                             values=sample_normal_value,
                                             dense_shape=logits.dense_shape)
            sample_normal = tf.sparse_softmax(sample_normal)
            mean_normal = tf.sparse_softmax(logprobs)
            out_coefs = tf.SparseTensor(indices=logits.indices,
                                        values=sample_normal.values * training + mean_normal.values * (1 - training),
                                        dense_shape=logits.dense_shape)
            if args.att_prior_type == 'contextual':
                mean_normal_prior_sparse = tf.SparseTensor(indices=logits.indices,
                                                           values=tf.gather_nd(tf.squeeze(tf.squeeze(mean_normal_prior, 0), 0),
                                                                               tf.expand_dims(logits.indices[:, 1], axis=1)),
                                                           dense_shape=logits.dense_shape)

                KL = tf.reduce_mean((args.sigma_normal_posterior ** 2 + (
                        mean_normal_prior_sparse.values - mean_normal_posterior) ** 2) / (
                        2 * args.sigma_normal_prior ** 2))
                KL_backward = KL
            else:
                # Only include terms that have gradients.
                KL = tf.reduce_mean((args.sigma_normal_posterior ** 2 + (
                        mean_normal_prior - mean_normal_posterior) ** 2) / (
                        2 * args.sigma_normal_prior ** 2))
                KL_backward = KL
            tf.add_to_collection('kl_list', KL_backward)
        else:
            out_coefs = coefs

        if coef_drop != 0.0:
            out_coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(out_coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
        out_coefs = tf.sparse_reshape(out_coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(out_coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

