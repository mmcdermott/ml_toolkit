import tensorflow as tf, numpy as np, pandas as pd, math

DISTANCES = ['euc', 'man', 'cos']
DISTANCE_MAPPINGS = {'euc': 'euclidean', 'man': 1}

# Shape Constructions
def shape(X, static=True):
    if type(X) == pd.DataFrame: return X.shape
    elif type(X) == tf.Tensor:
        return X.get_shape().as_list() if static else tf.shape(X)
    raise TypeError("shape() only supports Tensor & DataFrame inputs; X is of type %s" % str(type(X)))

def get_dim(X, static=True): return shape(X, static=static)[1]
def num_samples(X, static=False): return shape(X, static=static)[0]

def make_compatible(*tensors):
    max_num_samples = tf.reduce_max(map(num_samples, tensors))

    #TODO(mmd): Upsample properly
    pad = lambda T: tf.reshape(tf.pad(T, [[0, max_num_samples - num_samples(T)], [0, 0]], mode='Symmetric'),
        [max_num_samples, get_dim(T, static=True)])
    return map(pad, tensors)

# Distances
def _cosine_distance(X, Y, pad=1e-7):
    return 1 - tf.reduce_mean((pad + tf.reduce_sum(X*Y, axis=1))/(pad + tf.norm(X, axis=1)*tf.norm(Y, axis=1)))

def dist(X, Y, ord='euc'):
    assert ord in DISTANCES

    if ord == 'cos': return _cosine_distance(X, Y)
    else: return tf.reduce_mean(tf.norm(X - Y, axis=1, ord=DISTANCE_MAPPINGS[ord]))

# Centering data:
def center(sample_df, axis=0):
    sample_mean = np.mean(sample_df, axis=axis)
    return (
        lambda X: X - sample_mean,
        lambda Y: Y + sample_mean,
    )

# Neural Network Constructions
def leaky_relu(X, alpha=0.2): return tf.maximum(alpha*X, X)
def linear(
      X, out_dim, scope,
      weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.2),
      bias_initializer=tf.constant_initializer(0.0),
):
    source_dim = get_dim(X)

    with tf.variable_scope(scope):
        weights = tf.get_variable('weights', shape=[source_dim, out_dim], initializer=weights_initializer)
        bias = tf.get_variable('bias', shape=[out_dim], initializer=bias_initializer)

    return tf.matmul(X, weights) + bias

def _feedforward_step(
    X, out_dim, scope,
    skip_connections    = False,
    activation          = tf.nn.relu,
    batch_normalization = None,
    training            = True,
):
    # For now only apply skip connections if out_dim == in_dim
    l = linear(X, out_dim, scope)
    if batch_normalization:
        l = tf.layers.batch_normalization(l, center=True, scale=True, is_training=training, scope=scope)
    ff = activation(l)
    if skip_connections and get_dim(X) == out_dim: return X + ff
    else: return ff

# TODO(mmd): Use enums for dim_change options.
# TODO(mmd): Dropout on and output_layer = False may not make any sense...
def feedforward(
    X, out_dim,
    hidden_layers       = 2,
    hidden_dim          = -1,
    activation          = tf.nn.relu,
    skip_connections    = False,
    output_activation   = tf.identity,
    output_layer        = True,
    dim_change          = 'jump',
    dropout_keep_prob   = None,
    batch_normalization = None,
    training            = True,
):
    assert dim_change in ['jump', 'step'], "'%s' not valid (should be in ['jump', 'step'])" % dim_change
    assert isinstance(hidden_layers, int) and hidden_layers >= 1
    assert isinstance(hidden_dim, int) and (hidden_dim == -1 or hidden_dim > 0)

    source_dim = get_dim(X)
    if hidden_dim == -1: hidden_dim = source_dim

    layer_dim = hidden_dim
    running = _feedforward_step(X, layer_dim, 'layer_0', skip_connections=skip_connections,
        activation=activation)

    for layer in range(1, hidden_layers):
        if dim_change == 'step': layer_dim -= math.floor((hidden_dim - out_dim)/(hidden_layers - 1))
        running = _feedforward_step(running, layer_dim, 'layer_%d' % layer,
            skip_connections=skip_connections, activation=activation)
    if dropout_keep_prob is not None: running = tf.nn.dropout(running, dropout_keep_prob)

    if output_layer: return _feedforward_step(running, out_dim, 'output_layer',
        skip_connections=skip_connections, activation=output_activation)
    else: return running

def step_variable(name='global_step'): return tf.Variable(0, name=name, trainable=False)
