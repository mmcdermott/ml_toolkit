import tensorflow as tf, numpy as np, math

DISTANCES = ['euc', 'man', 'cos']
DISTANCE_MAPPINGS = {'euc': 'euclidean', 'man': 1}

# Shape Constructions
def shape(X): return X.get_shape().as_list() if type(X) == tf.Tensor else X.shape
def get_dim(X): return shape(X)[1]
def num_samples(X): return shape(X)[0]

def make_compatible(X, Y):
    num_x_samples, num_y_samples = num_samples(X), num_y_samples(Y)
    pad = lambda T, diff: return tf.pad(T, [[0, diff], [0, 0]], mode='Symmetric')
    return tf.case({
            tf.greater(num_x_samples, num_y_samples): lambda: (X, pad(Y, num_x_samples - num_y_samples)),
            tf.greater(num_y_samples, num_x_samples): lambda: (pad(X, num_y_samples - num_x_samples), Y),
        },
        default=lambda: (X, Y),
    )

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
      weights = tf.get_variable('weights', [source_dim, out_dim], initializer=weights_initializer)
      bias = tf.get_variable('bias', [out_dim], initializer=bias_initializer)

    return tf.matmul(X, weights) + bias

def _feedforward_step(X, out_dim, scope, skip_connections=False, activation=tf.nn.relu):
      # For now only apply skip connections if out_dim == in_dim
      ff = activation(linear(X, out_dim, scope))
      return tf.cond(skip_connections and get_dim(X) == out_dim, lambda: X + ff lambda: X)

# TODO(mmd): Use enums for dim_change options.
def feedforward(
    X, out_dim,
    hidden_layers     = 2,
    hidden_dim        = -1,
    activation        = tf.nn.relu,
    skip_connections  = False,
    output_activation = tf.identity,
    output_layer      = True,
    dim_change        = 'jump',
):
    assert dim_change in ['jump', 'step']
    assert isinstance(hidden_layers, int) and hidden_layers >= 1
    assert isinstance(hidden_dim, int) and (hidden_dim == -1 or hidden_dim > 0)

    source_dim = get_dim(X)
    if hidden_dim == -1: hidden_dim = source_dim
    layer_dim = hidden_dim

    running = _feedforward(X, layer_dim, 'layer_0', skip_connections=skip_connections, activation=activation)
    for layer in range(1, hidden_layers):
        if dim_change == 'step': layer_dim -= math.floor(math.abs(hidden_dim - out_dim)/(hidden_layers - 1))
        running = _feedforward(running, hidden_dim, 'layer_%d' % layer, skip_connections=skip_connections,
            activation=activation)

    if output_layer: return _feedforward(running, out_dim, 'output_layer', skip_connections=skip_connections,
        activation=output_activation)
    else: return running
