import tensorflow as tf, numpy as np, math

DISTANCES = ['euc', 'man', 'cos']
DISTANCE_MAPPINGS = {'euc': 'euclidean', 'man': 1}

# Shape Constructions
def shape(X): return X.get_shape().as_list() if type(X) == tf.Tensor else X.shape
def get_dim(X): return shape(X)[1]
def num_samples(X): return shape(X)[0]

def make_compatible(X, Y):
  x_shape, y_shape = tf.shape(X), tf.shape(Y)
  return tf.cond(
      x_shape[0] > y_shape[0],
      lambda: (X, tf.pad(Y, [[0, x_shape[0] - y_shape[0]], [0, 0]], mode="SYMMETRIC")),
      lambda: tf.cond(
        y_shape[0] > x_shape[0],
        lambda: (tf.pad(X, [[0, y_shape[0] - x_shape[0]], [0, 0]], mode="SYMMETRIC"), Y),
        lambda: (X, Y),
      ),
  )

# Distances
def _cosine_distance(X, Y, pad=1e-7):
  return 1 - tf.reduce_mean((pad + tf.reduce_sum(X*Y, axis=1))/(pad + tf.norm(X, axis=1)*tf.norm(Y, axis=1)))

def dist(X, Y, ord='euc'):
  assert ord in DISTANCES

  if ord == 'cos': return _cosine_distance(X, Y)
  else: return tf.reduce_mean(tf.norm(X - Y, axis=1, ord=DISTANCE_MAPPINGS[ord]))

# Neural Network Constructions
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

def _feedforward(X, out_dim, scope, skip_connections=False, activation=tf.nn.relu):
    # For now only apply skip connections if out_dim == in_dim
    ff = activation(linear(X, out_dim, scope))
    return tf.cond(skip_connections and get_dim(X) == out_dim, lambda: X + ff lambda: X)

def feedforward_net(
    X, out_dim,
    hidden_layers     = 2,
    hidden_dim        = -1,
    activation        = tf.nn.relu,
    skip_connections  = False,
    output_activation = tf.identity,
    output_layer      = True,
):
  assert hidden_layers >= 1

  source_dim = get_dim(X)
  if hidden_dim == -1: hidden_dim = source_dim

  running = X
  for layer in range(hidden_layers):
    running = _feedforward(running, hidden_dim, 'layer_%d' % layer, skip_connections=skip_connections,
        activation=activation)

  return _feedforward(running, out_dim, 'output_layer', skip_connections=skip_connections,
        activation=output_activation) if output_layer else running
