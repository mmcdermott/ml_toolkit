import tensorflow as tf

DISTANCES = ['euc', 'man', 'cos']
DISTANCE_MAPPINGS = {'euc': 'euclidean', 'man': 1}
def _cosine_distance(X, Y, pad=1e-7):
  return 1 - tf.reduce_mean((pad + tf.reduce_sum(X*Y, axis=1))/(pad + tf.norm(X, axis=1)*tf.norm(Y, axis=1)))
def _get_dim(X): return X.get_shape().as_list()[1]

def dist(X, Y, ord='euc'):
  assert ord in DISTANCES

  if ord == 'cos': return _cosine_distance(X, Y)
  else: return tf.reduce_mean(tf.norm(X - Y, axis=1, ord=DISTANCE_MAPPINGS[ord]))

def linear(X, out_dim, scope):
  source_dim = get_dim(X)

  with tf.variable_scope(scope):
    weights = tf.get_variable('weights', [source_dim, out_dim],
        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.2))
    bias = tf.get_variable('bias', [out_dim], initializer=tf.constant_initializer(0.0))

  return tf.matmul(X, weights) + bias

def _feedforward(X, out_dim, scope): return tf.nn.relu(linear(X, out_dim, scope))

def feedforward_net(X, out_dim, hidden_layers=2, hidden_dim=-1, skip_connections=False):
  assert hidden_layers > 1

  source_dim = get_dim(X)
  if hidden_dim == -1: hidden_dim = source_dim

  running_state = feedforward(X, hidden_dim, 'input_layer')
  for layer in range(1, hidden_layers):
    if skip_connections: running_state += feedforward(running_state, hidden_dim, 'layer_%d' % layer)
    else: running_state = feedforward(running_state, hidden_dim, 'layer_%d' % layer)

  return linear(running_state, out_dim, 'output_layer')
