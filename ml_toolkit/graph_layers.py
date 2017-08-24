# Adapted from https://github.com/mdeff/cnn_graph
from .tensorflow_constructions import leaky_relu

import scipy.sparse, networkx, tensorflow as tf, numpy as np, pandas as pd

# TODO(mmd): Use generators properly.
def split_into_components(X_df, G):
  """set(X_df.columns) must == set(G.nodes)"""
  components = list(networkx.components.connected_components(G))

  X_splits = [X_df.filter(items=component) for component in components]
  subgraphs = [G.subgraph(component) for component in components]
  return X_splits, subgraphs

def _pool_step(
    X,
    pool_size, #TODO(mmd): Better name
    pooler = tf.nn.max_pool,
):
    """Pooling of size p. Should be a power of 2 greater than 1."""
    # TODO(mmd): Why all the expansion squeezing necessary?
    x = tf.expand_dims(x, 3)  # num_samples x num_features x num_filters_in x 1
    x = pooler(x, ksize=[1,pool_size,1,1], strides=[1,pool_size,1,1], padding='SAME')
    #tf.maximum
    return tf.squeeze(x, [3])  # num_samples x num_features / p x num_filters

# TODO(mmd): Unify shape API for graph_conf layers.
# TODO(mmd): Better name.
def _full_fourier_graph_conv_step(
    X,
    G,
    scope,
    nodelist,
    receptive_field_size = 10,
    num_filters_out      = 32,
    activation           = leaky_relu,
    batch_normalization  = None,
    training             = True,
    weights_init         = tf.truncated_normal_initializer(mean=0.0, stddev=0.05),
    bias_init            = tf.constant_initializer(0.0),
):
    """Graph CNN with full weight matrices, i.e. patch has the same size as input."""
    num_samples, num_features, num_filters_in = X.shape.as_list()

    L = networkx.normalized_laplacian_matrix(G, nodelist=nodelist)
    U = tf.constant(np.linalg.eigh(L.toarray())[1], dtype=tf.float32)

    # TODO(mmd): Get the below to work.
    #_, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

    x = tf.transpose(X, [0, 2, 1]) # num_samples x num_filters_in x num_features
    x = tf.reshape(x, [num_samples * num_filters_in, num_features])
    xf = tf.expand_dims(tf.matmul(x, U), 1)
    xf = tf.reshape(xf, [num_samples, num_filters_in, num_features])
    xf = tf.transpose(xf, [2, 1, 0]) # num_features x num_filters_in x num_samples

    with tf.variable_scope(scope):
        # TODO(mmd): Shapes probably wrong.
        W = tf.get_variable(
            'graph_convolution',
            [num_features * num_filters_in, num_filters_out, 1],
            tf.float32,
            initializer = weights_init,
        )
        b = tf.get_variable(
            'graph_bias',
            [1, num_filters_out, 1],
            tf.float32,
            initializer = bias_init,
        )

        yf = tf.matmul(W, xf)
        yf = tf.reshape(tf.transpose(yf, [2, 1, 0]), [num_samples * num_filters_out, num_features])
        y = tf.matmul(yf, tf.transpose(U))

        return activation(tf.reshape(y, [num_samples, num_filters_out, num_features]) + b)

# Chebyshev
def _chebyshev_graph_conv_step(
    X,
    G,
    scope,
    nodelist,
    receptive_field_size = 10,
    num_filters_out      = 32,
    activation           = leaky_relu,
    batch_normalization  = None,
    training             = True,
    weights_init         = tf.truncated_normal_initializer(mean=0.0, stddev=0.05),
    bias_init            = tf.constant_initializer(0.0),
):
    """Graph CNN with full weights, i.e. patch has the same size as input."""
    num_samples, num_features, num_filters_in = X.shape.as_list()

    L = networkx.normalized_laplacian_matrix(G, nodelist=nodelist).astype(np.float32)
    L = (L - scipy.sparse.identity(num_features, dtype=L.dtype, format='csr')).tocoo()
    indices = np.column_stack((L.row, L.col))

    L = tf.sparse_reorder(tf.SparseTensor(indices=indices, values=L.data, dense_shape=L.shape))

    # Transform to Chebyshev basis
    # TODO(mmd): Are the permutations/reshapes really necessary or would this just work with smart
    # broadcasting?
    x0 = tf.transpose(X, perm=[1, 2, 0])  # num_features x num_filters_in x num_samples
    x0 = tf.reshape(x0, [num_features, num_filters_in*num_samples])

    chebyshev_terms = [x0]
    if receptive_field_size > 1:
        chebyshev_terms.append(tf.sparse_tensor_dense_matmul(L, chebyshev_terms[-1]))
        for _ in range(2, receptive_field_size):
            chebyshev_terms += [2*tf.sparse_tensor_dense_matmul(L, chebyshev_terms[-1]) - chebyshev_terms[-2]]
    x = tf.stack(chebyshev_terms) # receptive_field_size x num_features x num_filters_in*num_samples

    x = tf.reshape(x, [receptive_field_size, num_features, num_filters_in, num_samples])
    x = tf.transpose(x, perm=[3,1,2,0])  # num_samples x num_features x num_filters_in x receptive_field_size
    # TODO(mmd): Do I need to reshape like this or can this be handled fine with tensor multiplications?
    x = tf.reshape(x, [num_samples * num_features, num_filters_in * receptive_field_size])

    with tf.variable_scope(scope):
        # Filter: num_filters_in -> num_filters_out filters of order K, i.e. one filterbank per feature pair.
        W = tf.get_variable(
            'graph_convolution',
            [num_filters_in * receptive_field_size, num_filters_out],
            tf.float32,
            initializer = weights_init
        )
        b = tf.get_variable(
            'bias',
            [1, 1, num_filters_out],
            tf.float32,
            initializer = bias_init
        )

        x = activation(tf.matmul(x, W) + b)
    return tf.reshape(x, [num_samples, num_features, num_filters_out])

# TODO(mmd): Make accept num_filters_out (right now is effectively (though not in impl.) hard-coded @ 1)
def _graph_localized_ff_step(
    X,
    G,
    scope,
    nodelist,
    activation           = leaky_relu,
    batch_normalization  = None,
    training             = True,
    weights_init         = tf.truncated_normal_initializer(mean=0.0, stddev=0.05),
    bias_init            = tf.constant_initializer(0.0),
):
    num_samples, num_features = X.shape.as_list()

    A = networkx.adjacency_matrix(G, nodelist=nodelist).astype(np.float32)
    A = (A + scipy.sparse.identity(num_features, dtype=A.dtype, format='csr')).tocoo()

    indices = np.column_stack((A.row, A.col))
    num_edges = len(indices)

    with tf.variable_scope(scope):
        W = tf.get_variable(
            'graph_localized_ff_weights',
            [num_edges],
            tf.float32,
            initializer = weights_init,
        )
        W_tensor = tf.sparse_reorder(
            tf.SparseTensor(indices=indices, values=W, dense_shape=[num_features, num_features])
        )
        b = tf.get_variable(
            'bias',
            [num_features],
            tf.float32,
            initializer = bias_init,
        )
        return activation(tf.transpose(tf.sparse_tensor_dense_matmul(W_tensor, tf.transpose(X))) + b)
