# vim: set fileencoding=utf-8

from functools import reduce
import tensorflow as tf, numpy as np, pandas as pd
import ml_toolkit.tensorflow_constructions as tfc, ml_toolkit.pandas_constructions as pdc
import time, math, os, pickle

from constants import *
from utils import *
from benchmarks_printer import BenchmarkPrinter, ITER_HEADER, TIME_HEADER, STEP_HEADER

class TensorFlowModel(object):
    """
    Epochs, loss multipliers, patiences are lists to represent different training stages (so we can train one
    critic heavy stage with an unsupervised loss to optimality, and then a translator heavy stage with a
    predictive loss more fully as a second stage of trianing)
    """
    def __init__(
        self,
        config                       = tf.ConfigProto(),
        save_dir                     = '',
        train_suffix                 = '/train',
        dev_suffix                   = '/dev',
        model_ckpt_name              = '',
        model_params_name            = 'model_params',
        flush_secs                   = 15,
        random_state                 = None,
    ):
        assert model_ckpt_name != '', "Must provide a valid model checkpoint name file."
        assert save_dir != '' and os.path.isdir(save_dir), "Must provide a valid 'save_dir'."

        self.config             = config
        self.save_dir           = save_dir
        self.train_suffix       = train_suffix
        self.dev_suffix         = dev_suffix
        self.model_ckpt_name    = model_ckpt_name
        self.model_params_name  = model_params_name
        self.flush_secs         = flush_secs
        self.random_state       = random_state
        self.graph_is_built     = False
        self.session_is_started = False
        self.ready              = False
        self.tensor_names       = {}

        if self.random_state is not None: random.seed(self.random_state)

    def setup(self):
        model_params_file = os.path.join(self.save_dir, self.model_params_name) + '.pkl'
        if os.path.isfile(model_params_file):
            with open(model_params_file, 'r') as f: self.__dict__ = pickle.load(f).copy()
            self.save_dir = save_dir

        with open(model_params_file, 'w') as f: pickle.dump(self.__dict__, f)

        self.build_graph()
        assert self.graph_is_built, "Graph should be built!"

        self.start_session()
        assert self.session_is_started, "Session should be started!"

        self.ready = True

    def add_tensor(self, attr_name, tensor):
        """
        Sets self.attr_name = tensor.
        Also adds {attr_name: tensor.name} to the mapping of field names to tensor names to be restored in a loaded model.
        :param str attr_name: name to use for the attribute of self
        :param tensor: a tensorflow Tensor; tensor's name cannot already be used in this model
        :returns: None
        """
        assert tensor.name not in self.tensor_names.values(), "Trying to overwrite existing tensor!"
        self.tensor_names.update({attr_name: tensor.name})
        self.__setattr__(attr_name, tensor)

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.init_op = tf.global_variables_initializer()

    def start_session(self):
        """
        Starts a session with self.graph.
        If self.save_dir contains a previously trained model, then the graph from that run is loaded for
        further training/inference. If self.save_dir != '' then a Saver and summary writers are also created.
        :returns: None
        """
        meta_graph_file = os.path.join(self.save_dir, self.model_ckpt_name) + '.meta'
        if os.path.isfile(meta_graph_file): # load saved model
            self.sess = tf.Session(config=self.config)
            if self.print_anything: print("Loading graph from:", meta_graph_file)
            self.saver = tf.train.import_meta_graph(meta_graph_file)
            self.saver.restore(self.sess, os.path.join(self.save_dir, self.model_ckpt_name))
            self.graph = tf.get_default_graph()

            # update self's fields
            for attr_name, tensor_name in self.tensor_names.items():
                try: self.__setattr__(attr_name, self.graph.get_tensor_by_name(tensor_name))
                except KeyError:
                    continue

            # update indirect references
            for training_stage in xrange(self.training_stages):
                self.loss_ops[training_stage] = {}
                for key, name in self.loss_op_names[training_stage].items():
                    try: self.loss_ops[training_stage][key] = self.graph.get_tensor_by_name(name)
                    except KeyError:
                        continue
                self.train_ops[training_stage] = {}
                for key, name in self.train_op_names[training_stage].items():
                    try: self.train_ops[training_stage][key] = self.graph.get_operation_by_name(name)
                    except KeyError:
                        continue
        else:
                self.sess = tf.Session(config=self.config, graph=self.graph)
                self.sess.run(self.init_op)

        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.train_writer = tf.summary.FileWriter(self.save_dir + self.train_suffix, self.sess.graph,
                                                      flush_secs=self.flush_secs)
            self.dev_writer = tf.summary.FileWriter(self.save_dir + self.dev_suffix, self.sess.graph,
                                                        flush_secs=self.flush_secs)
        self.session_is_started = True

    def save(self):
        if self.save_dir == '': return
        assert self.session_is_started, "Session must be started to save."

        self.saver.save(self.sess, os.path.join(self.save_dir, self.model_ckpt_name))
