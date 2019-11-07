import argparse, json, pickle
from abc import ABC, abstractmethod
from typing import Sequence
from dataclasses import dataclass, asdict

from ..constants import *

def intlt(bounds):
    start, end = bounds if type(bounds) is tuple else (0, bounds)
    def fntr(x):
        x = int(x)
        if x < start or x >= end: raise ValueError("%d must be in [%d, %d)" % (x, start, end))
        return x
    return fntr

def within(s):
    def fntr(x):
        if x not in s: raise ValueError("%s must be in {%s}!" % (x, ', '.join(s)))
        return x
    return fntr

class BaseArgs(ABC):
    @classmethod
    def from_json_file(cls, filepath):
        with open(filepath, mode='r') as f: return cls(**json.loads(f.read()))
    @staticmethod
    def from_pickle_file(filepath):
        with open(filepath, mode='rb') as f: return pickle.load(f)

    def to_dict(self): return asdict(self)
    def to_json_file(self, filepath):
        with open(filepath, mode='w') as f: f.write(json.dumps(asdict(self)))
    def to_pickle_file(self, filepath):
        with open(filepath, mode='wb') as f: pickle.dump(self, f)

    @classmethod
    @abstractmethod
    def _build_argparse_spec(cls, parser):
        raise NotImplementedError("Must overwrite in base class!")

    @classmethod
    def from_commandline(cls):
        parser = argparse.ArgumentParser()

        # To load from a run_directory (not synced to overall structure above):
        parser.add_argument(
            "--do_load_from_dir", action='store_true',
            help="Should the system reload from the sentinel args.json file in the specified run directory "
                 "(--run_dir) and use those args rather than consider those set here? If so, no other args "
                 "need be set (they will all be ignored).",
            default=False
        )

        main_dir_arg, args_filename = cls._build_argparse_spec(parser)

        args = parser.parse_args()

        if args.do_load_from_dir:
            load_dir = vars(args)[main_dir_arg]
            assert os.path.exists(load_dir), "Dir (%s) must exist!" % load_dir
            args_path = os.path.join(load_dir, args_filename)
            assert os.path.exists(args_path), "Args file (%s) must exist!" % args_path

            return cls.from_json_file(args_path)

        args_dict = vars(args)
        if 'do_load_from_dir' in args_dict: args_dict.pop('do_load_from_dir')

        return cls(**args_dict)

# Sample usage
# @dataclass
# class EvalArgs(BaseArgs):
#     run_dir:                  str  = "" # required
#     notes:                    str   = "no_notes" # {no_notes, integrate_note_bert} TODO: add topics, doc2vec
#     rotation:                 int  = 0
#     do_save_all_reprs:        bool = True
#     do_eval_train:            bool = False
#     do_eval_tuning:           bool = True
#     do_eval_test:             bool = True
#     num_dataloader_workers:   int  = 8 # Num dataloader workers. Can increase.
# 
#     @classmethod
#     def _build_argparse_spec(cls, parser):
#         parser.add_argument("--run_dir", type=str, required=True, help="Dir for this generalizability exp.")
#         parser.add_argument("--rotation", type=intlt(10), default=0, help="Rotation")
#         parser.add_argument('--do_save_all_reprs', action='store_true', default=True, help='Save all reprs.')
#         parser.add_argument('--no_do_save_all_reprs', action='store_false', dest='do_save_all_reprs')
#         parser.add_argument('--do_eval_train', action='store_true', default=False, help='Eval Train')
#         parser.add_argument('--no_do_eval_train', action='store_false', dest='do_eval_train')
#         parser.add_argument('--do_eval_tuning', action='store_true', default=True, help='Eval Tuning')
#         parser.add_argument('--no_do_eval_tuning', action='store_false', dest='do_eval_tuning')
#         parser.add_argument('--do_eval_test', action='store_true', default=True, help='Eval Test')
#         parser.add_argument('--no_do_eval_test', action='store_false', dest='do_eval_test')
#         parser.add_argument('--notes', type=within({'no_notes', 'integrate_note_bert'}), default='no_notes')
#         parser.add_argument('--num_dataloader_workers', type=int, default=4, help='# dataloader workers.')
# 
#         return 'run_dir', EVAL_ARGS_FILENAME
