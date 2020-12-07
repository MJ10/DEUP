from hashlib import sha256
import pickle
import os


def hash_list_str(str_, hash_size=35):
    """ Hash a list of strings.
    """
    # python hash() not deterministic across runs, so need hashlib
    hashed = sha256(str_.encode('utf-8'))
    hashed = hashed.hexdigest()[:hash_size]
    return hashed


def represent_dict_as_str(d):
    """ Compute a string representation of a dataclass instance.
    In particular, abbreviate some terms and call `dataclass.get_dict_dirname`
    to shorten the representation.
    """

    s = ''
    items = d.items()
    for k, v in items:
        v = str(v)
        if v is not None and v != []:
            k = k.replace('_', '')
            s += k + '_' + str(v) + '_'
    return s[:-1]


def hash_args(args, hash_size=35):
    return hash_list_str(represent_dict_as_str(vars(args)), hash_size)


def compute_exp_dir(args):
    exps_root = os.getenv('EP_EXPS_ROOT')
    args_hash = hash_args(args)
    exp_dir = os.path.join(exps_root, args_hash)
    if os.path.isdir(exp_dir):
        exp_dir = os.path.join(exps_root, 'RERUN_' + args_hash)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    return exp_dir


def log_args(args, exp_dir):
    filename = 'hyperparameters.pkl'
    path = os.path.join(exp_dir, filename)
    with open(path, 'wb+') as f:
        pickle.dump(vars(args), f)


def log_results(results, exp_dir):
    filename = 'results.pkl'
    path = os.path.join(exp_dir, filename)
    with open(path, 'wb') as f:
        pickle.dump(results, f)