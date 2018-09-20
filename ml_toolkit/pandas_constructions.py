import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
# TODO(mmd): Fix improper _ usage on private methods.

def or_idxs(idxs):
    assert len(idxs) > 0, "Must provide a non-empty list of indices."
    running = idxs[0]
    for idx in idxs[1:]: running = (running | idx)
    return running

# TODO(mmd): Document pre-conditions.
def join(dfs): return dfs[0].join(dfs[1:], how='inner')

def stack_columns(df, col_name):
    assert len(df.columns.names) == 1, "stack_columns does not yet support stackable multi-indexing."
    cols = df.columns
    name = df.columns.names[0]
    dfs = []
    for col in cols:
        tmp_df = df.filter(items=[col])
        tmp_df[name] = col
        tmp_df.set_index(name, append=True, inplace=True)
        tmp_df.rename(columns={col: col_name}, inplace=True)
        dfs += [tmp_df]
    return pd.concat(dfs)

# TODO(mmd): Better handling of case where levels_to_keep \cap df_idx.names == {}.
def __keep_levels(df_idx, levels_to_keep=[]):
    all_levels = df_idx.names
    for level in all_levels:
        if level not in levels_to_keep:
            if len(df_idx.names) > 1: df_idx = df_idx.droplevel(level)
            else: df_idx = pd.Index(range(len(df_idx)))
    return df_idx
def __drop_levels(df_idx, levels_to_drop=[]):
    for level in levels_to_drop: df_idx = df_idx.droplevel(level)
    return df_idx
def keep_indices(df, index_levels=None, column_levels=None, inplace=False):
    assert index_levels is None or type(index_levels) in (list, tuple), "index_levels must be a list/tuple"
    assert column_levels is None or type(column_levels) in (list, tuple), "column_levels must be a list/tuple"

    df_cp = df.copy() if not inplace else df
    if index_levels != None: df_cp.index = __keep_levels(df_cp.index, levels_to_keep=index_levels)
    if column_levels != None: df_cp.columns = __keep_levels(df_cp.columns, levels_to_keep=column_levels)
    return df_cp
def drop_indices(df, index_levels=[], column_levels=[], inplace=False):
    df_cp = df.copy() if not inplace else df
    df_cp.index = __drop_levels(df_cp.index, index_levels)
    df_cp.columns = __drop_levels(df_cp.columns, column_levels)
    return df_cp

def get_index_levels(df, levels, make_objects_categories=True):
    df_2 = pd.DataFrame(index=df.index)
    for level in levels: df_2[level] = df_2.index.get_level_values(level)
    if make_objects_categories:
        for column in df_2.columns:
            if df_2[column].dtype == object: df_2[column] = df_2[column].astype('category')

    return df_2

def _split_pairs(l): return [l[i] for i in range(0, len(l), 2)], [l[i] for i in range(1, len(l), 2)]
def _interleave(*ls):
    result = []
    for elems in zip(*ls):
        for el in elems:
            result += [el]
    return result

# TODO(mmd): Should take a list of sizes, not just these two.
def split(tables, test_size=0.2, dev_size=.125, random_state=None):
    """
    Splits the data into train/test (and possibly also dev)
    :param test_size: fraction of the data to put in the test set
    :param dev_size: fraction of the remaining data (not including test) to put in the dev set
                     if None, only train/test splits are done
    :param random_state: random state passed to the train_test_split function (from sklearn)
    :returns: train_source, (dev_source,) test_source, train_target, (dev_target,) test_target
    """

    present_tables, none_indices = [], []
    for i, table in enumerate(tables):
        if table is not None: present_tables.append(table)
        else: none_indices.append(i)
    table_cnt_multiplier = 3 if dev_size is not None else 2

    if test_size is not None and test_size > 0:
        results = list(train_test_split(*present_tables, test_size=test_size, random_state=random_state))
    elif test_size is not None and test_size == 0:
        results = list(_interleave(present_tables, map(lambda df: df.iloc[0:0], present_tables)))

    if dev_size is not None:
        train_tables, test_tables = _split_pairs(results)
        dev_splits = train_test_split(*train_tables, test_size=dev_size, random_state=random_state)
        train_tables, dev_tables = _split_pairs(dev_splits)
        results = list(_interleave(train_tables, dev_tables, test_tables))

    for raw_idx in none_indices:
        idx = table_cnt_multiplier * raw_idx
        results = results[:idx] + [None]*table_cnt_multiplier + results[idx:]

    return results

#TODO(mmd): Add n= option.
def index_k_fold(df, k=10, stratify_by=None, random_state=None, inplace=True, fold_name = 'Fold'):
    kf_args = dict(n_splits=k, random_state=random_state, shuffle=True)
    if stratify_by is None:
        kf = KFold(**kf_args)
        splits = kf.split(df.values)
    else:
        kf = StratifiedKFold(**kf_args)
        if type(stratify_by) is list:
            labels, cnt, indices_map, indices_list = [], 0, {}, []
            levels = np.concatenate(
                [np.expand_dims(df.index.get_level_values(l).values, 1) for l in stratify_by],
                axis=1
            )
            for l in levels:
                key = tuple(l)
                if key not in indices_map:
                    cnt += 1
                    indices_map[key] = cnt
                indices_list.append(indices_map[key])
            splits = kf.split(df.values, indices_list)
        else: splits = kf.split(df.values, df.index.get_level_values(stratify_by))

    fold_indices = np.ones(len(df), dtype=np.uint8)
    for fold, (_, fold_idx) in enumerate(splits):
        fold_indices[fold_idx] = fold

    if not inplace: df = df.copy()
    df[fold_name] = fold_indices
    df.set_index(fold_name, append=True, inplace=True)
    return df
