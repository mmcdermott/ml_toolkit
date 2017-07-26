import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split

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
    table_cnt_multiplier = 3 if dev_size is not None and dev_size > 0 else 2

    results = list(train_test_split(*present_tables, test_size=test_size, random_state=random_state))
    if dev_size is not None and dev_size > 0:
        train_tables, test_tables = _split_pairs(results)
        dev_splits = train_test_split(*train_tables, test_size=dev_size, random_state=random_state)
        train_tables, dev_tables = _split_pairs(dev_splits)
        results = list(_interleave(train_tables, dev_tables, test_tables))

    for raw_idx in none_indices:
        idx = table_cnt_multiplier * raw_idx
        results = results[:idx] + [None]*table_cnt_multiplier + results[idx:]

    return results
