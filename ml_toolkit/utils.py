import numpy as np

def flatten(arr):
  if type(arr) is np.ndarray: return np.reshape(arr, [len(arr), -1])
  elif type(arr) is list:
    r = []
    for l in arr: r += l
    return r
  raise NotImplementedError

def pad(arr, max_len, pad_value=0):
  assert len(arr) <= max_len, "Cannot negatively pad!"
  if type(arr) is list: return arr + [pad_value] * (max_len-len(arr))

  raise NotImplementedError
