import os
import time
import argparse
import socket
import random
from collections import defaultdict
os.environ["JAX_PLATFORMS"] = "cpu"
import pickle

import tensorflow as tf
from google.cloud import storage
import seqio
import functools
from t5.data import preprocessors as t5_preprocessors
import jax
import numpy as np
from praxis import py_utils
import jax.numpy as jnp

# lsp
def _compute_slide_atten_mask(w, window_size, length: int, dtype: jnp.dtype = jnp.bfloat16):
  """
  w: query chunk size
  window_size: window size
  length: query length that before split
  dtype: query dtype
  """
  # w = 256
  # length = 2048
  # window_size = 1600
  if w is None:
    w = length
  if window_size is None:
    offset = length - w
  else:
    offset = min(window_size, length - w)
  x = jnp.ones([w, w + offset])
  m1 = jnp.triu(x, k=offset + 1)
  if window_size is not None:
    if window_size < length - w:
        m2 = jnp.tril(x, k=0)
    else:
        m2 = jnp.tril(x, k=length - window_size - w)
    m = m1 + m2
  else:
    m = m1
  large_negative_number = py_utils.get_large_negative_number(dtype)
  m = m.astype(dtype)
  # m = m * large_negative_number or as follow:
  m = jnp.where((m > 0.5), large_negative_number, m)
  # bnts
  return m[jnp.newaxis, jnp.newaxis, ...]

t = 2048
w = 512
window_size = 4096
query = jax.random.uniform(jax.random.PRNGKey(0), [4, 2048, 4096], dtype=jnp.bfloat16)
key = jax.random.uniform(jax.random.PRNGKey(0), [4, 2048, 4096], dtype=jnp.bfloat16)
value = jax.random.uniform(jax.random.PRNGKey(0), [4, 2048, 4096], dtype=jnp.bfloat16)

atten_mask = _compute_slide_atten_mask(w, window_size, t, query.dtype)

_atten_masks = []
for i in range(t // w):
    start, stop = i * w, (i + 1) * w
    kv_start = max(0, stop - w - window_size) if window_size is not None else 0
    _query = query[:, start : stop]
    _key, _value = key[:, kv_start : stop], value[:, kv_start : stop]
    # lsp
    _atten_mask = atten_mask[..., -_key.shape[1]:]
    _atten_masks.append(_atten_mask)