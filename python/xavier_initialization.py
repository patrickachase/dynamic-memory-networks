import numpy as np
import tensorflow as tf

def xavier_weight_init():
  """
  Returns function that creates random tensor. 

  The specified function will take in a shape (tuple or 1-d array) and must
  return a random tensor of the specified shape and must be drawn from the
  Xavier initialization distribution.

  """
  def _xavier_initializer(shape, **kwargs):
    """Defines an initializer for the Xavier distribution.

    This function will be used as a variable scope initializer.

    Args:
      shape: Tuple or 1-d array that species dimensions of requested tensor.
    Returns:
      out: tf.Tensor of specified shape sampled from Xavier distribution.
    """
    eta = tf.sqrt(6.0)/tf.sqrt(tf.to_float(tf.reduce_sum(shape)))
    out = tf.random_uniform(shape, minval = -eta, maxval = eta)

    return out
  return _xavier_initializer