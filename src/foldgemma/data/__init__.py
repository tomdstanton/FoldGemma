"""Data processing package for FoldGemma."""

import os

try:
    # Suppress TensorFlow C++ initialization warnings (like Cannot dlopen GPU libraries)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    # Hide GPUs from TensorFlow to prevent it from crashing when CUDA libraries are missing,
    # and to prevent it from allocating VRAM that PyTorch needs.
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    pass
