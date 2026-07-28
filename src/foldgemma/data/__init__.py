"""Data processing package for FoldGemma."""

try:
    import tensorflow as tf
    # Hide GPUs from TensorFlow to prevent it from crashing when CUDA libraries are missing,
    # and to prevent it from allocating VRAM that PyTorch needs.
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    pass
