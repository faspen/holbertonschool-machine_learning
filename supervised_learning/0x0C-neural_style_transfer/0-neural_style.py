#!/usr/bin/env python3
"""Neural style transfer"""


import numpy as np
import tensorflow as tf


class NST():
    """Class the performs neural style transfer"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1']
    content_layer = 'block5_conv2'
