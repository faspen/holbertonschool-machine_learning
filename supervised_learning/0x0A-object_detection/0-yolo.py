#!/usr/bin/env python3
"""0 yolo, set up init"""


import tensorflow.keras as K


class Yolo():
    """You only live once..."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Yolo initializer for yolo class"""
        self.model = K.models.load_model(filepath=model_path)
        with open(classes_path, 'r') as f:
            self.classes_names = [lines.strip() for lines in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
