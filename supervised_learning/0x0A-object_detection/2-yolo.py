#!/usr/bin/env python3
"""0 yolo, set up init"""


import tensorflow.keras as K
import numpy as np


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

    def sigmoid(self, x):
        """Sigmoid helper function"""
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """Process the Darknet model's output"""
        boxes = []
        box_confidence = []
        box_class_probs = []

        for idx, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            box = np.zeros(output[:, :, :, :4].shape)

            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            aw_total = self.anchors[:, :, 0]
            aw = np.tile(aw_total[idx], grid_width)
            aw = aw.reshape(grid_width, 1, len(aw_total[idx]))
            ah_total = self.anchros[:, :, 1]
            ah = np.tile(ah_total[idx], grid_height)
            ah = ah.reshape(grid_height, 1, len(ah_total[idx]))

            xc = np.tile(np.arange(grid_width), grid_height)
            xc = xc.reshape(grid_width, grid_width, 1)
            yc = np.tile(np.arange(grid_width), grid_height)
            yc = yc.reshape(grid_height, grid_height).T
            yc = yc.reshape(grid_height, grid_height, 1)

            xb = self.sigmoid(t_x) + xc
            yb = self.sigmoid(t_y) + yc
            wb = np.exp(t_w) * aw
            hb = np.exp(t_h) * ah

            xb /= grid_width
            yb /= grid_height
            wb /= self.model.input.shape[1]
            hb /= self.model.input.shape[2]

            x1 = (xb - (wb / 2)) * image_size[1]
            y1 = (yb - (hb / 2)) * image_size[0]
            x2 = (xb + (wb / 2)) * image_size[1]
            y2 = (yb + (hb / 2)) * image_size[0]

            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)

            tmp = output[:, :, :, 4]
            sig = self.sigmoid(tmp)
            box_confidence.append(sig)

            tmp = output[:, :, :, 5:]
            sig = self.sigmoid(tmp)
            box_class_probs.append(sig)

        return (boxes, box_confidence, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Returns filtered boxes"""

        return (filtered_boxes, box_classes, box_scores)
