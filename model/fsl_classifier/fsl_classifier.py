#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class FSLClassifier(object):
    def __init__(
        self,
        model_path='model/fsl_classifier/fsl_classifier-right.tflite',
        num_threads=1,
    ):
        # Load TFLite interpreter
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=num_threads,
        )
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        """
        landmark_list: 42-element list (x1,y1,...,x21,y21) preprocessed
        returns: class_index (int)
        """
        # Prepare input tensor
        idx = self.input_details[0]['index']
        inp = np.array([landmark_list], dtype=np.float32)
        self.interpreter.set_tensor(idx, inp)

        # Run inference
        self.interpreter.invoke()

        # Read output tensor
        out_idx = self.output_details[0]['index']
        result = self.interpreter.get_tensor(out_idx)
        return int(np.argmax(result))
