#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class FSLWordsClassifier(object):
    def __init__(
        self,
        model_path='model/fsl_words_classifier/fsl_words_classifier.tflite',
        score_th=0.5,
        invalid_value=0,
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value
        
        # Print model input shape for debugging
        print(f"FSL Words Classifier input shape: {self.input_details[0]['shape']}")
        print(f"Expected input dtype: {self.input_details[0]['dtype']}")

    def __call__(
        self,
        point_history_all,
    ):
        """
        Classify FSL words based on all landmarks history
        
        Args:
            point_history_all: Preprocessed point history for all landmarks
                              Should be a flattened list of normalized coordinates
                              Format: [frame0_point0_x, frame0_point0_y, ..., frameN_pointM_x, frameN_pointM_y]
                              Expected length: HISTORY_LENGTH * ALL_LANDMARKS_COUNT * 2
                              (60 frames * 48 landmarks * 2 coordinates = 5760 values)
        
        Returns:
            int: Predicted class ID, or invalid_value if confidence is too low
        """
        if point_history_all is None:
            return self.invalid_value
            
        # Ensure input is the right type and shape
        input_data = np.array(point_history_all, dtype=np.float32)
        
        # Check if input size matches expected model input
        expected_size = self.input_details[0]['shape'][1]  # Assuming shape is [1, input_size]
        
        if len(input_data) != expected_size:
            print(f"Warning: Input size mismatch. Got {len(input_data)}, expected {expected_size}")
            # Pad with zeros if too short, truncate if too long
            if len(input_data) < expected_size:
                padding = np.zeros(expected_size - len(input_data), dtype=np.float32)
                input_data = np.concatenate([input_data, padding])
            else:
                input_data = input_data[:expected_size]
        
        # Reshape to match model input (add batch dimension)
        input_data = input_data.reshape(1, -1)
        
        try:
            # Set input tensor
            input_details_tensor_index = self.input_details[0]['index']
            self.interpreter.set_tensor(input_details_tensor_index, input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_details_tensor_index = self.output_details[0]['index']
            result = self.interpreter.get_tensor(output_details_tensor_index)
            
            # Get predicted class
            result_squeeze = np.squeeze(result)
            result_index = np.argmax(result_squeeze)
            confidence = result_squeeze[result_index]
            
            # Check confidence threshold
            if confidence < self.score_th:
                return self.invalid_value
                
            return result_index
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return self.invalid_value
    
    def get_confidence(self, point_history_all):
        """
        Get confidence scores for all classes
        
        Args:
            point_history_all: Preprocessed point history for all landmarks
        
        Returns:
            numpy.ndarray: Array of confidence scores for each class
        """
        if point_history_all is None:
            return np.array([])
            
        # Ensure input is the right type and shape
        input_data = np.array(point_history_all, dtype=np.float32)
        
        # Check if input size matches expected model input
        expected_size = self.input_details[0]['shape'][1]
        
        if len(input_data) != expected_size:
            # Pad with zeros if too short, truncate if too long
            if len(input_data) < expected_size:
                padding = np.zeros(expected_size - len(input_data), dtype=np.float32)
                input_data = np.concatenate([input_data, padding])
            else:
                input_data = input_data[:expected_size]
        
        # Reshape to match model input
        input_data = input_data.reshape(1, -1)
        
        try:
            # Set input tensor
            input_details_tensor_index = self.input_details[0]['index']
            self.interpreter.set_tensor(input_details_tensor_index, input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_details_tensor_index = self.output_details[0]['index']
            result = self.interpreter.get_tensor(output_details_tensor_index)
            
            return np.squeeze(result)
            
        except Exception as e:
            print(f"Error during confidence calculation: {e}")
            return np.array([])
    
    def predict_with_confidence(self, point_history_all):
        """
        Get both prediction and confidence scores
        
        Args:
            point_history_all: Preprocessed point history for all landmarks
        
        Returns:
            tuple: (predicted_class_id, confidence_score, all_confidences)
        """
        if point_history_all is None:
            return self.invalid_value, 0.0, np.array([])
            
        # Get all confidence scores
        confidences = self.get_confidence(point_history_all)
        
        if len(confidences) == 0:
            return self.invalid_value, 0.0, np.array([])
        
        # Get predicted class and its confidence
        predicted_class = np.argmax(confidences)
        confidence_score = confidences[predicted_class]
        
        # Check confidence threshold
        if confidence_score < self.score_th:
            return self.invalid_value, confidence_score, confidences
            
        return predicted_class, confidence_score, confidences