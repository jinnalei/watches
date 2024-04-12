"""
Module providing normalization functions for features and audio signals.
"""

import numpy as np


class Normalization:
    """
    Class containing static methods for normalization.
    """

    @staticmethod
    def zscore_normalize_feature(feature, feature_axis=None):
        """
        Perform Z-score normalization on features.

        Args:
            feature (numpy.ndarray): N-dimensional array with features.
            feature_axis (int): Axis along which to perform normalization. Defaults to the last axis.

        Returns:
            numpy.ndarray: The normalized feature.
        """
        # Set feature_axis to the last axis if it is not specified
        if feature_axis is None:
            feature_axis = feature.ndim - 1

        # Compute mean and std along the specified axis
        means = np.mean(feature, axis=feature_axis, keepdims=True)
        stds = np.std(feature, axis=feature_axis, keepdims=True)

        # To avoid division by zero, add epsilon to stds directly
        epsilon = 1e-10
        stds += epsilon

        # Perform z-score normalization
        normalized = (feature - means) / stds
        return normalized

    @staticmethod
    def scale_by_max_abs(feature, feature_axis=None):
        """
        Scale an N-dimensional array by dividing by the maximum absolute value along the specified axis.
        This normalizes the feature to a range of [-1, 1].

        Args:
            feature (numpy.ndarray): N-dimensional array where the scaling is to be applied.
            feature_axis (int): Axis along which to perform scaling. Defaults to the last axis.

        Returns:
            numpy.ndarray: The feature scaled to the range [-1, 1].
        """
        # Set feature_axis to the last axis if it is not specified
        if feature_axis is None:
            feature_axis = feature.ndim - 1

        # Ensure that the feature_axis is in the range [-ndim, ndim-1]
        if not (-feature.ndim <= feature_axis < feature.ndim):
            raise ValueError(
                "feature_axis must be within the range [-ndim, ndim-1]")

        # Find the maximum absolute value in the specified axis
        max_abs_values = np.max(
            np.abs(feature), axis=feature_axis, keepdims=True)

        # Avoid division by zero by setting zeros to one
        max_abs_values[max_abs_values == 0] = 1

        # Scale features by their maximum absolute value
        return feature / max_abs_values

    @staticmethod
    def rms_normalization(audio, desired_rms=0.1):
        """
        Normalizes audio using root mean square (RMS) normalization to a desired RMS level.

        Args:
            audio (numpy.ndarray): The input audio signal (assumed to be a 1-D numpy array).
            desired_rms (float): The desired RMS level to which the audio will be normalized. Default is 0.1.

        Returns:
            numpy.ndarray: The RMS-normalized audio.
        """
        # Calculate the current RMS of the audio
        current_rms = np.sqrt(np.mean(np.square(audio)))

        # Avoid division by zero
        if current_rms == 0:
            return audio  # Returning the original audio if RMS is 0 to avoid division by zero

        # Calculate the scaling factor and normalize the audio
        scaling_factor = desired_rms / current_rms
        normalized_audio = audio * scaling_factor

        return normalized_audio

    @staticmethod
    def scale_audio_using_interp(audio):
        """
        Scales an audio signal to fit exactly within the range [-1, 1] using interpolation.

        Args:
            audio (numpy.ndarray): The input audio signal (assumed to be a 1-D numpy array).

        Returns:
            numpy.ndarray: The audio scaled to the range [-1, 1].
        """
        return np.interp(audio, (audio.min(), audio.max()), (-1, 1))
