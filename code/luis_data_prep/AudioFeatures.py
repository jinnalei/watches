"""
kyyras_audio_processing: A module for audio feature extraction and processing.

This module provides classes and functions for extracting audio features, segmenting audio data,
and performing various audio processing tasks.

Author: Luis Ramirez
Date: January 23, 2024
"""


import os
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
from scipy.stats import kurtosis
import soundfile as sf



class AudioFeatureExtractor:
    """
    A class for audio feature extraction and processing.

    This class provides methods for extracting and processing audio features from audio files
    or audio segments. It includes methods for feature extraction, segmentation, noise reduction,
    and more.

    Attributes:
    None

    Methods:
    - flatten_features: Flattens and concatenates audio features into a single array.
    - pad_features: Pads audio features to have the same first dimension.
    - preprocess: Preprocesses audio data, including noise reduction and normalization.
    - get_audio_file_features: Extracts features from an audio file and returns them in a list.
    - get_audio_features: Extracts features from an audio segment and returns them in a list.
    - segment_audio_file_to_memory: Segments an audio file into specified-length samples in memory.
    - segment_audio_to_memory: Segments an audio segment into specified-length samples in memory.
    - print_feature_shapes: Prints the shapes of the features in the given dictionary.
    - playSegment: Plays an audio segment using sounddevice.
    - is_noise: Determines if the given audio data is mostly noise based on kurtosis measure.

    """

    @staticmethod
    def segment_audio_to_memory(audio_segment, segment_length_sec, overlap_sec, target_sample_rate):
        """
        Segments the given audio into specified-length samples with specified overlap in memory.
        This function makes a copy of each segmented part of the input audio to ensure the segments
        are independent of the original data. It assumes the audio is already in the correct format
        and does not perform preprocessing. The function raises an error if the audio segment is not
        mono or not in float32 format.

        Args:
        audio_segment (numpy.ndarray): Input audio segment.
        segment_length_sec (float): Length of each audio segment in seconds.
        overlap_sec (float): Overlap time in seconds between consecutive audio segments.
        target_sample_rate (int): The target sample rate in Hz.

        Returns:
        list: List of audio segments, each being an independent copy of a part of the
        original audio data.

        Raises:
        ValueError: If the audio segment is not mono or not a float32 array.
        """

        # Check if audio segment is mono
        if len(audio_segment.shape) > 1:
            raise ValueError("Audio segment must be mono.")

        # Check if audio segment is float32
        if audio_segment.dtype != np.float32:
            raise ValueError("Audio segment must be of type float32.")

        # Calculate segment lengths in samples
        sample_duration = int(segment_length_sec * target_sample_rate)
        overlap_samples = int(overlap_sec * target_sample_rate)

        segments = []
        start = 0
        while start + sample_duration <= len(audio_segment):
            end = start + sample_duration
            # Create a copy of each segment to ensure independence from original data
            segments.append(audio_segment[start:end].copy())
            start += sample_duration - overlap_samples

        return segments

