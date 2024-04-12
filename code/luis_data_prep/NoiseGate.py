"""
This module provides a NoiseGate class for processing audio signals.

Author: Luis Ramirez
"""

import scipy.io.wavfile as wav
import numpy as np
from scipy.signal import butter, filtfilt
from functools import lru_cache
from scipy.io import wavfile
from scipy.signal import resample
import librosa


class NoiseGate:
    """
    Class implementing a noise gate for audio signals.
    """

    @staticmethod
    def db_to_amplitude(db_level):
        """
        Convert decibel level to amplitude.

        Args:
            db_level (float): Decibel level.

        Returns:
            float: Amplitude.
        """
        return 10 ** (db_level / 20)

    @staticmethod
    def short_term_energy(signal, frame_size):
        """
        Compute the short-term energy of an audio signal.

        Args:
            signal (ndarray): Input audio signal.
            frame_size (int): Size of the frame.

        Returns:
            ndarray: Short-term energy of the signal.
        """
        return np.array([np.sum(np.abs(signal[i:i+frame_size])**2) for i in range(0, len(signal), frame_size)])

    @staticmethod
    def clamp_below_threshold(signal, energy, threshold, frame_size):
        """
        Clamp the signal windows below the threshold to 0.

        Args:
            signal (ndarray): Input audio signal.
            energy (ndarray): Short-term energy of the signal.
            threshold (float): Energy threshold.
            frame_size (int): Size of the frame.

        Returns:
            ndarray: Clamped audio signal.
        """
        clamped_signal = signal.copy()
        for i, e in enumerate(energy):
            if e < threshold:
                clamped_signal[i*frame_size:(i+1)*frame_size] = 0
        return clamped_signal

    @staticmethod
    def remove_below_threshold(signal, energy, threshold, frame_size):
        """
        Remove the signal windows below the threshold.

        Args:
            signal (ndarray): Input audio signal.
            energy (ndarray): Short-term energy of the signal.
            threshold (float): Energy threshold.
            frame_size (int): Size of the frame.

        Returns:
            ndarray: Processed audio signal.
        """
        indices = np.where(energy >= threshold)[0]
        return np.concatenate([signal[i*frame_size:(i+1)*frame_size] for i in indices])

    @staticmethod
    def apply(data: np.ndarray, sample_rate: int, window_ms: float, noise_threshold_percentile: float) -> np.ndarray:
        """
        Apply a noise gate to an audio signal.

        Args:
            data (ndarray): Input audio signal.
            sample_rate (int): Sampling rate of the audio signal.
            window_ms (float): Length of the window in milliseconds.
            noise_threshold_percentile (float): Percentile threshold for noise.

        Returns:
            ndarray: Processed audio signal.
        """
        frame_size = int(window_ms * sample_rate /
                         1000)  # Convert ms to samples
        energy = NoiseGate.short_term_energy(data, frame_size)
        threshold = np.percentile(energy, noise_threshold_percentile)
        clamped_audio = NoiseGate.clamp_below_threshold(
            data, energy, threshold, frame_size)
        return clamped_audio

    @staticmethod
    def apply_and_remove(data: np.ndarray, sample_rate: int, window_ms: float, noise_threshold_percentile: float) -> np.ndarray:
        """
        Apply a noise gate and remove noise from an audio signal.

        Args:
            data (ndarray): Input audio signal.
            sample_rate (int): Sampling rate of the audio signal.
            window_ms (float): Length of the window in milliseconds.
            noise_threshold_percentile (float): Percentile threshold for noise.

        Returns:
            ndarray: Processed audio signal.
        """
        frame_size = int(window_ms * sample_rate /
                         1000)  # Convert ms to samples
        energy = NoiseGate.short_term_energy(data, frame_size)
        threshold = np.percentile(energy, noise_threshold_percentile)
        removed_silence_audio = NoiseGate.remove_below_threshold(
            data, energy, threshold, frame_size)
        return removed_silence_audio

    @staticmethod
    @lru_cache(maxsize=10)  # Caching the filter coefficients
    def _get_bandpass_filter_coefficients(sample_rate, low_freq, high_freq):
        """
        Get bandpass filter coefficients.

        Args:
            sample_rate (int): Sampling rate of the audio signal.
            low_freq (float): Lower frequency cutoff of the filter.
            high_freq (float): Upper frequency cutoff of the filter.

        Returns:
            tuple: Filter coefficients (b, a).
        """
        nyq = 0.5 * sample_rate
        low = low_freq / nyq
        high = high_freq / nyq
        return butter(5, [low, high], btype='band', analog=False)

    @staticmethod
    def apply_band_pass_filtfilt(data, sample_rate, low_freq, high_freq):
        """
        Apply a bandpass filter to an audio signal.

        Args:
            data (ndarray): Input audio signal.
            sample_rate (int): Sampling rate of the audio signal.
            low_freq (float): Lower frequency cutoff of the filter.
            high_freq (float): Upper frequency cutoff of the filter.

        Returns:
            ndarray: Filtered audio signal.
        """
        b, a = NoiseGate._get_bandpass_filter_coefficients(
            sample_rate, low_freq, high_freq)

        try:
            filtered_data = filtfilt(b, a, data)
        except Exception as e:
            print(f"Filtering failed: {e}")
            return data

        return filtered_data

    @staticmethod
    @lru_cache(maxsize=10)  # Caching the filter coefficients
    def _get_highpass_filter_coefficients(sample_rate, low_freq):
        """
        Get highpass filter coefficients.

        Args:
            sample_rate (int): Sampling rate of the audio signal.
            low_freq (float): Lower frequency cutoff of the filter.

        Returns:
            tuple: Filter coefficients (b, a).
        """
        nyq = 0.5 * sample_rate
        cutoff = low_freq / nyq
        return butter(5, cutoff, btype='high', analog=False)

    @staticmethod
    def apply_highpass_filtfilt(data, sample_rate, low_freq):
        """
        Apply a highpass filter to an audio signal.

        Args:
            data (ndarray): Input audio signal.
            sample_rate (int): Sampling rate of the audio signal.
            low_freq (float): Lower frequency cutoff of the filter.

        Returns:
            ndarray: Filtered audio signal.
        """
        b, a = NoiseGate._get_highpass_filter_coefficients(
            sample_rate, low_freq)

        try:
            filtered_data = filtfilt(b, a, data)
        except Exception as e:
            print(f"Filtering failed: {e}")
            return data

        return filtered_data
