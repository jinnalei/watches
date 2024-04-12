"""
Module for audio augmentation operations.

This module provides classes and functions to augment audio data, including adding
Gaussian noise, adjusting duration, applying time masking, and encoding to MP3 format.

Author: Luis Ramirez
"""

import os
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from audiomentations import AddGaussianNoise, AddGaussianSNR, AdjustDuration, TimeMask
from kyyras_audio_processing.audio_converter import AudioConverter

class AudioAugmentation:
    """
    Class for augmenting audio files.
    """

    @staticmethod
    def process_file(file, sample_rate):
        """
        Process a single audio file.

        Args:
            file (str): Path to the audio file.
            sample_rate (int): Target sample rate.

        Returns:
            str: Path to the processed file.
        """
        directory_path = os.path.dirname(file)
        audio, sr = librosa.load(file, sr=sample_rate)

        AudioAugmentation.augment_audio_data_AddGaussianNoise(file, directory_path,
                                                              audio, sr)

        AudioAugmentation.augment_audio_data_AddGaussianSNR(file, directory_path,
                                                            audio, sr)

        AudioAugmentation.augment_audio_data_by_mp3_encoding(
            file, directory_path, sample_rate)

        return f"{file} processed"

    @staticmethod
    def augment_files(files_dict, process_file_callback, parallel, sample_rate):
        """
        Augment multiple audio files.

        Args:
            files_dict (dict): Dictionary containing lists of files.
            process_file_callback (function): Callback function to handle processed files.
            parallel (bool): Whether to process files in parallel.
            sample_rate (int): Target sample rate.
        """
        all_files = [file for files in files_dict.values() for file in files]

        if parallel:
            with ThreadPoolExecutor() as executor:
                future_to_file = {executor.submit(
                    AudioAugmentation.process_file, file, sample_rate): file for file in all_files}

                for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing files"):
                    file = future_to_file[future]
                    try:
                        result = future.result()
                        process_file_callback(result)
                    except Exception as exc:
                        print(f"{file} generated an exception: {exc}")
        else:
            for file in tqdm(all_files, desc="Processing files"):
                try:
                    result = AudioAugmentation.process_file(file, sample_rate)
                    process_file_callback(result)
                except Exception as exc:
                    print(f"{file} generated an exception: {exc}")

    @staticmethod
    def augment_audio_data_by_mp3_encoding(file_path, target_dir, sample_rate):
        """
        Augment audio data by encoding it to MP3 format.

        Args:
            file_path (str): Path to the audio file.
            target_dir (str): Directory to save augmented files.
            sample_rate (int): Target sample rate.
        """
        filename, extension = os.path.splitext(os.path.basename(file_path))

        # List of target bit rates
        bit_rates = ['128k', '160k', '192k', '320k']

        for bit_rate in bit_rates:
            try:
                decoded_audio, sr = AudioConverter.augment_audio_by_encoding_transform(
                    file_path, 'mp3', bit_rate, sample_rate)

                new_filename = f"{filename}_{bit_rate}{extension}"
                output_path = os.path.join(target_dir, new_filename)
                sf.write(output_path, decoded_audio, sr, format='WAV')
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    @staticmethod
    def augment_audio_data_AddGaussianNoise(file_path, target_dir, waveform, sample_rate):
        """
        Add Gaussian noise to audio data.

        Args:
            file_path (str): Path to the audio file.
            target_dir (str): Directory to save augmented files.
            waveform (ndarray): Audio waveform data.
            sample_rate (int): Sample rate of the audio.
        """
        augment = AddGaussianNoise(
            min_amplitude=0.001, max_amplitude=0.015, p=1.0)
        augmented_waveform = augment(samples=waveform, sample_rate=sample_rate)

        filename, extension = os.path.splitext(os.path.basename(file_path))
        new_filename = f"{filename}_gaussian_noise{extension}"
        output_path = os.path.join(target_dir, new_filename)

        sf.write(output_path, augmented_waveform, sample_rate)

    @staticmethod
    def augment_audio_data_AddGaussianSNR(file_path, target_dir, waveform, sample_rate):
        """
        Add Gaussian Signal-to-Noise Ratio (SNR) to audio data.

        Args:
            file_path (str): Path to the audio file.
            target_dir (str): Directory to save augmented files.
            waveform (ndarray): Audio waveform data.
            sample_rate (int): Sample rate of the audio.
        """
        augment = AddGaussianSNR(min_snr_db=0.3, max_snr_db=0.8, p=1.0)
        augmented_waveform = augment(samples=waveform, sample_rate=sample_rate)

        filename, extension = os.path.splitext(os.path.basename(file_path))
        new_filename = f"{filename}_gaussian_snr{extension}"
        output_path = os.path.join(target_dir, new_filename)

        sf.write(output_path, augmented_waveform, sample_rate)

    @staticmethod
    def augment_audio_data_AdjustDuration(file_path, target_dir, waveform, sample_rate, duration_in_seconds):
        """
        Adjust the duration of audio data.

        Args:
            file_path (str): Path to the audio file.
            target_dir (str): Directory to save augmented files.
            waveform (ndarray): Audio waveform data.
            sample_rate (int): Sample rate of the audio.
            duration_in_seconds (float): Target duration in seconds.
        """
        augment = AdjustDuration(duration_seconds=duration_in_seconds)
        augmented_waveform = augment(samples=waveform, sample_rate=sample_rate)

        filename, extension = os.path.splitext(os.path.basename(file_path))
        new_filename = f"{filename}_adjust_duration{extension}"
        output_path = os.path.join(target_dir, new_filename)

        sf.write(output_path, augmented_waveform, sample_rate)

    @staticmethod
    def augment_audio_data_TimeMask(file_path, target_dir, waveform, sample_rate):
        """
        Apply time masking to audio data.

        Args:
            file_path (str): Path to the audio file.
            target_dir (str): Directory to save augmented files.
            waveform (ndarray): Audio waveform data.
            sample_rate (int): Sample rate of the audio.
        """
        augment = TimeMask(min_band_part=0.05,
                           max_band_part=0.1, fade=True, p=.5)
        augmented_waveform = augment(samples=waveform, sample_rate=sample_rate)

        filename, extension = os.path.splitext(os.path.basename(file_path))
        new_filename = f"{filename}_time_mask{extension}"
        output_path = os.path.join(target_dir, new_filename)

        sf.write(output_path, augmented_waveform, sample_rate)
