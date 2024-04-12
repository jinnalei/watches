"""
Module for audio conversions.

Author: Luis Ramirez
"""

from pydub import AudioSegment
from io import BytesIO
import librosa



class AudioConverter:

    @staticmethod
    def augment_audio_by_encoding_transform(filepath, encoding, bitrate, sample_rate):
        # Use context manager for working with AudioSegment
        with open(filepath, 'rb') as f:
            original_audio = AudioSegment.from_wav(f)

        # Use BytesIO for the encoded audio with a context manager
        with BytesIO() as encoded_audio_bytes:
            original_audio.export(encoded_audio_bytes,
                                  format=encoding, bitrate=bitrate)
            encoded_audio_bytes.seek(0)

            # Load the encoded audio bytes using pydub
            encoded_audio = AudioSegment.from_file(
                encoded_audio_bytes, format=encoding)

            # Use another BytesIO for the WAV conversion with a context manager
            with BytesIO() as wav_bytes:
                encoded_audio.export(wav_bytes, format='wav')
                wav_bytes.seek(0)

                # Load the WAV bytes using librosa
                decoded_audio, sr = librosa.load(
                    wav_bytes, sr=sample_rate, mono=True)

        # Return the decoded WAV array and sample rate outside the context manager
        return decoded_audio, sr

