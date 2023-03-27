from pydub import AudioSegment
import numpy as np
import gradio as gr
import wave 
from scipy.io import wavfile



def read_audio_from_file(filepath):
    # 读取MP3文件并转换为WAV格式
    sound = AudioSegment.from_mp3(filepath)
    sound.export("./output/edgeBot.wav", format="wav")
    samplerate, data = wavfile.read('./output/edgeBot.wav')
    return samplerate, data


def save_wav_file(file_path, audio, sample_rate):
    """
    Save a numpy array as a WAV file.

    Parameters:
    file_path (str): Path to save the WAV file.
    audio (np.ndarray): Numpy array containing the audio data.
    sample_rate (int): Sample rate of the audio data.

    Returns:
    None
    """
    # convert audio data to 16-bit integers and scale to the range of -32768 to 32767
    scaled_audio = np.int16(audio * 32767)
    # save the WAV file
    wavfile.write(file_path, sample_rate, scaled_audio)
