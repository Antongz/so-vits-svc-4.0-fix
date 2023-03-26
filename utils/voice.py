from pydub import AudioSegment
import numpy as np


def read_audio_from_file(file_path):
    # 使用pydub加载mp3文件
    audio = AudioSegment.from_file(file_path, format="mp3")
    # 将音频格式转换为wav
    audio_temp = audio.set_frame_rate(16000).set_channels(1)
    # 将音频数据转换为numpy数组
    audio = np.array(audio_temp.get_array_of_samples())
    # 获取采样率
    sample_rate = audio_temp.frame_rate
    return (sample_rate, audio)
