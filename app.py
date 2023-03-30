import io
import os
import openai
import argparse
import gradio as gr
import librosa
import numpy as np
import soundfile
from inference.infer_tool import Svc
from scipy.io import wavfile
from utils import *
import logging
import json


parser = argparse.ArgumentParser()
parser.add_argument("--user", type=str, help='set gradio user', default=None)
parser.add_argument("--password", type=str,
                    help='set gradio password', default=None)
parser.add_argument('--share', action='store_true', help='enable sharing')
parser.add_argument('--key', help='openai api key', default="your openai key")
cmd_opts = parser.parse_args()
share = cmd_opts.share
key = cmd_opts.key

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

openai.api_key = key
messages = [{"role": "system", "content": ''}]

# google colab 环境判断


def is_google_colab():
    try:
        import google.colab
        return True
    except:
        return False

# 加载模型


def load_model_func(ckpt_name, cluster_name, config_name):
    global model, cluster_model_path
    config_path = "configs/" + config_name
    with open(config_path, encoding='utf-8-sig', errors='ignore') as f:
        print("load config from %s" % config_path, f)
        config = json.load(f)
    spk_dict = config["spk"]
    ckpt_path = "logs/44k/" + ckpt_name
    cluster_path = "logs/44k/" + cluster_name
    if cluster_name == "no_clu":
        model = Svc(ckpt_path, config_path)
    else:
        model = Svc(ckpt_path, config_path, cluster_model_path=cluster_path)
    spk_list = list(spk_dict.keys())
    return "模型加载成功", gr.Dropdown.update(choices=spk_list)


# 读取ckpt和cluster列表
file_list = os.listdir("logs/44k")
ckpt_list = [ck for ck in file_list if os.path.splitext(
    ck)[-1] == ".pth" and ck[0] != "k"]
cluster_list = [ck for ck in file_list if ck[0] == "k"]
if not cluster_list:
    cluster_list = ["你没有聚类模型"]
    print("no clu, 你没有聚类模型")

# 合成语音


def vc_fn(sid, input_audio, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale):
    if input_audio is None:
        return "请上传音频文件", None
    sampling_rate, audio = input_audio
    print(audio.shape,sampling_rate, audio, audio.dtype)
    duration = audio.shape[0] / sampling_rate
    if duration > 90:
        return "请上传小于90s的音频，需要转换长音频请本地进行转换", None
    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    out_wav_path = "./output/temp.wav"
    soundfile.write(out_wav_path, audio, 16000, format="wav")
    samplerate, data = wavfile.read(out_wav_path)
    print(samplerate, data, data.dtype)
    print(cluster_ratio, auto_f0, noise_scale)
    _audio = model.slice_inference(
        out_wav_path, sid, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale)
    save_wav_file("./output/edgeBotFinal.wav", _audio, 44100)
    return "Success", (44100, _audio)


# 根据之前的历史生成回复
def chatgpt_clone(input, history, sid, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale):
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output = openai_create(inp)
    text_To_Speech(output)
    history.append((input, output))
    # 从./output/edgeBot.wav 提取音频档
    audio_ = read_audio_from_file("./output/edgeBot.mp3")
    text, audio = vc_fn(sid, audio_, vc_transform, auto_f0,
                  cluster_ratio, slice_db, noise_scale)
    return history, history, audio


def text_to_speech_clone( input, sid, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale):
    text_To_Speech(input)
    audio_ = read_audio_from_file("./output/edgeBot.mp3")
    text, audio = vc_fn(sid, audio_, vc_transform, auto_f0,
                  cluster_ratio, slice_db, noise_scale)
    test_sound = AudioSegment.from_file("./output/edgeBotFinal.wav", "wav")
    match_target_amplitude(test_sound, -30.0)
    return audio

# 默认prompt 参数
prompt = ""

app = gr.Blocks()
with app:
    with gr.Row():
        with gr.Column():
            voice_uid = "xingtong"
            cover = f"assets/{voice_uid}/{voice_uid}.png" if os.path.exists(
                f"assets/{voice_uid}/{voice_uid}.png") else None
            gr.Markdown(
            '<div align="center">'
            f'<img style="width:auto;height:300px;" src="file/{cover}">' if cover else ""
            '</div>'
            )

    with gr.Tabs():
        with gr.TabItem("语音推理"):
            with gr.Row():
                with gr.Column():

                    with gr.Row():
                        with gr.Column():
                            choice_ckpt = gr.Dropdown(
                                label="模型选择", choices=ckpt_list, value="")
                        with gr.Column():
                            config_choice = gr.Dropdown(
                                label="配置文件", choices=os.listdir("configs"), value="")
                        with gr.Column():
                            cluster_choice = gr.Dropdown(
                                label="聚类模型", choices=cluster_list, value="")

                    gr.Markdown(value="""
                    请稍等片刻，模型加载大约需要10秒。后续操作不需要重新加载模型
                    """)

                    loadckpt = gr.Button("加载模型", variant="primary")

                    with gr.Row():
                        with gr.Column():
                            sid = gr.Dropdown(label="音色", value="")
                        with gr.Column():
                            model_message = gr.Textbox(label="Output Message")

                    loadckpt.click(load_model_func, [choice_ckpt, cluster_choice, config_choice], [
                                   model_message, sid])

                    vc_transform = gr.Number(
                        label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
                    cluster_ratio = gr.Number(
                        label="聚类模型混合比例，0-1之间，默认为0不启用聚类，能提升音色相似度，但会导致咬字下降（如果使用建议0.5左右）", value=0)
                    auto_f0 = gr.Checkbox(
                        label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声不要勾选此项会究极跑调）", value=False)

                    with gr.Row():
                        with gr.Column():
                            slice_db = gr.Number(label="切片阈值", value=-40)
                        with gr.Column():
                            noise_scale = gr.Number(
                                label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)

                with gr.Column():
                    gr.Markdown(
                    '## <center> </br> \n'
                    "\n\n"
                    '# <center> sovits4.0 webui 推理\n'
                    "### <center>感谢 bilibili@麦哲云 bilibili@羽毛布団\n"
                    "### <center>禁止用于血腥、暴力、性相关、政治相关内容\n"
                    "### <center>Colab适配与功能优化 By Antongz Lucwu\n"
                    '## <center> </br> \n'
                    )
                    vc_input3 = gr.Audio(label="上传音频（长度小于90秒）")
                    vc_output1 = gr.Textbox(label="Output Message")
                    vc_output2 = gr.Audio(label="Output Audio")
                    vc_submit = gr.Button("转换", variant="primary")
                    print(vc_input3, vc_input3.value)
                    vc_submit.click(vc_fn, [sid, vc_input3, vc_transform, auto_f0,
                                            cluster_ratio, slice_db, noise_scale], [vc_output1, vc_output2])

        with gr.TabItem("gpt语音对话"):
            chatbot = gr.Chatbot()
            state = gr.State([])
            gpt_output = gr.Audio(label="Output Audio")
            message = gr.Textbox(placeholder=prompt)
            submit = gr.Button("发送")
            submit.click(chatgpt_clone, [message, state, sid, vc_transform,
                                         auto_f0, cluster_ratio, slice_db, noise_scale], [chatbot, state, gpt_output])

        with gr.TabItem("TTS+音色转换"):
            text_input = gr.Textbox(placeholder="请输入你的文本")
            sound_output = gr.Audio(label="Output Audio")
            submit_text = gr.Button("发送")
            submit_text.click(text_to_speech_clone, [text_input, sid, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale], [sound_output])
            
app.launch(share=share)
