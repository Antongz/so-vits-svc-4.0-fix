import os
import openai
import subprocess
import gradio as gr
from gtts import gTTS

# 获取api生成的回复
def openai_create(prompt: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.9,
        messages=[
            {
                "role": "user",  # whether we will act as user or assistant.
                "content": prompt,
            }
        ],
    )
    return response["choices"][0]["message"]["content"]
        
def text_To_Speech(text):
    tts = gTTS(text=text, lang='zh')
    tts.save('./output/bot.wav')
    # 将 AI 生成的文本传递给 edge-tts 命令
    command = f'edge-tts --voice zh-CN-XiaoyiNeural --text "{text}" --write-media ./output/edgeBot.mp3'  
    subprocess.run(command, shell=True)  # 执行命令行指令
    #if os.name == 'posix':  # macOS / Linux
        #subprocess.call(['afplay', 'temp.wav'])
    #else:  # Windows
        #subprocess.call(['start', 'temp.wav'])