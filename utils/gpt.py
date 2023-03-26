import os
import openai
import subprocess
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

# 根据之前的历史生成回复
def chatgpt_clone(input:str, history=[]):
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output = openai_create(inp)
    text_To_Speech(output)
    history.append((input, output))
    return history, history

def text_To_Speech(text):
    tts = gTTS(text=text, lang='zh')
    tts.save('bot.wav')
    return 'bot.wav'
    #if os.name == 'posix':  # macOS / Linux
        #subprocess.call(['afplay', 'temp.wav'])
    #else:  # Windows
        #subprocess.call(['start', 'temp.wav'])