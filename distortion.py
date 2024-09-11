import os
import random
import torch
import torchaudio

import numpy as np
import soundfile as sf
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
from tqdm import tqdm
import string

def fixed_dis(SAMPLE_SPEECH_DIR, SAMPLE_NOISE_LIST, TARGET_FOLDER, snr_list = [20, 10, 3]):
    # 获取语音文件列表
    speech_files = [os.path.join(SAMPLE_SPEECH_DIR, f) for f in os.listdir(SAMPLE_SPEECH_DIR) if f.endswith('.wav')]

    # 循环处理每个语音文件
    for speech_file in tqdm(speech_files):
        # 读取语音文件
        speech, speech_sr = torchaudio.load(speech_file)

        # 随机选择一个噪声文件
        noise_file = random.choice(SAMPLE_NOISE_LIST)
        noise, noise_sr = torchaudio.load(noise_file)
        
        resampler = T.Resample(orig_freq=noise_sr, new_freq=speech_sr)
        noise = resampler(noise)
        # output_noise_file = os.path.join(TARGET_FOLDER, f"resampled_{os.path.basename(noise_file)}")
        # torchaudio.save(output_noise_file, noise, sample_rate=speech_sr)

        # 如果噪声比语音短，则重复噪声以匹配语音长度
        if noise.size(1) < speech.size(1):
            repeats = (speech.size(1) // noise.size(1)) + 1
            noise = noise.repeat(1, repeats)[:, :speech.size(1)]
        # 如果噪声比语音长，则随机选择一个片段
        else:
            max_start = noise.size(1) - speech.size(1)
            start = random.randint(0, max_start)
            noise = noise[:, start:start + speech.size(1)]

        # 添加不同信噪比的噪声到语音文件
        snr_dbs = torch.tensor(snr_list)
        noisy_speeches = F.add_noise(speech, noise, snr_dbs)

        # 将混合的音频文件切分并写入目标文件夹
        for i in range(len(snr_dbs)):
            noisy_speech = noisy_speeches[i, :]

            # 转换为 NumPy 数组
            noisy_speech_np = noisy_speech.numpy().flatten()

            # 创建子目录
            snr_folder = os.path.join(TARGET_FOLDER, f'SNR_{snr_dbs[i]}dB')
            os.makedirs(snr_folder, exist_ok=True)

            # 获取噪声文件的基本名称和开始位置
            noise_basename = os.path.basename(noise_file).split('.')[0]
            noise_start = start / noise_sr

            # 写入单独的音频文件到子目录中
            output_file = os.path.join(snr_folder, f"{os.path.basename(speech_file).split('.')[0]}_noisy_{noise_basename}_start_{noise_start}.wav")
            sf.write(output_file, noisy_speech_np, speech_sr)

def process_audio(speech, noise):
    # print(type(speech))
    # print(type(noise))
    speech_length = speech.size(1)
    noise_length = noise.size(1)

    # 如果语音比噪音长
    if speech_length > noise_length:
        # 对噪音进行零填充
        padding = torch.zeros(1, speech_length - noise_length)
        noise = torch.cat((noise, padding), dim=1)
    else:
        # 噪音比语音长或一样长，随机截取噪音
        max_start = noise_length - speech_length
        start = random.randint(0, max_start)
        noise = noise[:, start:start + speech_length]

    return noise

def random_silence_segments(speech, noise, num_segments, sample_rate=16000):
    assert speech.size(1) == noise.size(1), "Speech and noise must be of the same length"

    noise_length = noise.size(1)
    segment_durations = []

    for _ in range(num_segments):
        duration = random.uniform(0.5, 1.0) * sample_rate
        segment_durations.append(int(duration))

    segment_starts = []
    for duration in segment_durations:
        start = random.randint(0, noise_length - duration)
        segment_starts.append(start)

    mask = torch.zeros_like(noise)
    for start, duration in zip(segment_starts, segment_durations):
        end = start + duration
        mask[:, start:end] = 1

    noisy_segments = noise * mask
    return noisy_segments

def generate_random_string(length=5):
    letters = string.ascii_letters  # 包含所有英文字母（大写和小写）
    random_string = ''.join(random.choice(letters) for _ in range(length))
    return random_string

def random_dis(SAMPLE_SPEECH_DIR, SAMPLE_NOISE_LIST, TARGET_FOLDER, max_snr = 25, min_snr = 3):
    # 获取语音文件列表
    speech_files = [os.path.join(SAMPLE_SPEECH_DIR, f) for f in os.listdir(SAMPLE_SPEECH_DIR) if f.endswith('.wav')]

    # 循环处理每个语音文件
    for speech_file in tqdm(speech_files):
        speech, speech_sr = torchaudio.load(speech_file)
        noise_file = random.choice(SAMPLE_NOISE_LIST)
        noise, _ = librosa.load(noise_file, sr=speech_sr)
        noise = torch.tensor(noise).unsqueeze(0)
        noise = process_audio(speech, noise)
        num_segments = random.randint(1, 5)
        noise = random_silence_segments(speech, noise, num_segments, speech_sr)
        
        snr = random.randint(min_snr, max_snr)
        noisy_speeches = F.add_noise(speech, noise, torch.tensor([snr]))
        noisy_speech_np = noisy_speeches.numpy().flatten()
        sf.write(os.path.join(TARGET_FOLDER,f"{os.path.basename(speech_file).split('.')[0]}_seg_{num_segments}_snr_{snr}.wav"),noisy_speech_np,speech_sr)


if __name__=='__main__':
    # 设置路径
    SAMPLE_SPEECH_DIR = '/mnt/md1/user_wago/data/LJSpeech-1.1/wavs/'
    # SAMPLE_NOISE_LIST = ['/mnt/md1/user_wago/data/musan/speech/librivox/speech-librivox-0003.wav', '/mnt/md1/user_wago/data/musan/speech/librivox/speech-librivox-0011.wav', '/mnt/md1/user_wago/data/musan/speech/librivox/speech-librivox-0017.wav']
    SAMPLE_NOISE_LIST = ['/mnt/md1/user_wago/data/musan/music/jamendo/music-jamendo-0011.wav', '/mnt/md1/user_wago/data/musan/noise/free-sound/noise-free-sound-0001.wav', '/mnt/md1/user_wago/data/musan/noise/free-sound/noise-free-sound-0005.wav' ,'/mnt/md1/user_wago/data/musan/noise/sound-bible/noise-sound-bible-0030.wav', '/mnt/md1/user_wago/data/musan/speech/librivox/speech-librivox-0011.wav']
    TARGET_FOLDER = '/mnt/md1/user_wago/data/LJSpeech-1.1/distortion/random'
    
    random_dis(SAMPLE_SPEECH_DIR, SAMPLE_NOISE_LIST, TARGET_FOLDER)
    # speech_file = '/mnt/md1/user_wago/data/LJSpeech-1.1/wavs/LJ001-0001.wav'
    # speech, speech_sr = torchaudio.load(speech_file)
    # for noise_file in SAMPLE_NOISE_LIST:
    #     noise, _ = librosa.load(noise_file, sr=speech_sr)
    #     noise = torch.tensor(noise).unsqueeze(0)
    #     noise = process_audio(speech, noise)
    #     noise = random_silence_segments(speech, noise, speech_sr)
        
    #     snr = random.randint(3, 25)
    #     noisy_speeches = F.add_noise(speech, noise, torch.tensor([snr]))
    #     noisy_speech_np = noisy_speeches.numpy().flatten()
    #     sf.write(os.path.join(TARGET_FOLDER,f'{generate_random_string()}_{snr}.wav'),noisy_speech_np,speech_sr)
        