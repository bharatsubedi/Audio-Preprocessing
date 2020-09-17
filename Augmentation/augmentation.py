import contextlib
import os
import time
import wave

import numpy as np
import pyaudio
import sounddevice as sd
import soundfile as sf

from utils import util, conf
from utils.voice_util.sound import Sound

np.random.seed(conf.seed)
SIZE = 160
sound_volumes = [25, 35, 40]


def get_duration(f_name):
    with contextlib.closing(wave.open(f_name, 'r')) as f:
        nb_frames = f.getnframes()
        rate = f.getframerate()
        duration = nb_frames / float(rate)
        return duration


def play_audio(path):
    sd.playrec(data=sf.read('dataset/' + path)[0], samplerate=16000, channels=1)


def record_voice(files, base_path):
    for file_path in files:
        play_audio(file_path)
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=16000,
                            input=True,
                            frames_per_buffer=SIZE)
        dir_names = file_path.split('/')
        print("--- ", end="")
        print(os.path.join(base_path, dir_names[-2], dir_names[-1]))
        frames = []
        for i in range(0, int(16000 / SIZE * get_duration('dataset/' + file_path))):
            data = stream.read(SIZE)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        audio.terminate()
        util.check_dir(path=base_path + dir_names[-2])
        util.save_wav(frames, file_path=os.path.join(base_path, dir_names[-2], dir_names[-1]))


save_path_ = 'dataset\\{}\\OVISION_{}_britz_microsoft_25\\'.format('CMD', 'CMD')
voice_path_ = 'dataset/labels/OVISION_recording.scp'
record_voice(util.read_list(voice_path_), save_path_)

Sound.volume_set(35)
time.sleep(5)

save_path_ = 'dataset\\{}\\OVISION_{}_britz_microsoft_35\\'.format('CMD', 'CMD')
voice_path_ = 'dataset/labels/OVISION_recording.scp'
record_voice(util.read_list(voice_path_), save_path_)

Sound.volume_set(40)
time.sleep(5)

save_path_ = 'dataset\\{}\\OVISION_{}_britz_microsoft_40\\'.format('CMD', 'CMD')
voice_path_ = 'dataset/labels/OVISION_recording.scp'
record_voice(util.read_list(voice_path_), save_path_)
