import contextlib
import os
import socket
import time
import wave

import numpy as np
import pyaudio
import sounddevice as sd
import soundfile as sf

from utils import util, conf

np.random.seed(conf.seed)

SIZE = 160
PORT_SEND = 6000
PORT_RECEIVE = 6000
IP = '202.31.200.71'
times = []

names = np.load('dataset/labels/OVISION_names.npy', allow_pickle=True).item()
with open(conf.name_id_path, encoding='utf-8') as reader:
    ids_names = reader.readlines()
ids = {}
for id_name in ids_names:
    id_, name = id_name.rstrip().split(',')
    ids[name] = id_

def read_list(list_file):
    with open(list_file, "r", encoding='utf-8') as reader:
        lines = reader.readlines()
        list_sig = []
        for x in lines:
            list_sig.append(x.rstrip())
    return list_sig


def record_voice(files, base_path):
    for file_path in files:
        play_audio(file_path)
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=16000,
                            input=True,
                            frames_per_buffer=SIZE)
        names = file_path.split('/')
        print("--- ", end="")
        print(os.path.join(base_path, names[1], names[2]))
        frames = []
        for i in range(0, int(16000 / SIZE * get_duration('dataset/' + file_path))):
            data = stream.read(SIZE)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        audio.terminate()
        util.check_dir(path=base_path + names[1])
        util.save_wav(frames, file_path=os.path.join(base_path, names[1], names[2]))


def file_based(files):
    receiver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    receiver.bind(('', PORT_RECEIVE))
    receiver.listen(100)
    t_count = 0
    f_count = 0
    wrong_paths = []
    try:
        for path in files:
            sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sender.connect((IP, PORT_SEND))
            start_time = time.time()
            frames = []
            wf = wave.open('./dataset/' + path, 'rb')
            wav_bytes = wf.readframes(SIZE)
            while wav_bytes != b'':
                frames.append(wav_bytes)
                sender.sendall(wav_bytes)
                wav_bytes = wf.readframes(SIZE)
            sender.close()
            print(path)
            print("nb_frames : {}".format(len(frames)))
            connection, address = receiver.accept()
            data_received = connection.recv(2 * SIZE)
            data_received = data_received.decode()
            end_time = time.time()
            print('Result : {}'.format(data_received))
            t_name = conf.label_dict[path]
            r_name = int(data_received.rstrip())
            if t_name == r_name:
                t_count += 1
                print("True")
            else:
                wrong_paths.append(path)
                f_count += 1
                print("False")
            print('Time : {}'.format(end_time - start_time))
            print("--------------------------------------")
    finally:
        receiver.close()
    print("Correct :", t_count)
    print("Wrong :", f_count)
    print(wrong_paths)
    with open("results/wrong_results.txt", 'w') as writer:
        for wrong_path in wrong_paths:
            writer.write(wrong_path + '\n')


def play_audio(path):
    sd.playrec(data=sf.read('dataset/' + path)[0], samplerate=16000, channels=1)


def get_duration(f_name):
    with contextlib.closing(wave.open(f_name, 'r')) as f:
        nb_frames = f.getnframes()
        rate = f.getframerate()
        duration = nb_frames / float(rate)
        return duration


def mic_based(files):
    t_count = 0
    f_count = 0
    wrong_files = []
    for index, file_path in enumerate(files):
        try:
            if os.path.exists('dataset/' + file_path):
                play_audio(file_path)
                audio = pyaudio.PyAudio()
                stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=SIZE)
                print("--- ", end='')
                frames = []
                start_time = time.time()
                sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sender.connect((IP, PORT_SEND))
                s_time = time.time()
                for i in range(0, int(16000 / SIZE * get_duration('dataset/' + file_path))):
                    data = stream.read(SIZE)
                    frames.append(data)
                    sender.sendall(data)
                stream.stop_stream()
                stream.close()
                audio.terminate()
                sender.shutdown(socket.SHUT_WR)
                times.append(time.time() - s_time)
                data_received = sender.recv(2 * SIZE)
                print(data_received.decode(), end='')
                end_time = time.time()

                print(', time : {}'.format(end_time - start_time))
                gt_id = int(ids[util.get_keys_by_value(names, int(conf.label_dict[file_path]))[0]])
                pred_id = int(data_received.decode())
                if gt_id == pred_id:
                    t_count += 1
                else:
                    f_count += 1
                    wrong_files.append(file_path + '-' + data_received.decode() + '\n')
        except:
            pass
    print("Correct : {}".format(t_count))
    print("Wrong : {}".format(f_count))
    for wrong_file in wrong_files:
        print(str(wrong_file).rstrip())


save_path_ = 'dataset\\OVISION_CMD_britz_jtum_40\\'
voice_path_ = 'dataset/labels/OVISION_test_mic.scp'

mic_based(read_list(voice_path_))
