import collections
import functools
import operator
import socket
import time
import pyaudio
import tensorflow as tf

from utils.conf import *
from utils.dnn import get_sinc_net
from utils.util import wav2array, save_wav, get_keys_by_value

names = np.load('dataset/labels/OVISION_names.npy', allow_pickle=True).item()
with open(name_id_path, encoding='utf-8') as reader:
    ids_names = reader.readlines()
ids = {}
for id_name in ids_names:
    id_, name = id_name.rstrip().split(',')
    ids[name] = id_

tf.compat.v1.set_random_seed(seed)
np.random.seed(seed)

model = get_sinc_net()
model.load_weights(pt_file)

SIZE = 160
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
PORT_RECEIVE = 6000
PORT_SEND = 6001

receiver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
receiver.bind(('', PORT_RECEIVE))
receiver.listen(100)
print("- - - WAITING FOR CLIENT - - -")
while True:
    connection, address = receiver.accept()
    data_arrays = []
    count = 0
    res_list = []
    nb_channels = 1
    predictions = []
    total_frames = []
    detection_frames = []
    print('-- connected by', address, end='')
    start_time = time.time()
    try:
        while True:
            data_bytes = connection.recv(2 * SIZE)
            if not data_bytes:
                break
            total_frames.append(data_bytes)
            data_array = wav2array(1, 2, data_bytes)
            power = 1.0 / (2 * data_array.size + 1) * np.sum(np.copy(data_array).astype(float) ** 2) / 16000
            if power > 0:
                detection_frames.append(data_bytes)
                data_array = data_array / 32767.0
                if data_array.shape[0] == SIZE:
                    data_arrays.append(data_array)
                    if len(data_arrays) == 20:
                        label = {}
                        x = np.array(data_arrays)
                        del data_arrays[0]
                        x = x.reshape(1, w_length)[..., np.newaxis]

                        prediction = model.predict(x)
                        best_class = np.argmax(np.sum(prediction, axis=0))
                        accuracy = prediction[0][best_class]
                        label[best_class] = accuracy
                        res_list.append([best_class, accuracy])
                        predictions.append(label)

                        label = {}
                        prediction = np.delete(prediction, best_class)
                        best_class = np.argmax(prediction)
                        accuracy = prediction[best_class]
                        label[best_class] = accuracy
                        predictions.append(label)

                        label = {}
                        prediction = np.delete(prediction, best_class)
                        best_class = np.argmax(prediction)
                        accuracy = prediction[best_class]
                        label[best_class] = accuracy
                        predictions.append(label)

        results = dict(functools.reduce(operator.add, map(collections.Counter, predictions)))
        results = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
        label_class = results[0][0]
        label = np.full(len(predictions), label_class)
        err_sum_snt = 0
        for idx, prediction in enumerate(predictions[::3]):
            for key, value in prediction.items():
                err_sum_snt += float((key != label[idx]))
        acc_snt = 1 - err_sum_snt / float(len(predictions) / 3)
        name = get_keys_by_value(names, int(label_class))
        msg = "{}".format(ids[name[0]])
        print(' ---', msg)
        connection.sendall(msg.encode())
        save_wav(total_frames, 'results/total_frames_{}.wav'.format(count))
        save_wav(total_frames, 'results/detection_frames_{}.wav'.format(count))
        count += 1
        connection.close()
    except:
        pass
