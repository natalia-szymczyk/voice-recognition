import soundfile
import numpy as np
import sys
from scipy.signal import decimate


def main():
    file = sys.argv[1]

    data, rate = soundfile.read(file)
    y = [np.mean(value) for value in data]

    x = np.fft.fftfreq(len(data), 1 / rate)

    y = y * np.hamming(len(y))
    y = np.fft.fft(y)
    y = abs(y)

    hps_y = y.copy()
    for k in range(2, 5):
        tmp_d = decimate(y, k)
        hps_y[:len(tmp_d)] *= tmp_d

    hps_y[:10] = 0

    male = [80, 173]
    female = [173, 350]
    mask = (male[0] <= x) & (x <= female[1])

    result = x[mask][np.argmax(hps_y[mask])]

    #
    # print(mask.shape)
    # print(hps_y.shape)

    # mask2 = np.ndarray(hps_y.shape, dtype=bool)
    #
    # for i, value in enumerate(hps_y):
    #     if i < male[0] or i > female[1]:
    #         mask2[i] = False
    #     else:
    #         mask2[i] = True

    # print(len(mask2))
    # print(mask == mask2)
    # result = x[np.argmax(voice)]

    # print(hps_y[mask])
    # print(voice)

    answer = 'M'
    if female[0] <= result <= female[1]:
        answer = 'K'

    correct_answer = file[-5]

    print(answer)


if __name__ == '__main__':
    try:
        main()
    except:
        print('K')
