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
        hps_y[:len(decimate(y, k))] *= decimate(y, k)

    hps_y[:10] = 0

    male = [80, 173]
    female = [173, 350]
    mask = (male[0] <= x) & (x <= female[1])

    result = x[mask][np.argmax(hps_y[mask])]

    answer = 'M'
    if female[0] <= result <= female[1]:
        answer = 'K'

    correct_answer = file[-5]

    print(answer, result)


if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        print('K')
