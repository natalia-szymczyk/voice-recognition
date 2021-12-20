import copy
import numpy
import scipy.signal as signal
import soundfile
import sys
import os

from matplotlib import pyplot as plt
from matplotlib.pyplot import stem


def main():
    files = os.listdir("train")
    odp = 0

    k_k = 0
    k_m = 0
    m_m = 0
    m_k = 0

    for file in files:
        # filename = sys.argv[1]
        filename = f"train/{file}"

        data, rate = soundfile.read(f"{filename}")

        data = [numpy.mean(value) for value in data]

        x = numpy.fft.fftfreq(len(data), 1 / rate)

        y = data * numpy.kaiser(len(data), beta = 50)
        y = numpy.fft.fft(y)
        y = numpy.abs(y)

        tmp_y = y.copy()

        for i in range(2, 5):
            tmp_d = signal.decimate(y, i)
            tmp_y[:len(tmp_d)] *= tmp_d

        # plt.plot(tmp_y)
        # plt.show()

        m0, m1 = 85, 175  # typically [85,180]
        f0, f1 = 175, 355  # typically [165,255]
        mask = (m0 <= x) & (x <= f1)
        result = x[mask][numpy.argmax(y[mask])]
        answer = ''

        if m0 <= result < m1:
            answer = 'M'
        elif f0 <= result < f1:
            answer = 'K'


        correct_answer = file[4]

        if answer == correct_answer:
            odp+=1

        if correct_answer == "K" and answer == "K":
            k_k += 1
        elif correct_answer == "K" and answer == "M":
            k_m += 1
        elif correct_answer == "M" and answer == "M":
            m_m += 1
        elif correct_answer == "M" and answer == "K":
            m_k += 1

        print(file, correct_answer, answer, result)

    print(f"Skuteczność = {odp}/{len(files)} = {odp/len(files)}, k_k: {k_k}, k_m: {k_m}, m_m: {m_m}, m_k: {m_k} ")

if __name__ == "__main__":
    main()
