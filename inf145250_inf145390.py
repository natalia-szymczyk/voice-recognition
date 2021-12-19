import numpy
import scipy.signal as signal
import soundfile
import sys
import os

from matplotlib import pyplot as plt


def main():
    files = os.listdir("train")
    odp = 0

    for file in files:
        # filename = sys.argv[1]
        filename = f"train/{file}"

        data, rate = soundfile.read(f"{filename}")

        # max_value = numpy.max(data)
        # data = data.astype(float) / 2 ** 16

        data = [numpy.mean(value) for value in data]
        # data = [value * (max_value / 2) for value in data]


        lenght = len(data)
        offset = int(len(data) / 4)

        data = data[offset:(lenght - offset)]

        # processed = data * numpy.kaiser(len(data), 5.0) # Może hamming?
        processed = data * numpy.hamming(len(data))
        processed = numpy.abs(numpy.fft.rfft(processed))

        decimate2 = signal.decimate(processed, 2)
        decimate3 = signal.decimate(processed, 3)
        decimate4 = signal.decimate(processed, 4)

        lenght = len(decimate4)

        end_signal = processed[:lenght] * decimate2[:lenght] * decimate3[:lenght] * decimate4[:lenght]

        shift = 60

        result = (numpy.argmax(end_signal[shift:]) + shift) / (len(data) / rate)

        answer = 'M'
        if result > 165:
            answer = 'K'


        correct_answer = file[4]

        if answer == correct_answer:
            odp+=1

        print(file, correct_answer, answer, result)

    print(odp, odp/len(files))

if __name__ == "__main__":
    main()
