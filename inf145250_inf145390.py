import numpy
import soundfile
import sys

from matplotlib import pyplot as plt


def main():
    filename = sys.argv[1]

    data, rate = soundfile.read(f"{filename}")

    # s(n)
    signal = data[:, 0]

    signal = signal * numpy.hamming(len(signal))

    # x(n)
    signal = numpy.fft.fft(signal)

    # |x(w)|
    signal = numpy.abs(signal)

    # log(x(w))
    signal = numpy.log(signal)

    # c(n)
    signal = numpy.fft.ifft(signal)
    plt.plot(signal)
    plt.show()

    move = 60

    result = (numpy.argmax(signal[move:])) / (len(data) / rate)
    result2 = rate / numpy.argmax(signal[60 : -60])

    print(result, result2, numpy.argmax(signal[100 : -100]),  numpy.max(signal[100 : -100]))


if __name__ == "__main__":
    main()
