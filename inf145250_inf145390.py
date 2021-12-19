import numpy
import soundfile
import sys

from matplotlib import pyplot as plt


def main():
    filename = sys.argv[1]

    data, rate = soundfile.read(f"{filename}")

    # s(n)
    signal = data[:, 0]
    plt.plot(signal)
    plt.show()

    signal = signal * numpy.hamming(len(signal))
    plt.plot(signal)
    plt.show()

    # x(n)
    signal = numpy.fft.fft(signal)
    plt.plot(signal)
    plt.show()

    # |x(w)|
    signal = numpy.abs(signal)
    plt.plot(signal)
    plt.show()

    # log(x(w))
    signal = numpy.log(signal)
    plt.plot(signal)
    plt.show()

    # c(n)
    signal = numpy.fft.ifft(signal)
    plt.plot(signal)
    plt.show()


if __name__ == "__main__":
    main()
