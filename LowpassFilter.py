import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


class LowpassFilter:
    def __init__(self, cutoff, fs, order=5):
        self.b, self.a = butter(order, cutoff, fs=fs, btype='low', analog=False)

    def butter_lowpass_filter(self, data):
        y = filtfilt(self.b, self.a, data, padlen=2)        # padlen=2 seems to provide the best results: it allows the signal to tend a little toward the final value
        return y


# Test the filter on some hypothetical data.
if __name__ == "__main__":
    # Filter requirements.
    order = 5
    fs = 10.0   # sample rate, Hz
    cutoff = 1  # desired cutoff frequency of the filter, Hz

    filter = LowpassFilter(cutoff, fs, order)

    # First make some data to be filtered.
    # data = [-30, 30, 20, 10, 0, 50, 0, 10, 20, 20, 30, 10, 0, 10, 10, 20, -30, 0, 20, 20]
    # data = [-30, 0, 20, 10, 0, 30, 30, 30, 30, 0, 0, 0, 0, 10, 10, 20, -30, 0, 20, 20]
    data = [-30, 0, 20, -10, 20, -30, 15, 30, 30, 0, 0, 0, 0, 10, 10, 20, -30, 0, 10, 10]
    data = np.radians(data)
    n = len(data)
    T = n / fs
    t = np.linspace(0, T, n, endpoint=False)

    # Filter the data, and plot both the original and filtered signals.
    y = filter.butter_lowpass_filter(data)

    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.show()
