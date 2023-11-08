import pyabf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


## Load the .abf file into Python

recording1 = pyabf.ABF("30 min_0001.abf")
fs = 50000 # Sampling frequency


## Bandpass Filter for picking up the LFP and Spike Signals

# Convert the unit from mV to uV and only 
# Use the first 60 seconds to save the processing time

RawData = 1000 * recording1.data[0, 5 * fs:65 * fs]

Time = np.arange(1/fs, 60 + 1/fs, 1/fs) # Construct the Time Vector

SpikeFilter = butter(4, [300, 5000], btype = 'bandpass', fs = fs)
LfpFilter = butter(2, [0.5, 300], btype = 'bandpass', fs = fs)

# Bandpass Filters for Spike and LFP Data
SpikeData = filtfilt(SpikeFilter[0], SpikeFilter[1], RawData)
LfpData = filtfilt(LfpFilter[0], LfpFilter[1], RawData)


## Plot the Raw, LFP and Spike Data 

plt.figure(figsize = (7, 5))

# Raw Data
plt.subplot(3, 1, 1)
plt.plot(Time, RawData, 'k', linewidth = 0.2)
plt.xlim(0, 60)
plt.ylim(-700, 500)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (µV)')
plt.title("Raw Signal", fontweight = "bold")

# LFP Data
plt.subplot(3, 1, 2)
plt.plot(Time, RawData, 'k', linewidth = 0.2)
plt.xlim(0, 60)
plt.ylim(-3000, 1000)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (µV)')
plt.title("LFP Signal", fontweight = "bold")

# Spike Data
plt.subplot(3, 1, 3)
plt.plot(Time, SpikeData, 'k', linewidth = 0.2)
plt.xlim(0, 60)
plt.ylim(-100, 100)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (µV)')
plt.title("Spike Signal", fontweight = "bold")

plt.tight_layout()
plt.show()
