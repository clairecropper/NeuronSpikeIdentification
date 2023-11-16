import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Load the mat data
mat_data = scipy.io.loadmat('10 min recording1.mat')
recording1 = mat_data['recording1'].flatten()

fsSpikes = 50000
rawsignal = recording1[20 * fsSpikes:260 * fsSpikes]

# Bandpass filter for Spikes and LFP
Fc1_spikes, Fc2_spikes = 300, 3000
b_spikes, a_spikes = butter(4, [Fc1_spikes, Fc2_spikes], btype='band', fs=fsSpikes)
spikes = filtfilt(b_spikes, a_spikes, rawsignal) * 1000

Fc1_lfp, Fc2_lfp = 0.5, 300
b_lfp, a_lfp = butter(2, [Fc1_lfp, Fc2_lfp], btype='band', fs=fsSpikes)
LFP = filtfilt(b_lfp, a_lfp, rawsignal) * 1000

# Plot
time = np.arange(0, len(spikes)) / fsSpikes

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time, rawsignal)
plt.title('Raw Signal')
plt.subplot(3, 1, 2)
plt.plot(time, spikes)
plt.title('Filtered Spikes')
plt.subplot(3, 1, 3)
plt.plot(time, LFP)
plt.title('Filtered LFP')

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

## Shi section

upper_threshold = -100
threshold = -20

ind_above = np.where(spikes < threshold)[0]
vector = np.zeros_like(spikes)
vector[ind_above] = spikes[ind_above]

above_threshold = ind_above
# 1.5 ms delay in sample numbers
delay = int(fsSpikes / 1000 * 1.5)  

end_spike_index = []
n = 1

for ii in range(len(above_threshold) - 1):
    if spikes[above_threshold[ii]] != spikes[above_threshold[ii + 1]] and \
       above_threshold[ii] < above_threshold[ii + 1] - delay:
        end_spike_index.append(above_threshold[ii])
        n += 1

spike_index = []
for ii in range(1, len(end_spike_index)):
    window = spikes[end_spike_index[ii] - delay:end_spike_index[ii] + delay]
    M = np.min(window)
    index = np.argmin(window)
    if M > upper_threshold:
        spike_index.append(end_spike_index[ii] - delay + index)

spike_index = np.array(spike_index)

# Delete the fake oscillation spike 
deleteEv = []

for i in range(len(spike_index)):
    before = range(spike_index[i] - 50, spike_index[i] - 20, 1)
    after = range(spike_index[i] + 20, spike_index[i] + 50, 1)
    
    beforeFind = [index for index, val in enumerate(spikes[before]) if val < -10 or val > 25]
    afterFind = [index for index, val in enumerate(spikes[after]) if val < -10 or val > 25]
    
    if len(beforeFind) > 1 or len(afterFind) > 1:
        deleteEv.append(i)

for i in deleteEv:
    spike_index[i] = 0

spike_index = [element for element in spike_index if element != 0]

# Get the 3 ms spike cutout
per = 50
detected_spikes = spike_index
num_spikes = len(detected_spikes)
data = np.zeros((num_spikes, 2 * per))

start = np.zeros(num_spikes, dtype=int)
stop = np.zeros(num_spikes, dtype=int)
starttime = np.zeros(num_spikes, dtype=int)

for i in range(num_spikes):
    start[i] = detected_spikes[i] - per
    stop[i] = start[i] + (2 * per)
    data[i][:] = spikes[start[i]:stop[i]]
    starttime[i] = start[i]


# plot data
time = np.arange(0, 2 * per, 1 / fsSpikes) * 1000  

plt.figure()
plt.plot(time[:data.shape[1]], data.T)
plt.title('Spikes')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.xlim(0, 2 * per * 1000 / fsSpikes)
plt.grid(True)
plt.show()

