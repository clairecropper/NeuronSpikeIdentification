import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
# plt.show()

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

#pca analysis

# plot data
time = np.arange(0, 2 * per, 1 / fsSpikes) * 1000  

plt.figure()
plt.plot(time[:data.shape[1]], data.T)
plt.title('Spikes')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.xlim(0, 2 * per * 1000 / fsSpikes)
plt.grid(True)

# line 112 
desired_k = 3

# Perform k-means clustering
kmeans = KMeans(n_clusters=desired_k, init='k-means++', n_init=desired_k + 6, random_state=42)
kmeans.fit(data)
IDX = kmeans.labels_
C = kmeans.cluster_centers_

# Define colors for each cluster
color_cluster = [
    [0.259, 0.62, 0.741],
    [0, 0, 1],
    [1, 0, 1],
    [0.949, 0.498, 0.047],
    [1, 1, 0],
    [0, 1, 1],
    [0.5, 0, 1],
    [0, 0.5, 1],
    [1, 0.5, 0],
    [1, 0, 0.5]
]

# Plot each cluster
plt.figure()
for i in range(desired_k):
    # Select data for the current cluster
    cluster_data = data[IDX == i, :]
    # Time vector for plotting
    time = np.linspace(0, (2 * per) * 1e3 / fsSpikes, cluster_data.shape[1])
    # Plot each waveform in the cluster
    for waveform in cluster_data:
        plt.plot(time, waveform, color=color_cluster[i])

plt.title('Sorted Neuron Signals')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')

# line 130
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

for i in range(desired_k):
    plt.figure()
    # Select data for the current cluster
    cluster_data = data[IDX == i, :]
    # Time vector for plotting
    time = np.linspace(0, (2 * per) * 1e3 / fsSpikes, cluster_data.shape[1])
    # Plot each waveform in the cluster
    for waveform in cluster_data:
        plt.plot(time, waveform, color=color_cluster[i])
    
    # Calculate the mean waveform for the current cluster
    meanofdata = np.mean(cluster_data, axis=0)
    # Plot the mean waveform
    plt.plot(time, meanofdata, 'k', linewidth=1.5)  # 'k' is for black color
    
    # plt.title(f'Sorted neuron signals cluster={i+1}')  # i+1 to convert from 0-based to 1-based indexing
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (µV)')
    plt.ylim([-70, 40])


plt.show()
# Perform PCA on the entire dataset
pca = PCA(n_components=3)  # Adjust the number of components as necessary
pca.fit(data)
coeff = pca.components_
score = pca.transform(data)
ev = pca.explained_variance_

