import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load HDF5 data
with h5py.File('Sample_spikeTimeStamp.h5', 'r') as file:
    # Assuming the structure of the HDF5 file is similar to that in MATLAB
    data = file['Recording']['TimeStampStream']['TimeStamps']

# Initialize matrices and variables
spikemat = np.zeros((256, 1506000))
V1 = np.zeros(256)
V2 = np.zeros(256)
T1 = 4 * 1000000  # Time range in microseconds
T2 = 7 * 1000000  # Time range in microseconds

# Process data
for j in range(256):
    time1 = data[j]
    if time1.size == 0:
        V1[j] = 0
        V2[j] = 0
    else:
        count_spike = 0
        for k in range(time1.size):
            timess = time1[k]
            if T1 < timess < T2:
                count_spike += 1
        V1[j] = count_spike
        V2[j] = count_spike / 3  # Frequency of the spikes

# Reshape and create heatmap
V3 = V2.reshape(16, 16)
x_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
y_labels = [str(i) for i in range(1, 17)]

plt.figure(figsize=(10, 8))
sns.heatmap(V3, xticklabels=x_labels, yticklabels=y_labels, annot=True)
plt.title(f'Time period: {T1/1000000} ~ {T2/1000000} seconds')
plt.xlabel('Channels')
plt.ylabel('Channels')
plt.show()
