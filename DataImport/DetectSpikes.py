import numpy as np

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
    if spikes[above_threshold[ii]] != spikes[above_threshold[ii] + 1] and \
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