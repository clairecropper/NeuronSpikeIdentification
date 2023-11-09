# Delete the fake oscillation spike 
import numpy as np

deleteEv = []

for i in range(len(spike_index)):
    before = range(spike_index(i)-50, spike_index(i)-20,1)
    after = range(spike_index(i)+20, spike_index(i)+50,1)
    
    #enumarate will return a tuple that contains index and value of each element 
    beforeFind = [index for index, val in enumerate(spikes[before]) if val < -10 or val >25]
    afterFind = [index for index, val in enumerate(spikes[after]) if val < -10 or val > 25]
    
    if len(beforeFind) > 1 or len(afterFind) > 1:
        deleteEv[end + 1] = i

for i in deleteEv:
    spike_index[i] = 0

spike_index = [element for element in spike_index if element != 0]


# Get the 3 ms spike cutout
per = 50
detected_spikes = spike_index
num_spikes = len(detected_spikes)
data = np.zeros((num_spikes,per))

for i in range(num_spikes):
    start[i] = detected_spikes[i]-per
    stop[i] = start[i] + (2*per)
    data[i][0:2*per+1] = spikes[start[i]:stop[i]+1]
    starttime[i] = start[i]
