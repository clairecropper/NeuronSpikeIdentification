function data = spikes_cutout(spikes, spike_index, per)
arguments
    spikes
    spike_index {mustBeInteger}
    per (1,1)
end

detected_spikes=spike_index;
num_spikes=length(detected_spikes);
data=zeros(num_spikes,per);

for i=1:num_spikes
    start = detected_spikes(i)-per;
    stop = start + (2*per);
    data(i,1:2*per+1) = spikes(start:stop);
    % starttime(i)=start(i);
end

end % function