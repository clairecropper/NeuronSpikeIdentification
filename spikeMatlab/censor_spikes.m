function spike_index = censor_spikes(spikes, spike_index)
arguments
    spikes
    spike_index {mustBeInteger,mustBePositive}
end


for i = 1:length(spike_index)
    before = spike_index(i)-50:1:spike_index(i)-20;
    after = spike_index(i)+20:1:spike_index(i)+50;
    beforeFind = [find(spikes(before) < -10); find(spikes(before) > 25)];
    afterFind = [find(spikes(after)< -10); find(spikes(after) > 25)];
    if length(beforeFind) > 1 || length(afterFind) > 1
        spike_index(i) = 0;
    end
end

spike_index = nonzeros(spike_index);

end % function