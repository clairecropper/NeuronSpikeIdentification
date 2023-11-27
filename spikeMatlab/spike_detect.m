function spike_index = spike_detect(spikes, fsSpikes)

upperthreshold=-100;
threshold=-20;

% voltage is bigger than threshold
i = find(spikes < threshold);

above_threshold = i;
delay = fsSpikes/1000*1.5; %1.5ms in length
n=1;

% find out when the spikes happen
for ii=1:length(above_threshold)-1 % for all time stamp that has the voltage bigger than threshold
    if (spikes(above_threshold(ii))~=spikes(above_threshold(ii)+1)) && (above_threshold(ii)<above_threshold(ii+1)-delay)
        %the voltage is at its maximum and no spikes for at least 1.5 ms
            end_spike_index(n)=above_threshold(ii);
            n=n+1;
    end
end

spike_index=[];
for ii=2:length(end_spike_index)
        [M,index]=min(spikes(end_spike_index(ii)-delay:end_spike_index(ii)+delay));
        if M>upperthreshold
            spike_index(end+1)=end_spike_index(ii)-delay+1+index;
        end

end

end % function