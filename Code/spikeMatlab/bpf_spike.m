function out = bpf_spike(in, fc_lo, fc_hi, fsSpikes)
arguments
    in
    fc_lo (1,1) {mustBePositive}
    fc_hi (1,1) {mustBePositive}
    fsSpikes (1,1) {mustBePositive}
end

H = designfilt('bandpassiir',...
       FilterOrder=4, ...
       HalfPowerFrequency1=fc_lo, ...
       HalfPowerFrequency2=fc_hi, ...
       SampleRate=fsSpikes, ...
       DesignMethod='butter');

out = filtfilt(H, in)*1000;

end
