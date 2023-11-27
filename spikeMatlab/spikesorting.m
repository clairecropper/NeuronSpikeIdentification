%% Load the recording data and select the part of interest
function neuron = spikesorting(file, firstcluster, secondcluster, doPlot)
arguments
    file (1,1) string {mustBeFile}
    firstcluster (1,1) {mustBeInteger}
    secondcluster (1,1) {mustBeInteger}
    doPlot (1,1) logical = true
end

dat = load(file, "recording1");
recording1 = dat.recording1;
clear('dat')

fsSpikes=50000;
per=50;
rawsignal = recording1(20*fsSpikes:260*fsSpikes);

%% Bandpass filter (BPF) for Spikes and LFP

% midband BPF
spikes = bpf_spike(rawsignal, 300, 3000, fsSpikes);

% lowband BPF
LFP = bpf_spike(rawsignal, 0.5, 300, fsSpikes);

%plot the overall spikes
if(doPlot)
plot_spikes(rawsignal, spikes, LFP, fsSpikes)
end

%% Detect the spike according to the threshold
spike_index = spike_detect(spikes, fsSpikes);

%% Delete the fake oscillation spike
spike_index = censor_spikes(spikes, spike_index);

%% Get the 3 ms spike cutout
data = spikes_cutout(spikes, spike_index, per);

%% Do PCA analysis on the spike array
[coeff,score,ev]  = pca(data);
mu=mean(data);
pcadataready=data*coeff;
pcadata=pcadataready(:,1:3);

[coeff,score,ev]  = pca(data);


desired_k=3;
% cluster number

neuron = cell(desired_k,1);

neuron = plot_pca(data, neuron, coeff, ...
    desired_k, fsSpikes, per, ...
    firstcluster, secondcluster, doPlot);

% axis off
clusterone=neuron{firstcluster}(:,1:3);
clustertwo=neuron{secondcluster}(:,1:3);
d2 = mahal(clustertwo,clusterone);
MD=sort(d2);
ID=MD(min(length(clusterone(:,1)),length(clustertwo(:,2))));
p = chi2cdf(MD,3);
Lratio=sum(1.-p)/length(clusterone(:,1));

end %function
