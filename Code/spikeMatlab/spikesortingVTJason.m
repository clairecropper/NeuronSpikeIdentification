%% Load the recording data and select the part of interest
clear
close("all")
load('../spikesortingVTJason/10 min recording1.mat')
fsSpikes=50000;
rawsignal =recording1(20*fsSpikes:260*fsSpikes);

%% Bandpass filter for Spikes and LFP
Fc1=300;
Fc2=3000;
H1 = designfilt('bandpassiir','FilterOrder',4, ...
         'HalfPowerFrequency1',Fc1,'HalfPowerFrequency2',Fc2, ...
         'SampleRate',fsSpikes,'DesignMethod','butter');
spikes=filtfilt(H1,rawsignal)*1000;

Fc1=0.5;
Fc2=300;
H2 = designfilt('bandpassiir','FilterOrder',4, ...
         'HalfPowerFrequency1',Fc1,'HalfPowerFrequency2',Fc2, ...
         'SampleRate',fsSpikes,'DesignMethod','butter');
LFP=filtfilt(H2,rawsignal)*1000;

figure
ha(1)=subplot(3,1,1);
plot(0:1/fsSpikes:(length(spikes)-1)/fsSpikes,rawsignal);
ha(2)=subplot(3,1,2);
plot(0:1/fsSpikes:(length(spikes)-1)/fsSpikes,spikes);
ha(3)=subplot(3,1,3);
plot(0:1/fsSpikes:(length(spikes)-1)/fsSpikes,LFP); %plot the overall spikes
linkaxes([ha(1),ha(2),ha(3)], 'x');

%% Detect the spike according to the threshold
upperthreshold=-100;
threshold=-20;
ind_above=find(spikes<threshold); %find the index where the voltage is bigger than threshold
vector=0*spikes;
for ii=1:length(ind_above)
    vector(ind_above(ii))=spikes(ind_above(ii));
end
above_threshold=ind_above;
delay=fsSpikes/1000*1.5; %1.5ms in length
spike_index=[];
n=1;
spikenum=0;

% find out when the spikes happe n
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

%% Delete the fake oscillation spike
deleteEv = [];
for i = 1:length(spike_index)
    before = spike_index(i)-50:1:spike_index(i)-20;
    after = spike_index(i)+20:1:spike_index(i)+50;
    beforeFind = [find(spikes(before) < -10); find(spikes(before) > 25)];
    afterFind = [find(spikes(after)< -10); find(spikes(after) > 25)];
    if length(beforeFind) > 1 || length(afterFind) > 1
        deleteEv(end + 1) = i;
    end
end

for i = deleteEv
    spike_index(i) = 0;
end
spike_index = nonzeros(spike_index);

%% Get the 3 ms spike cutout
per=50;
detected_spikes=spike_index;
num_spikes=length(detected_spikes);
data=zeros(num_spikes,per);

for i=1:num_spikes
    start(i)=detected_spikes(i)-per;
    stop(i)=start(i)+(2*per);
    data(i,1:2*per+1)=spikes(start(i):stop(i));
    starttime(i)=start(i);
end

%% Do PCA analysis on the spike array
[coeff,score,ev]  = pca(data);
mu=mean(data);
pcadataready=data*coeff;
pcadata=pcadataready(:,1:3);

figure
plot(0:(1/fsSpikes)*1e3:(2*per)*1e3/fsSpikes,data')
title('Spikes')
xlabel('Time(ms)')
ylabel('Voltage(mV)')
set(gca,'LineWidth',2,'FontSize',16,'Fontname','Arial Bold')

desired_k=3;%%cluster number
[IDX,C]=kmeans(data,desired_k,'Distance','cityblock','Display','final','Replicates',desired_k+6);
% [s,h] = silhouette(data,IDX);

figure

cluster=cell(desired_k,1);
color_cluster = {[0.259, 0.62, 0.741],[0 0 1],[1 0 1],[0.949, 0.498, 0.047],[1 1 0],[0 1 1],[0.5 0 1],[0 0.5 1],[1 0.5 0],[1 0 0.5]};
color_centroid = {'r*','gs','bo','kd'};
for i=1:desired_k
    cluster{i}=data(IDX==i,:);
    plot(0:(1/fsSpikes)*1e3:(2*per)*1e3/fsSpikes,cluster{i}','Color',color_cluster{i})
    title('sorted neuron signals')
    xlabel('time (ms)');
    ylabel('voltage (mV)');
    set(gca,'LineWidth',2,'FontSize',16,'Fontname','SansSerif')
    hold on
end

for i=1:desired_k
   figure
    cluster{i}=data(IDX==i,:);
    plot(0:(1/fsSpikes)*1e3:(2*per)*1e3/fsSpikes,cluster{i}','Color',color_cluster{i})
    meanofdata=mean(cluster{i},1);
    hold on
    plot(0:(1/fsSpikes)*1e3:(2*per)*1e3/fsSpikes, meanofdata,'k','LineWidth',1.5)
%     title(['sorted neuron signals cluster=' num2str(i)])
    xlabel('Time (ms)');
    ylabel('Voltage (\muV)');
    ylim([-70 40]);
    set(gca,'LineWidth',1.5,'FontSize',20,'Fontname','SansSerif')

end

[coeff,score,ev]  = pca(data);

neuron=cell(desired_k,1);
mu=mean(data);
marker_neuron = {'o','*','s','x','d','p','+','.','v','>'};
figure
for i=1:desired_k
    sizeofcluster=size(cluster{i});

    neuron{i}=(cluster{i}-repmat(mu,sizeofcluster(1),1))*coeff;

plot(neuron{i}(:,1),neuron{i}(:,2),marker_neuron{i},'Color',color_cluster{i})
% title('PCA analysis')
xlabel('PC1','FontSize',24,'LineWidth',5) % x-axis label
 ylabel('PC2','FontSize',24,'LineWidth',5) % y-axis label

set(gca,'LineWidth',2,'FontSize',16,'Fontname','Arial Bold')

hold on
end

figure
scatter(1:length(IDX),IDX)

firstcluster=input('please enter the first wanted cluster:');
secondcluster=input('please enter the second wanted cluster.:');

figure
hb(1)=subplot(2,1,1);
plot(0:1/fsSpikes:(length(rawsignal)-1)/fsSpikes,rawsignal,'LineWidth',1.3)
hold on
i=firstcluster;
    timecluster{i}=detected_spikes(IDX==i)+30;
    scatter(timecluster{i}/fsSpikes,-900*ones(1,length(timecluster{i})),5,'^','MarkerFaceColor',color_cluster{i},'MarkerEdgeColor',color_cluster{i})
    hold on
    i=secondcluster;
    timecluster{i}=detected_spikes(IDX==i)+30;
    scatter(timecluster{i}/fsSpikes,-900*ones(1,length(timecluster{i})),5,'^','MarkerFaceColor',color_cluster{i},'MarkerEdgeColor',color_cluster{i})

title('Spike Trace')
xlabel('Time(s)')
ylabel('Voltage(\muV)')
set(gca,'LineWidth',2,'FontSize',16,'Fontname','SansSerif')

hb(2)=subplot(2,1,2);
plot(0:1/fsSpikes:(length(rawsignal)-1)/fsSpikes,spikes,'LineWidth',1.5)
set(gca,'FontSize',16, 'FontName','SansSerif', 'Linewidth', 1.5);
hold on
i=firstcluster;
timecluster{i}=detected_spikes(IDX==i);
scatter(timecluster{i}/fsSpikes,-60*ones(1,length(timecluster{i})),5,'^','MarkerFaceColor',color_cluster{i},'MarkerEdgeColor',color_cluster{i})
hold on
i=secondcluster;
timecluster{i}=detected_spikes(IDX==i);
scatter(timecluster{i}/fsSpikes,-60*ones(1,length(timecluster{i})),5,'^','MarkerFaceColor',color_cluster{i},'MarkerEdgeColor',color_cluster{i})

linkaxes([hb(1),hb(2)], 'x');

figure
time = 1/fsSpikes:1/fsSpikes:60;
trace = spikes(1:60*fsSpikes);
length(time)
length(trace)
plot(time,trace,'LineWidth',1.5)
set(gca,'FontSize',16, 'FontName','SansSerif', 'Linewidth', 1.5);
hold on
timecluster{i}=detected_spikes(IDX==firstcluster);
scatter(timecluster{i}/fsSpikes,-60*ones(1,length(timecluster{i})),5,'^','MarkerFaceColor',color_cluster{1},'MarkerEdgeColor',color_cluster{1})
hold on
i=secondcluster;
timecluster{i}=detected_spikes(IDX==i);
scatter(timecluster{i}/fsSpikes,-60*ones(1,length(timecluster{i})),5,'^','MarkerFaceColor',color_cluster{4},'MarkerEdgeColor',color_cluster{4})

figure
i=firstcluster;
plot(neuron{i}(:,1),neuron{i}(:,2),marker_neuron{i},'Color',[0.259, 0.62, 0.741], "Marker", "o")
hold on
i=secondcluster;
plot(neuron{i}(:,1),neuron{i}(:,2),marker_neuron{i},'Color',[0.949, 0.498, 0.047], 'Marker', "square")
xlabel('PC1','FontSize',24,'LineWidth',5) % x-axis label
ylabel('PC2','FontSize',24,'LineWidth',5) % y-axis label
set(gca,'FontSize',16, 'FontName','SansSerif', 'Linewidth', 1.5)

figure
hc(1)=subplot(3,1,1);
plot(0:1/fsSpikes:(length(rawsignal)-1)/fsSpikes,rawsignal,'LineWidth',1.5)
hold on
i=firstcluster;
timecluster{i}=detected_spikes(IDX==i)+30;
scatter(timecluster{i}/fsSpikes,-800*ones(1,length(timecluster{i})),25,'^','MarkerFaceColor',color_cluster{i},'MarkerEdgeColor',color_cluster{i})
hold on
i=secondcluster;
timecluster{i}=detected_spikes(IDX==i)+30;
scatter(timecluster{i}/fsSpikes,-800*ones(1,length(timecluster{i})),25,'^','MarkerFaceColor',color_cluster{i},'MarkerEdgeColor',color_cluster{i})
xlim([39.8,40.2]);
ylim([-800,200]);
%     axis off
% xlabel('Time(s)')
% ylabel('Voltage(\muV)')
% set(gca,'LineWidth',2,'FontSize',16,'Fontname','Arial Bold')
hc(2)=subplot(3,1,2);
plot(0:1/fsSpikes:(length(spikes)-1)/fsSpikes,LFP,'LineWidth',1.5);
ylim([-600,400]);
% axis off
hc(3)=subplot(3,1,3);
plot(0:1/fsSpikes:(length(rawsignal)-1)/fsSpikes,spikes,'LineWidth',1.5)
hold on
i=firstcluster;
timecluster{i}=detected_spikes(IDX==i);
scatter(timecluster{i}/fsSpikes,-100*ones(1,length(timecluster{i})),25,'^','MarkerFaceColor',color_cluster{i},'MarkerEdgeColor',color_cluster{i})
hold on
i=secondcluster;
timecluster{i}=detected_spikes(IDX==i);
scatter(timecluster{i}/fsSpikes,-100*ones(1,length(timecluster{i})),25,'^','MarkerFaceColor',color_cluster{i},'MarkerEdgeColor',color_cluster{i})

linkaxes([hc(1),hc(2),hc(3)], 'x');
xlim([39.8,40.2]);
% axis off
clusterone=neuron{firstcluster}(:,1:3);
clustertwo=neuron{secondcluster}(:,1:3);
d2 = mahal(clustertwo,clusterone);
MD=sort(d2);
ID=MD(min(length(clusterone(:,1)),length(clustertwo(:,2))));
p = chi2cdf(MD,3);
Lratio=sum(1.-p)/length(clusterone(:,1));
