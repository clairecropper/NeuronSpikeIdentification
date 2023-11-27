function neuron = plot_pca(data, neuron, coeff, desired_k, fsSpikes, per, ...
     firstcluster, secondcluster, doPlot)
arguments
    data
    neuron
    coeff
    desired_k (1,1) {mustBeInteger}
    fsSpikes (1,1) {mustBeInteger}
    per (1,1)
    firstcluster (1,1) {mustBeInteger}
    secondcluster (1,1) {mustBeInteger}
    doPlot (1,1) logical = true
end

if doPlot
figure
plot(0:(1/fsSpikes)*1e3:(2*per)*1e3/fsSpikes,data')
title('Spikes')
xlabel('Time(ms)')
ylabel('Voltage(mV)')
set(gca,'LineWidth',2,'FontSize',16,'Fontname','Arial Bold')
end

[IDX,C]=kmeans(data,desired_k,'Distance','cityblock','Display','final','Replicates',desired_k+6);
% [s,h] = silhouette(data,IDX);

if doPlot
figure
end

cluster=cell(desired_k,1);
color_cluster = {[0.259, 0.62, 0.741],[0 0 1],[1 0 1],[0.949, 0.498, 0.047],[1 1 0],[0 1 1],[0.5 0 1],[0 0.5 1],[1 0.5 0],[1 0 0.5]};
color_centroid = {'r*','gs','bo','kd'};
for i=1:desired_k
    cluster{i}=data(IDX==i,:);
    if doPlot
    plot(0:(1/fsSpikes)*1e3:(2*per)*1e3/fsSpikes,cluster{i}','Color',color_cluster{i})
    title('sorted neuron signals')
    xlabel('time (ms)');
    ylabel('voltage (mV)');
    set(gca,'LineWidth',2,'FontSize',16,'Fontname','SansSerif')
    hold on
    end
end

for i=1:desired_k
    cluster{i}=data(IDX==i,:);
   if doPlot
    figure
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
end

mu=mean(data);

marker_neuron = {'o','*','s','x','d','p','+','.','v','>'};
if doPlot
figure
end
for i=1:desired_k
    sizeofcluster=size(cluster{i});

    neuron{i}=(cluster{i}-repmat(mu,sizeofcluster(1),1))*coeff;

if doPlot
plot(neuron{i}(:,1),neuron{i}(:,2),marker_neuron{i},'Color',color_cluster{i})
% title('PCA analysis')
xlabel('PC1','FontSize',24,'LineWidth',5) % x-axis label
 ylabel('PC2','FontSize',24,'LineWidth',5) % y-axis label

set(gca,'LineWidth',2,'FontSize',16,'Fontname','Arial Bold')

hold on
end
end

if ~doPlot
    return
end

figure
scatter(1:length(IDX),IDX)

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

end % function