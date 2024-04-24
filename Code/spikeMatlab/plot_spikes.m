function plot_spikes(rawsignal, spikes, LFP, fsSpikes)

figure
t = tiledlayout("vertical");
ax1 = nexttile(t);
plot(ax1, 0:1/fsSpikes:(length(spikes)-1)/fsSpikes, rawsignal);
ax2 = nexttile(t);
plot(ax2, 0:1/fsSpikes:(length(spikes)-1)/fsSpikes, spikes);
ax3 = nexttile(t);
plot(ax3, 0:1/fsSpikes:(length(spikes)-1)/fsSpikes, LFP);

linkaxes([ax1, ax2, ax3], "x")

end