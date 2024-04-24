function raw = select_signal(file, fsSpikes)
arguments
    file (1,1) string {mustBeFile}
    fsSpikes (1,1) {mustBeInteger,mustBePositive}
end

dat = load(file, "recording1");
recording1 = dat.recording1;

raw = recording1(20*fsSpikes:260*fsSpikes);

end
