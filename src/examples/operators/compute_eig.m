%% Compute eigenvalues from FlowSovler matrices
clearvars;

CYLINDER = false;
CAVITY = false;
LIDCAVITY = true;

%% Cylinder
if CYLINDER
    folder = "cylinder";
    Amat = read_coo(fullfile(folder, "A_coo.mat"));
    Emat = read_coo(fullfile(folder, "E_coo.mat"));
    
    k = 1;
    
    D = [];
    targets = [0.1 + 0.8j, 0];
    for ii=1:length(targets)
        sigma = targets(ii);
        [~, d] = eigs(Amat, Emat, k, sigma, "Display", true);
        D = [D; diag(d)];
    end
    D = unique(D);
    
    disp(D)

    figure(1);
    clf; hold on;
    plot(D, "r.", "MarkerSize", 12)
    plot(conj(D), "b.", "MarkerSize", 12)
    xline(0, "k--")
    yline(0, "k--")
    grid on;
    xlabel("\Re")
    ylabel("\Im")
    title("Cylinder eigenvalues")
end

%% Cavity
if CAVITY
    folder = "cavity";
    Amat = read_coo(fullfile(folder, "A_coo.mat"));
    Emat = read_coo(fullfile(folder, "E_coo.mat"));
    
    k = 1;
    
    D = [];
    targets = [0, 0.8 + 10j, 0.7 + 13j,... 
        0.4 + 7j, 16j];
    for ii=1:length(targets)
        sigma = targets(ii);
        [~, d] = eigs(Amat, Emat, k, sigma, "Display", true);
        D = [D; diag(d)];
    end
    D = unique(D);
    
    disp(D)

    figure(2);
    clf; hold on;
    plot(D, "r.", "MarkerSize", 12)
    plot(conj(D), "b.", "MarkerSize", 12)
    xline(0, "k--")
    yline(0, "k--")
    grid on;
    xlabel("\Re")
    ylabel("\Im")
    title("Cavity eigenvalues")
end

%% Lid-driven cavity
if LIDCAVITY
    folder = "lidcavity";
    Amat = read_coo(fullfile(folder, "A_coo.mat"));
    Emat = read_coo(fullfile(folder, "E_coo.mat"));
    
    k = 10;
    
    D = [];
    targets = [0, 1j, 2j, -0.5, 3j, 4j, 5j, 6j];
    for ii=1:length(targets)
        sigma = targets(ii);
        [~, d] = eigs(Amat, Emat, k, sigma, "Display", true);
        D = [D; diag(d)];
    end
    D = unique(D);
    
    disp(D)

    figure(3);
    clf; hold on;
    plot(D, "r.", "MarkerSize", 12)
    plot(conj(D), "b.", "MarkerSize", 12)
    xline(0, "k--")
    yline(0, "k--")
    grid on;
    xlabel("\Re")
    ylabel("\Im")
    title("Lid-driven cavity eigenvalues")
end