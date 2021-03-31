% IEM_ori_MEG.m
%
% Tommy Sprague, 3/16/2015, originally for Bernstein Center IEM workshop
% tommy.sprague@gmail.com or jserences@ucsd.edu
%
% Uses a dataset graciously donated by Edward Ester, results from which are 
% in preparation for submission (edward.ester01@gmail.com).
%
% Please DO NOT use data from these tutorials without permission. Feel free
% to adapt or use code for teaching, research, etc. If possible, please
% acknowledge relevant publications introducing different components of these
% methods (Brouwer & Heeger, 2009; 2011; Sprague & Serences, 2013; Garcia
% et al, 2013; Sprague, Saproo & Serences, 2015)
%
% Code adapted by Henry Allen

close all;

root = load_root();%'/usr/local/serenceslab/tommy/berlin_workshop/';
addpath([root 'mFiles/']);

subjlist = ["AK", "DI", "HHy", "HN", "JL", "KA", "MF", "NN", "SoM", "TE", "VA", "YMi"];
shuffleLabels = true;


% number of orientation channels
n_ori_chans = 9;
n_bins = 17;
ts = linspace(0, 0.375, 16);
chan_center = linspace(180/n_ori_chans,180,n_ori_chans);
all_max_resp = zeros(n_ori_chans);
all_mean_resp = zeros(n_ori_chans);
all_shift_coeffs = zeros(length(ts), length(chan_center));
sd_bins = zeros(n_bins, n_ori_chans);
sd_count = zeros(1, n_bins);


% main loop
for ff = 1:length(subjlist)
    % Load Data
    %
    % Each data file contains:
    % - trn:  data used for encoding model estimation (contrast detection task)
    %         n_trials x n_electrodes x n_timepoints wavelet coefficients
    %         (complex)
    % - trng: orientation label (0-160 deg) for each training trial
    
    subjname = subjlist(ff);
    
    
    filelocation = convertStringsToChars("MEG_ori/" + subjname + "_epochs2.mat");
    fprintf(filelocation);
    load([root filelocation]);
    
    avg_meg_response = mean(trn, [1, 2]);
    figure;
    plot(avg_meg_response(:));
    title("Average MEG Response");
    xlabel("Timestep");
    ylabel("Response");
    saveas(gcf, "../Figures/IEM/meg_response/" + subjname + "_meg.png");
    %trng = trng * 20;
    ts = linspace(0, 0.375, 16);
    if shuffleLabels
        subjname = subjname + "_shuffle";
        trng = trng(randperm(500));
    end
    trng_padded = [trng, trng(500)] / 20;
    diffs = -diff(trng_padded);
    diffs = diffs + 9;

    % generate orientation channels (orientation filters)
    % each orientation channel can be modeled as a "steerable filter" (see
    % Freeman and Adelson, 1991), which here is a (co)sinusoid raised to a high
    % power (specifically, n_ori_chans-mod(n_ori_chans,2)).
    make_basis_function = @(xx,mu) (cosd(xx-mu)).^(n_ori_chans-mod(n_ori_chans,2));

    xx = linspace(1,180,180);
    basis_set = nan(180,n_ori_chans);


    for cc = 1:n_ori_chans
        basis_set(:,cc) = make_basis_function(xx,chan_center(cc));
    end

    % Build stimulus mask
    trng = mod(trng,180);
    trng(trng==0) = 180;

    stim_mask = zeros(length(trng),length(xx));

    for tt = 1:size(stim_mask,1)  % loop over trials
        stim_mask(tt,trng(tt))=1;

    end

    % Generate design matrix
    trnX = stim_mask*basis_set;

    % We need to be sure our design matrix is full rank 
    fprintf('Design matrix rank = %i\n',rank(trnX));


    % let's look at the predicted response for a single trial
    tr_num = 6;
    figure;
    subplot(1,3,1);hold on;
    plot(xx,basis_set);
    plot(xx,stim_mask(tr_num,:),'k-');
    xlabel('Orientation (\circ)');title(sprintf('Stimulus, trial %i',tr_num));
    xlim([0 180]);
    subplot(1,3,2);hold on;
    plot(chan_center,trnX(tr_num,:),'k-');
    for cc = 1:n_ori_chans
        plot(chan_center(cc),trnX(tr_num,cc),'o','MarkerSize',8,'LineWidth',3);
    end
    xlabel('Channel center (\circ)');title('Predicted channel response');
    xlim([-5 180]);

    % and the design matrix
    subplot(1,3,3);hold on;
    imagesc(chan_center,1:size(trnX,1),trnX);
    title('Design matrix');
    xlabel('Channel center (\circ)');ylabel('Trial'); axis tight ij;
    saveas(gcf, "../Figures/IEM/single_response/" + subjname + "_ind_chan_resp.png")

    % Cross-validated training/testing on contrast discrimination task
    trn_ou = unique(trng);
    trn_repnum = nan(size(trng));
    n_trials_per_orientation = nan(length(trn_ou),1);
    for ii = 1:length(trn_ou)
        thisidx = trng==trn_ou(ii);
        trn_repnum(thisidx) = 1:(sum(thisidx));
        n_trials_per_orientation(ii) = sum(thisidx);
        clear thisidx;
    end
    
    trn_repnum(trn_repnum>min(n_trials_per_orientation))=NaN;
    trng_cv = trng(~isnan(trn_repnum));
    trn_cv = trn(~isnan(trn_repnum), :, :);
    diffs_cv = diffs(~isnan(trn_repnum));

    % cull all those trials from trn data (renamed trn_cv here for convenience)
    trn_cv_coeffs = nan(sum(~isnan(trn_repnum)),2*size(trn,2),size(trn,3));
    trn_cv_coeffs(:,1:size(trn,2),:) = real(trn(~isnan(trn_repnum),:,:)); 
    trn_cv_coeffs(:,(size(trn,2)+1):(2*size(trn,2)),:) = imag(trn(~isnan(trn_repnum),:,:)); 
    trnX_cv = trnX(~isnan(trn_repnum),:);
    trn_repnum = trn_repnum(~isnan(trn_repnum));

    chan_resp_cv_coeffs = nan(size(trn_cv_coeffs,1),length(chan_center),size(trn_cv_coeffs,3));

    % Cross-validation: we'll do "leave-one-group-of-trials-out" cross
    % validation by holding repnum==ii out at testset on each loop iteration

    n_reps = max(trn_repnum(:));
    tic
    for ii = 1:n_reps
        trnidx = trn_repnum~=ii;
        tstidx = trn_repnum==ii;

        thistrn = trn_cv_coeffs(trnidx,:,:);
        thistst = trn_cv_coeffs(tstidx,:,:);

        % loop over timepoints
        for tt = 1:size(thistrn,3)

            thistrn_tpt = thistrn(:,:,tt);
            thistst_tpt = thistst(:,:,tt);

            w_coeffs = trnX_cv(trnidx,:)\thistrn_tpt;

            %chan_resp_cv_coeffs(tstidx,:,tt) = (inv(w_coeffs*w_coeffs.')*w_coeffs*thistst_tpt.').'; % can also be written (w_coeffs.'\thistst_tpt.').';
            chan_resp_cv_coeffs(tstidx,:,tt) = (w_coeffs.'\thistst_tpt.').';
        end
    end
    toc

    trange = [0.1 .3];
    tidx = ts >= trange(1) & ts <= trange(2); % only look at this 3.5 s window

    % plot data for cross-validated, each orientation response plotted
    % separately

    figure; ax = [];
    for ii = 1:length(trn_ou)
        ax(end+1) = subplot(2,length(trn_ou),ii); hold on;

        thisidx = trng_cv==trn_ou(ii);
        imagesc(chan_center,ts,squeeze(mean(chan_resp_cv_coeffs(thisidx,:,:),1))');

        plot([trn_ou(ii) trn_ou(ii)],[ts(1) ts(end)],'k--');
        axis ij tight;

        title(sprintf('Ori: %i deg',trn_ou(ii)));
        if ii == 1
            ylabel('Time (s)');
            xlabel('Orientation channel (\circ)');
        end
        set(gca,'XTick',[0:45:180]);

    end
    match_clim(ax);

    % plot average from 0.0-0.4 s
    ax = [];
    tmean = mean(chan_resp_cv_coeffs(:,:,tidx),3);
    for ii = 1:length(trn_ou)
        ax(end+1) = subplot(2,length(trn_ou),ii+length(trn_ou)); hold on;

        thisidx = trng_cv==trn_ou(ii);

        plot(chan_center,mean(tmean(thisidx,:),1));

        title(sprintf('Ori: %i deg',trn_ou(ii)));
        if ii == 1
            ylabel('Channel response');
            xlabel('Orientation channel (\circ)');
        end
        set(gca,'XTick',[0:45:180]);

    end
    match_ylim(ax);
    saveas(gcf, "../Figures/IEM/ind_response/" + subjname + "_reconstructions.png")

    % Rotate all trials to center orientation and plot

    targ_ori = chan_center(round(length(chan_center)/2));
    targ_ori_idx = find(chan_center==targ_ori);

    chan_resp_cv_coeffs_shift = nan(size(chan_resp_cv_coeffs));
    for ii = 1:length(trn_ou)
        thisidx = trng_cv==trn_ou(ii);

        chan_resp_cv_coeffs_shift(thisidx,:,:) = circshift(chan_resp_cv_coeffs(thisidx,:,:), targ_ori_idx-find(trn_ou(ii)==chan_center) , 2 );
    end
    
    
    
    figure;
    subplot(1,3,1);hold on;
    curr_shift_coeffs = squeeze(mean(chan_resp_cv_coeffs_shift,1)).';
    all_shift_coeffs = all_shift_coeffs + curr_shift_coeffs;
    imagesc(chan_center,ts, curr_shift_coeffs);
    caxis([0.1, 0.4]);
    plot(targ_ori*[1 1],[ts(1) ts(end)],'k--');
    xlabel('Orientation channel (\circ)');
    ylabel('Time (s)');
    title('Coregistered channel response function timecourse');
    axis ij tight;

    subplot(1,3,2);hold on;
    tmean = mean(chan_resp_cv_coeffs_shift(:, :, :), 3);
    curr_mean_resp = mean(tmean,1);
    all_mean_resp = all_mean_resp + curr_mean_resp;
    plot(chan_center, curr_mean_resp);
    xlabel('Orientation channel (\circ)');
    ylabel('Reconstructed channel response');
    title(sprintf('Average (%0.01f-%0.01f s)',trange(1),trange(2)));
    axis([0 200 0 0.4]);

    
    subplot(1,3,3);
    average_resp = mean(trn, [1, 2]);
    average_resp(1:6) = -inf;
    average_resp(14:16) = -inf;
    [val, max_val] = max(average_resp(:));
    disp(max_val);
    curr_max_resp = mean(chan_resp_cv_coeffs_shift(:, :, max_val), 1);
    %curr_max_resp = mean(chan_resp_cv_coeffs_shift(:, :, max_val-1:max_val), 3);
    %curr_max_resp = mean(curr_max_resp, 1);
    
    all_max_resp = all_max_resp + curr_max_resp;
    
    
    plot(chan_center,curr_max_resp);
    xlabel('Orientation channel (\circ)');
    ylabel('Reconstructed channel response');
    title(sprintf('Max Timestep'));
    axis([0 200 0 0.4]);
    saveas(gcf, "../Figures/IEM/avg_response/" + subjname + "_tot_chan_response.png");
    for ii=1:length(diffs_cv)
        sd_bins(diffs_cv(ii), :) = sd_bins(diffs_cv(ii), :) + curr_max_resp;
        sd_count(diffs_cv(ii)) = sd_count(diffs_cv(ii)) + 1;
    end
end

figure;
subplot(1,3,1);
imagesc(chan_center, ts, all_shift_coeffs / length(subjlist));
caxis([0.2, 0.3]);

subplot(1,3,2);
plot(chan_center, all_mean_resp.' / length(subjlist));
axis([0 200 0 0.4]);
title("Mean Overall Channel Response");

subplot(1,3,3);
plot(chan_center, all_max_resp.' / length(subjlist));
axis([0 200 0 0.4]);
title("Max Overall Channel Response");
figname = "all";
if shuffleLabels
    figname = figname + "_shuffle";
end
saveas(gcf, "../Figures/IEM/" + figname + "max_chan_response.png");

figure;
sd_count = max(sd_count, ones(1, n_bins));
sd_bins = sd_bins./sd_count';
imagesc(sd_bins');
caxis([0.2, 0.3]);
title("Serial Dependence Channel Response");
ylabel('Orientation channel');
xlabel('Previous - Current Orientation');
figname = "sd";
if shuffleLabels
    figname = figname + "_shuffle";
end
saveas(gcf, "../Figures/IEM/" + figname + ".png");

