function plearn
% purpose: simulation of prior learning as structure learning
% dependences: dpkf.m
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools/')
setPretty
close all
%% simulate and use the experimentally learned prior

[blocked, inter, learned_prior] =log_prior_learn;
[blocked, inter, learned_prior] = prior_learn;
%%load('log_priors.mat')
[ts, tp] = reprod_task(blocked, inter, learned_prior);

%% simulate and use the experimentally learned prior
[blocked, inter, learned_prior] = prior_learn;
[ts, tp] = reprod_task(blocked, inter, learned_prior);


end

function [ts, tp] = reprod_task(blocked, inter, learned_prior)
% compute Bayesian estimate of reproduction task

%% interleaved experiment
p = learned_prior.i;
log_sig = 0.1;
x = 1:2000;
ts.i = inter;
lin_m = zeros([1 length(inter)]);
lin_s = zeros([1 length(inter)]);

for i = 2:length(inter)
    
    % go into linear space
    lin_m(i) = exp(inter(i) + log_sig ^2/2);
    lin_s(i) = sqrt((exp(log_sig^2)-1) * exp(2*inter(i) + log_sig^2));
    
    % calculate likelihood
    lik(i,:) = normpdf(x, lin_m(i)+normrnd(0,lin_s(i)),lin_s(i)); %p(log x | log D)
    
    % bayes rule
    post(i,:) = lik(i,:).*mean(p(1:i,:));
    
    % normalize and compute BLS estimate (mean)
    post(i,:) =  post(i,:)./sum( post(i,:));
    tp.i(i) =  sum(x.*post(i,:));
end

tp.i = tp.i/1000;
ts.i = lin_m/1000;

% plot tp vs time
figure(1); subplot 211; hold on;
plot(tp.i,'b.')
plot(movmean(tp.i,10),'b')
xlabel('trials')
ylabel('duration (s)')
title('produced duration ')
prettyplot

% plot tp vs ts
figure(2); hold on;
loglog(ts.i,tp.i,'b.','MarkerSize',15);
% xlabel('stimulus duration (s)'); ylabel('produced duration (s)')
% title('interleaved presentation')
% dline
% axis square
% axis([0 2 0 2])
% prettyplot

% plot ts vs time
figure(3); subplot 211;hold on;
plot(ts.i,'b.')
plot(movmean(ts.i,10),'b')
xlabel('trials')
ylabel('duration (s)')
prettyplot
title('stimulus duration ')

%% blocked experiment
p = learned_prior.b;
log_sig = 0.1;
x = 1:2000;
ts.b = [blocked.s blocked.l];
lin_m = zeros([1 length(inter)]);
lin_s = zeros([1 length(inter)]);

for i = 2:length(inter)
    
    % go into linear space
    lin_m(i) = exp(ts.b(i) + log_sig ^2/2);
    lin_s(i) = sqrt((exp(log_sig^2)-1) * exp(2*ts.b(i) + log_sig^2));
    
    % calculate likelihood
    lik(i,:) = normpdf(x, lin_m(i)+normrnd(0,lin_s(i)),lin_s(i)); %p(log x | log D)
    
    % bayes rule
    post(i,:) = lik(i,:).*mean(p(1:i,:));
    
    % normalize and compute BLS estimate (mean)
    post(i,:) =  post(i,:)./sum( post(i,:));
    tp.b(i) =  sum(x.*post(i,:));
end

tp.b = tp.b/1000;
ts.b = lin_m/1000;

% plot tp vs time
figure(1); subplot 212;hold on;
plot(tp.b,'r.')
plot(movmean(tp.b,10),'r')
xlabel('trials')
ylabel('duration (s)')
prettyplot

% plot tp vs ts
figure(2); hold on;
loglog(ts.b,tp.b,'r.','MarkerSize',15);
dline
xlabel('stimulus duration (s)'); ylabel('produced duration (s)')
legend('interleaved','blocked')
axis square
axis([0 2 0 2])
prettyplot

% plot ts vs time
figure(3); subplot 212;hold on;
plot(ts.b,'r.')
plot(movmean(ts.b,10),'r')
xlabel('trials')
ylabel('duration (s)')
prettyplot
end

function dyn_prior
%the dynamic prior updated on each trial
end

function [blocked, inter, learned_prior] = prior_learn
%% 1) generate samples from two priors
short = round(exp([ 5.6183 5.6683 5.7183 5.7683 5.8183 5.8683 5.9183])); % stimuli from short prior
long = round(exp([7.0046 7.0546 7.1046 7.1546 7.204 7.2546 7.3046])); % stimuli from long prior

%% 2) set up experimental conditions
% Roach 2015 experimental design:
% - blocked sessions: each of 7 stimulus durations was presented 10x in a pseudorandom order = 70 trials in 1 block
% - interleaved sessions: each of 14 durations presented 10x = 140 trials total
blocked.s = repmat(short,1,10);
blocked.l = repmat(long,1,10);
blocked.s = blocked.s(randperm(length(blocked.s)));
blocked.l = blocked.l(randperm(length(blocked.l)));

inter = [blocked.s blocked.l];
inter = inter(randperm(length(inter)));

%opts.R = 200^2; % noise covariance (sensory noise)
opts.R = 250^2; % noise covariance (sensory noise) 250
opts.Q = 0.1; %diffusion covariance (default: 0.01*eye(D)) how much env is actually changing?
opts.W = 1;% dynamics matrix (default: eye(D))
opts.C = 200^2; % prior state covariance (default: 10*eye(D)) 200
opts.sticky = 0; % stickiness of last mode
opts.alpha = 0.11; % concentration parameter (default: 0.1) 0.1
opts.x0 = mean(inter);
%opts.x0 = inter(1); % prior mean (start at the ultimate mean)
opts.Kmax = 10; % upper bound on number of states
%opts.sticky = 0;

Y = inter';
results = dpkf(Y,opts); % runs through each observation

%% interleaved --> 1 big prior
% plot the resulting learned priors
figure; colors_p = gradientCol(140,3);
set(gcf, 'DefaultAxesColorOrder',  colors_p)
hold on;
for i = 1:140
    idx(i,:) = results(i).pZ; %posterior probability of each mode
end
idx = nanmean(idx)*140;

subplot 121
bar(idx)
ylabel('number of samples in mode'); xlabel('mode')

% posterior mean state estimate
for i = 1:140
    means(i,:) = sum(results(i).x' .* results(i).pZ); % means x their posterior probability
end

% posterior of modes (what is your learned prior?)
subplot 122; hold on;

for i = 1:140
    post_std = sqrt(cell2mat(results(i).P)); %posterior stds
    MAP_idx = find(max(results(i).pZ)); % idx of highest probability mode
    
    learned_prior.i(i,:) = normpdf(0:2000,means(i), post_std(MAP_idx));
    plot(0:2000,learned_prior.i(i,:))
    pause(0.01)
    
end
plot(mean(learned_prior.i),'k','LineWidth',3) % what you should be observing at the end is the mean over time points
ylabel('posterior probability of modes'); xlabel('duration (ms)')
subprettyplot(1,2)


%% blocked --> 2 priors

Y = [blocked.s blocked.l]';
opts.x0 = mean(inter); % prior mean (start at the ultimate mean)
results = dpkf(Y,opts); % runs through each observation

% plot the resulting learned priors
figure;
colors_p1 = gradientCol(70,1); colors_p2 = gradientCol(70,2); colors_p = [colors_p1;colors_p2];
set(gcf, 'DefaultAxesColorOrder',  colors_p); hold on;

for i = 1:140
    idx(i,:) = results(i).pZ; %posterior probability of each mode
end
idx = nanmean(idx)*140;

subplot 121
bar(idx)
ylabel('number of samples in mode'); xlabel('mode')

% posterior mean state estimate
for i = 1:140
    means(i,:) = sum(results(i).x' .* results(i).pZ); % means x their posterior probability
end

% posterior of modes (what is your learned prior?)
subplot 122; hold on;

for i = 1:140
    post_std = sqrt(cell2mat(results(i).P)); %posterior stds
    MAP_idx = find(max(results(i).pZ)); % idx of highest probability mode
    
    learned_prior.b(i,:) = normpdf(0:2000,means(i), post_std(MAP_idx));
    semilogy(learned_prior.b(i,:))
    pause(0.01)
end


semilogy(mean(learned_prior.b),'k','LineWidth',3) % what you should be observing at the end is the mean over time points
ylabel('posterior probability of modes'); xlabel('duration (ms)')
subprettyplot(1,2)

end


function [blocked, inter, learned_prior]=log_prior_learn
%% 1) generate samples from two priors
short = [ 5.6183 5.6683 5.7183 5.7683 5.8183 5.8683 5.9183]; % stimuli from short prior
long = [7.0046 7.0546 7.1046 7.1546 7.204 7.2546 7.3046]; % stimuli from long prior

%% 2) set up experimental conditions
% Roach 2015 experimental design:
% - blocked sessions: each of 7 stimulus durations was presented 10x in a pseudorandom order = 70 trials in 1 block
% - interleaved sessions: each of 14 durations presented 10x = 140 trials total
blocked.s = repmat(short,1,10);
blocked.l = repmat(long,1,10);
blocked.s = blocked.s(randperm(length(blocked.s)));
blocked.l = blocked.l(randperm(length(blocked.l)));

inter = [blocked.s blocked.l];
inter = inter(randperm(length(inter)));

opts.R = 0.13; % noise covariance (sensory noise)
opts.Q = 0.01; %diffusion covariance (default: 0.01*eye(D)) how much env is actually changing?
opts.W = 1;% dynamics matrix (default: eye(D))
opts.C = 0.09; % prior state covariance (default: 10*eye(D))
opts.sticky = 0; % stickiness of last mode
opts.alpha = 0.14; % concentration parameter (default: 0.1)
opts.x0 = mean(inter);
%opts.x0 = inter(1); % prior mean (start at the ultimate mean)
opts.Kmax = 10; % upper bound on number of states
%opts.sticky = 0;

Y = inter';
results = dpkf(Y,opts); % runs through each observation

%% interleaved --> 1 big prior
% plot the resulting learned priors
figure; colors_p = gradientCol(140,3);
set(gcf, 'DefaultAxesColorOrder',  colors_p)
hold on;
for i = 1:140
    idx(i,:) = results(i).pZ; %posterior probability of each mode
end
idx = nanmean(idx)*140;

subplot 121
bar(idx)
ylabel('number of samples in mode'); xlabel('mode')

% posterior mean state estimate
for i = 1:140
    means(i,:) = sum(results(i).x' .* results(i).pZ); % means x their posterior probability
end

% posterior of modes (what is your learned prior?)
subplot 122; hold on;
for i = 1:140
    post_std = sqrt(cell2mat(results(i).P)); %posterior stds
    MAP_idx = find(max(results(i).pZ)); % idx of highest probability mode
    post_std(i) = post_std (MAP_idx);
    
    % change back to linear space
    lin_means.i(i) = exp(means(i) + post_std(i)^2/2);
    lin_post_std.i(i) = sqrt((exp(post_std(i)^2)-1) * exp(2*means(i) + post_std(i)^2));
    
    learned_prior.i(i,:) = normpdf(1:2000,lin_means.i(i), lin_post_std.i(i));
    plot([1:2000]/1000,learned_prior.i(i,:))
    pause(0.01)
end
plot([1:2000]/1000,mean(learned_prior.i),'k','LineWidth',3) % what you should be observing at the end is the mean over time points
ylabel('posterior probability of modes'); xlabel('duration (s)')
subprettyplot(1,2)


%% blocked --> 2 priors

Y = [blocked.s blocked.l]';
opts.x0 = mean(inter); % prior mean (start at the ultimate mean)
results = dpkf(Y,opts); % runs through each observation

% plot the resulting learned priors
figure;
colors_p1 = gradientCol(70,1); colors_p2 = gradientCol(70,2); colors_p = [colors_p1;colors_p2];
set(gcf, 'DefaultAxesColorOrder',  colors_p); hold on;

for i = 1:140
    idx(i,:) = results(i).pZ; %posterior probability of each mode
end
idx = nanmean(idx)*140;

subplot 121
bar(idx)
ylabel('number of samples in mode'); xlabel('mode')

% posterior mean state estimate
for i = 1:140
    means(i,:) = sum(results(i).x' .* results(i).pZ); % means x their posterior probability
end

% posterior of modes (what is your learned prior?)
subplot 122; hold on;
for i = 1:140
    post_std = sqrt(cell2mat(results(i).P)); %posterior stds
    MAP_idx = find(max(results(i).pZ)); % idx of highest probability mode
    post_std(i) = post_std (MAP_idx);
    
    % change back to linear space
    lin_means.b(i) = exp(means(i) + post_std(i)^2/2);
    lin_post_std.b(i) = sqrt((exp(post_std(i)^2)-1) * exp(2*means(i) + post_std(i)^2));
    
    learned_prior.b(i,:) = normpdf(1:2000,lin_means.b(i), lin_post_std.b(i));
    plot([1:2000]/1000,learned_prior.b(i,:))
    pause(0.01)
end
plot([1:2000]/1000,mean(learned_prior.b),'k','LineWidth',3) % what you should be observing at the end is the mean over time points
ylabel('posterior probability of modes'); xlabel('duration (s)')
subprettyplot(1,2)

end