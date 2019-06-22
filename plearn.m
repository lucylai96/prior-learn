function plearn
% purpose: simulation of prior learning as structure learning
% dependences: dpkf.m
setPretty
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

%opts.R = 200^2; %noise covariance (sensory noise)
opts.R = 1000;
opts.C = 500^2; %prior covariance
opts.sticky = .2; %stickiness
opts.alpha = 0.1;
opts.x0 = mean(inter);
%opts.x0 = inter(1); % prior mean (start at the ultimate mean)
opts.Kmax = 10; % upper bound on number of states
%opts.sticky = 0;

Y = inter';
results = dpkf(Y,opts); % runs through each observation

%% interleaved --> 1 big prior
% plot the resulting learned priors
figure; hold on;
for i = 1:140
    idx(i,:) = results(i).pZ; %posterior probability of each mode
end
idx = mean(idx);

subplot 131
bar(idx)
ylabel('number of samples in mode'); xlabel('mode')

% posterior mean state estimate
subplot 132
plot(results(end).x,'o')
ylabel('posterior mean state estimates'); xlabel('mode')

% posterior probability of modes (what is your learned prior?)
subplot 133; hold on;
for i = 1:10
    plot( normpdf(0:3000,results(end).x(i),sqrt(cell2mat(results(end).P(i)))))
end
ylabel('posterior probability of modes'); xlabel('duration (ms)')
subprettyplot(1,3)


%% blocked --> 2 priors

Y = [blocked.s blocked.l]';
opts.x0 = mean(inter); % prior mean (start at the ultimate mean)
results = dpkf(Y,opts); % runs through each observation

% plot the resulting learned priors
figure;hold on;
for i = 1:140
    idx(i,:) = results(i).pZ;
end
idx = mean(idx);

subplot 131
bar(idx)
ylabel('number of samples in mode'); xlabel('mode')

% posterior mean state estimate
subplot 132
plot(results(end).x,'o')
ylabel('posterior mean state estimates'); xlabel('mode')

% posterior probability of modes (what is your learned prior?)
subplot 133; hold on;
for i = 1:10
    plot( normpdf(0:3000,results(end).x(i),sqrt(cell2mat(results(end).P(i)))))
end
ylabel('posterior probability of modes'); xlabel('duration (ms)')
subprettyplot(1,3)


end

function generate_samples
% this function generates a new sample from the prior

% USAGE: results = dpkf(Y,[opts])
%
% INPUTS:
%   Y - [T x D] observation sequence, where Y(t,d) is dimension d of the observation at time t
%   opts (optional) - structure with any of the following fields
%                     (missing fields are set to defaults):
%                       .R = noise covariance (default: eye(D))
%                       .Q = diffusion covariance (default: 0.01*eye(D))
%                       .W = dynamics matrix (default: eye(D))
%                       .C = prior state covariance (default: 10*eye(D))
%                       .alpha = concentration parameter (default: 0.1)
%                       .sticky = stickiness of last mode (default: 0)
%                       .x0 = prior mean (default: zeros(1,D))
%                       .Kmax = upper bound on number of state (default: 10)
%                   Note: if R, Q, W, or C are given as scalars, it is
%                         assumed that they are the same for all dimensions
end