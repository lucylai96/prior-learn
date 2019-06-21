function plearn
% purpose: simulation of prior learning as structure learning 
% dependences: dpkf.m


%% first, need to generate samples from two priors

short = round(exp([ 5.6183 5.6683 5.7183 5.7683 5.8183 5.8683 5.9183]));
long = round(exp([7.0046 7.0546 7.1046 7.1546 7.204 7.2546 7.3046]));


randsample(short,
%Blocked sessions: each of seven stimulus durations was presented 10 times in a pseudorandom order
% 70 trials total

%Interleaved sessions: comprised 140 trials ? 70 for each duration range
%(roughly 10 times)

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