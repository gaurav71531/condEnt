clear all

% The code is written to reproduce the results of paper:

% Gaurav Gupta A.K. Chaturvedi, "Conditional Entropy Based User Selection 
% for Multiuser MIMO Systems", IEEE Comm Lett. 2013

% The code is open-access, and for using the code the user is required to 
% cite the above paper.
% 
%% generate channel matrices

n_t = 8; % number of transmit antennas
n_r = 2; % number of receiver antennas
users = 5:5:40;  % x-axis of plot (total number of users in the system)

channel_mat(n_t, n_r, users);
SNR = 20; % in dB

% execute algorithms

%% 1. c-algorithm
MU_MIMO_SC('capacity', n_t, n_r, SNR, users);

%% 2. n-algorithm
MU_MIMO_SC('frobenius', n_t, n_r, SNR, users);

%% 3. conditional entropy
MU_MIMO_SC('condent', n_t, n_r, SNR, users);

%% 4. chordal distance
MU_MIMO_SC('chordal', n_t, n_r, SNR, users);
% choosing alpha = 0.17 for n_t,n_r = 8,2

%% plot results
figure;
plot(users, R_F_cap_20, 'k', users, R_F_fro_20, 'b', ...
    users, R_F_gg_BD_20, 'r', users, R_F_cho_20, 'g');
grid