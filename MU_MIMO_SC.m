function MU_MIMO_SC(varargin)
% % % 
% The code is written to reproduce the results of paper:

% Gaurav Gupta and A.K. Chaturvedi, "Conditional Entropy Based User Selection 
% for Multiuser MIMO Systems", IEEE Comm Lett. 2013

% The code is open-access, and for using the code the user is required to 
% cite the above paper.
% 
% % % 
% 
% entering sequence " directive, way (optional), n_t, n_r, SNR, users_mat "

% directive =   code for selecting algorithm 
% n_t       =   number of Tx antennas
% n_r       =   number of Rx antennas
% SNR       =   Signal to noise ratio (in decibels)
% global data_gg
ii = 2;
if nargin == 6, ii = ii + 1;end
n_t = varargin{ii};
n_r = varargin{ii+1};
SNR = varargin{ii+2};
users_num = varargin{ii+3};

SNR = 10 ^ (SNR/10);

switch lower(varargin{1})
    
    case 'optimal'   
        if nargin == 6 && strcmp(varargin{2}, 'SZF')
            R_F = SumCapacity_BDOptimal_SZF(n_t, n_r, SNR, users_num);
            SNR = round(10*log10(SNR));
            if SNR < 0
                str = ['m',num2str(abs(SNR))];
            else
                str = num2str(SNR);
            end
            str = ['R_F_opt_SZF_',str];
        else
            R_F = SumCapacity_BDOptimal(n_t, n_r, SNR, users_num);
            SNR = round(10*log10(SNR));
            str = ['R_F_opt_BD_',num2str(SNR)];   
        end
        
    case 'capacity'
        R_F = SumCapacity_BDCapacity(n_t, n_r, SNR, users_num);
        SNR = round(10*log10(SNR));
        str = ['R_F_cap_',num2str(SNR)];
        
    case 'frobenius'
        R_F = SumCapacity_BDFrobenius(n_t, n_r, SNR, users_num);
        SNR = round(10*log10(SNR));
        str = ['R_F_fro_',num2str(SNR)];
        
    case 'capacity_upper'
        R_F = SumCapacity_CapacityUpperbound(n_t,n_r,SNR,users_num);
        SNR = round(10*log10(SNR));
        str = ['R_F_capU_',num2str(SNR)];
        
    case 'iter'
        R_F = SumCapacity_Frob_iter(n_t,n_r,SNR,users_num);
        SNR = round(10*log10(SNR));
        str = ['R_F_iter_',num2str(SNR)];
        
    case 'chordal'
        R_F = SumCapacity_Chordal(n_t, n_r, SNR, users_num);
        SNR = round(10*log10(SNR));
        str = ['R_F_cho_',num2str(SNR)];
        
    case 'condent'
        if nargin == 6 && strcmp(varargin{2}, 'SZF')
            R_F = SumCapacity_gg_SZF(n_t, n_r, SNR, users_num);
            SNR = round(10*log10(SNR));
            str = ['R_F_gg_SZF_',num2str(SNR)];
           
        else
            R_F = SumCapacity_condEnt(n_t,n_r,SNR,users_num);
            SNR = round(10*log10(SNR));
            str = ['R_F_gg_BD_',num2str(SNR)];
        end 
        
     case 'ggm'
        if nargin == 6 && strcmp(varargin{2}, 'SZF')
            R_F = SumCapacity_gg_SZF(n_t, n_r, SNR, users_num);
            SNR = round(10*log10(SNR));
            str = ['R_F_gg_SZF_',num2str(SNR)];
           
        else
             R_F = SumCapacity_gg_mod(n_t,n_r,SNR,users_num);
            SNR = round(10*log10(SNR));
            str = ['R_F_ggm_BD_',num2str(SNR)];
        end 
        
    case 'test'
        R_F = SumCapacity_test(n_t, n_r, SNR, users_num);
        SNR = round(10*log10(SNR));
        str = ['R_F_test_',num2str(SNR)];
        
    case 'szf'
        R_F = SumCapacity_SZF(n_t, n_r, SNR, users_num);
        SNR = round(10*log10(SNR));
        str = ['R_F_SZF_',num2str(SNR)];
        
    case 'zfbf'
        if nargin == 6 && strcmp(varargin{2}, 'wc')
            R_F = SumCapacity_ZFBF(n_t, n_r, SNR, users_num);
            SNR = round(10*log10(SNR));
            str = ['R_F_ZFBF_wc_',num2str(SNR)];
        else
            sec = 'regular';
            if nargin == 6 && strcmp(varargin{2}, 'RxTx')
                sec = varargin{2};
            end
            R_F = SumCapacity_ZFBF_WOC(sec, n_t, n_r, SNR, users_num);
            SNR = round(10*log10(SNR));
            str = ['R_F_ZFBF_woc_',num2str(SNR)];
        end
        
    case 'tdma'
        R_F = SumCapacity_TDMA(n_t, n_r, SNR, users_num);
        SNR = round(10*log10(SNR));
        str = ['R_F_TDMA_',num2str(SNR)];
        
    case 'double_search'
    R_F = SumCapacity_DSearch(n_t,n_r,SNR,users_num);
    SNR = round(10*log10(SNR));
    str = ['R_F_DS_',num2str(SNR)];
        
    otherwise
        disp('Wrong directive input, try again!');
        return
        
end
assignin('base', str, R_F);
% assignin('base', 'data', data_gg);


function R_F = SumCapacity_SZF(n_t,n_r,SNR,users_num)

R_F = zeros(1, length(users_num));
time = zeros(1, length(users_num));
for mi = 1:length(users_num)
    nusers = users_num(mi);
    R_final = zeros(1000,1);    
    str = [' Channel_Mat/','H_all_',num2str(n_t),'_',num2str(n_r),'_', num2str(nusers), '.mat'];
    eval(['load' str]);
    K = ceil(n_t / n_r);
    if K > nusers, K = nusers;end
    tic
    for avg_sim_ind = 1:1000
%         H = (randn(n_r, n_t, nusers) + sqrt(-1) * randn(n_r, n_t, nusers)) / sqrt(2);
        H = H_all(:,:,:,avg_sim_ind);
        
        T_set = 1:nusers;
        V_0 = cell(1,K);
        V_1 = cell(1,K);
        FrobTemp = 0;
        FrobNormC = zeros(length(T_set),1);
        for i = T_set
            FrobNormC(i) = trace(H(:,:,i) * H(:,:,i)');
            if FrobNormC(i) > FrobTemp
                FrobTemp = FrobNormC(i);
                maxValInd = i;
            end
        end
        T_set(T_set == maxValInd) = [];
        S_set = maxValInd;
        [~,G,V] = svd(H(:,:,maxValInd));
        rank = length(find(G));
        V_1{1} = V(:,1:rank);
        V_0{1} = V(:,rank+1:end);
        
        for i = 2:K
            
            H_Us = zeros(n_r*(i-1),n_t);
            for j = 1:i-1
                H_Us(n_r*(j-1)+1:n_r*j,:) = H(:,:,S_set(j));
            end
            [~,G,V] = svd(H_Us);
            rank = length(find(G));
            V_1{i} = V(:,1:rank);
            V_0{i} = V(:,rank+1:end);
            
            alpha = 0.65;
            U_set = [];
            for k = T_set
                Ent = sqrt(trace((H(:,:,k)*V_1{i})*(H(:,:,k)*V_1{i})') / ...
                    (FrobNormC(k) * trace(V_1{i}*V_1{i}')));
                if Ent < alpha
                    U_set = [U_set, k];
                end
            end
            if ~isempty(U_set)
                Ent_Temp = 0;
                for k = U_set
                    if i == 2
                        Ent = trace((H(:,:,k)*V_0{i})*(H(:,:,k)*V_0{i})');
                    else
                        qty = 0;
                        for j = 2:i-1
                            qty = qty + trace((H(:,:,k)*V_0{j})*(H(:,:,k)*V_0{j})');
                        end
                        Ent = trace((H(:,:,k)*V_0{i})*(H(:,:,k)*V_0{i})') / qty;
                    end
                    if Ent > Ent_Temp
                        Ent_Temp = Ent;
                        maxValInd = k;
                    end
                end
                S_set = [S_set, maxValInd];
                T_set(T_set == maxValInd) = [];
            else
                break;
            end
        end
        R_final(avg_sim_ind) = SumCapacityOfUserSet_SZF(S_set, H, SNR, n_t, n_r);
    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 
fprintf('Total time taken = %f\n', sum(time));


function R_F = SumCapacity_condEnt(n_t,n_r,SNR,users_num)

R_F = zeros(1, length(users_num));
time = zeros(1, length(users_num));
num_sim = 1000;
for mi = 1:length(users_num)
    nusers = users_num(mi);
    str = [' Channel_Mat/','H_all_',num2str(n_t),'_',num2str(n_r),'_', num2str(nusers), '.mat'];
    eval(['load' str]);
    R_final = zeros(num_sim,1);    
    tic
    for avg_sim_ind = 1:num_sim
%         H = (randn(n_r, n_t, nusers) + sqrt(-1) * randn(n_r, n_t, nusers)) / sqrt(2);
        H = H_all(:,:,:,avg_sim_ind);
        K = ceil(n_t / n_r);
        if K > nusers, K = nusers;end
        T_set = 1:nusers;
        
%        GG algorithm

        ohm = (SNR/n_t) * eye(n_t);
        R_Temp = 0;
%         G_set = [];
        ohm_n = ohm;
        ohm_s = zeros(n_t,n_t,K);
        Ent_Temp = 0;
        for i = T_set
            Ent = real(det(eye(n_r) + H(:,:,i)*ohm_n*H(:,:,i)'));
            if Ent > Ent_Temp
                Ent_Temp = Ent;
                maxValIndFirst = i;
            end
        end
        G_set = maxValIndFirst;
        T_set(T_set == maxValIndFirst) = [];
        ohm_s(:,:,1) = ohm_n;
        ohm_n = ohm_n - ohm_n * H(:,:,maxValIndFirst)' *...
            ((eye(n_r) + H(:,:,maxValIndFirst)*ohm_n*H(:,:,maxValIndFirst)') \...
            H(:,:,maxValIndFirst)*ohm_n);
        
        for i = 2:K
            Ent_Temp = 0;
            for k = T_set
                Hk = H(:,:,k);
                Ent_total = log2(real(det(eye(n_r) + Hk*ohm_n*Hk')));
                ohm_Temp = zeros(n_t,n_t,K);
                for g = 1:length(G_set)
                    ohm_use = ohm_s(:,:,g);
                    ohm_Temp(:,:,g) = ohm_use - (ohm_use*Hk')*...
                        ((eye(n_r)+Hk*ohm_use*Hk') \ Hk *ohm_use);
                    Ent_total = Ent_total + log2(real(det(eye(n_r) + H(:,:,G_set(g))*ohm_Temp(:,:,g)*H(:,:,G_set(g))')));
                end
                if Ent_total > Ent_Temp
                    Ent_Temp = Ent_total;
                    maxValInd = k;
                    ohm_sTemp = ohm_Temp;
                end
            end
            G_setTemp = [G_set,maxValInd];   
            R = SumCapacityOfUserSet_opt(G_setTemp, H, SNR, n_t, n_r);
            if R < R_Temp
                break
            else
                ohm_s = ohm_sTemp;
                R_Temp = R;
                G_set = G_setTemp;
                T_set(T_set == maxValInd) = [];
            end 
            ohm_s(:,:,i) = ohm_n;
            ohm_n = ohm_n - ohm_n * H(:,:,maxValInd)' *...
                ((eye(n_r) + H(:,:,maxValInd)*ohm_n*H(:,:,maxValInd)') \...
                H(:,:,maxValInd)*ohm_n);
        end  
       R_final(avg_sim_ind) = R_Temp;
        
    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 
fprintf('Total time taken = %f\n', sum(time));


function R_F = SumCapacity_test(n_t,n_r,SNR,users_num)

R_F = zeros(1, length(users_num));
time = zeros(1, length(users_num));
num_sim = 1000;
for mi = 1:length(users_num)
    nusers = users_num(mi);
    str = [' Channel_Mat/','H_all_',num2str(n_t),'_',num2str(n_r),'_', num2str(nusers), '.mat'];
    eval(['load' str]);
    R_final = zeros(num_sim,1);    
    tic
    for avg_sim_ind = 1:num_sim
%         H = (randn(n_r, n_t, nusers) + sqrt(-1) * randn(n_r, n_t, nusers)) / sqrt(2);
        H = H_all(:,:,:,avg_sim_ind);
        K = ceil(n_t / n_r);
        if K > nusers, K = nusers;end
        T_set = 1:nusers;
        
%        GG algorithm
        Ent_Temp = 0;
        P = SNR/n_t;
        for i = T_set
            Ent = real(det(eye(n_r) + P*H(:,:,i)*H(:,:,i)'));
            if Ent > Ent_Temp
                Ent_Temp = Ent;
                maxValIndFirst = i;
            end
        end
%         G_set = maxValIndFirst;
        H_1 = H(:,:,maxValIndFirst);
        Ent1 = log2(real(det(eye(n_r) + P*(H_1*H_1'))));
        T_set(T_set == maxValIndFirst) = [];
        Ent_temp = 0;
        for i = T_set
            H_2 = H(:,:,i);
            Ent2 = log2(real(det(eye(n_r) + P*(H_2*H_2'))));
            Ent12 = log2(real(det(eye(n_t) + P*(H_1'*H_1) + P*(H_2'*H_2))));
%             Ent = 2*Ent12 - Ent1 - Ent2;
            Inf12 = Ent1 + Ent2 - Ent12;
            Ent = Ent12 - 1.65*Inf12;
            if Ent > Ent_temp
                maxValInd = i;
                Ent_temp = Ent;
            end
        end
        G_set = [maxValIndFirst, maxValInd];
        R_Temp = SumCapacityOfUserSet_opt(G_set, H, SNR, n_t, n_r);
        R_final(avg_sim_ind) = R_Temp;
        
    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 
fprintf('Total time taken = %f\n', sum(time));


function R_F = SumCapacity_gg_mod(n_t,n_r,SNR,users_num)

R_F = zeros(1, length(users_num));
time = zeros(1, length(users_num));
num_sim = 1000;
for mi = 1:length(users_num)
    nusers = users_num(mi);
    str = [' Channel_Mat/','H_all_',num2str(n_t),'_',num2str(n_r),'_', num2str(nusers), '.mat'];
    eval(['load' str]);
    R_final = zeros(num_sim,1);    
    tic
    for avg_sim_ind = 1:num_sim
%         H = (randn(n_r, n_t, nusers) + sqrt(-1) * randn(n_r, n_t, nusers)) / sqrt(2);
        H = H_all(:,:,:,avg_sim_ind);
        K = ceil(n_t / n_r);
        if K > nusers, K = nusers;end
        T_set = 1:nusers;
        
%        GG algorithm

        ohm = (SNR/n_t) * eye(n_t);
        R_Temp = 0;
        ohm_n = ohm;
        Ent_Temp = 0;
        for i = T_set
            Ent = real(det(eye(n_r) + H(:,:,i)*ohm_n*H(:,:,i)'));
            if Ent > Ent_Temp
                Ent_Temp = Ent;
                maxValIndFirst = i;
            end
        end
        G_set = maxValIndFirst;
        T_set(T_set == maxValIndFirst) = [];
        ohm_n = ohm_n - ohm_n * H(:,:,maxValIndFirst)' *...
            ((eye(n_r) + H(:,:,maxValIndFirst)*ohm_n*H(:,:,maxValIndFirst)') \...
            H(:,:,maxValIndFirst)*ohm_n);
        
        for i = 2:K
            Ent_Temp = 0;
            for k = T_set
                Hk = H(:,:,k);
                Ent_total = 2*log2(real(det(eye(n_r) + Hk*ohm_n*Hk'))) - ...
                    log2(real(det(eye(n_r) + Hk*ohm*Hk')));
                if Ent_total > Ent_Temp
                    Ent_Temp = Ent_total;
                    maxValInd = k;
                end
            end
            G_setTemp = [G_set,maxValInd];
            R = SumCapacityOfUserSet_opt(G_setTemp, H, SNR, n_t, n_r);
            if R < R_Temp
                break
            else
                R_Temp = R;
                G_set = G_setTemp;
                T_set(T_set == maxValInd) = [];
            end 
            ohm_n = ohm_n - ohm_n * H(:,:,maxValInd)' *...
                ((eye(n_r) + H(:,:,maxValInd)*ohm_n*H(:,:,maxValInd)') \...
                H(:,:,maxValInd)*ohm_n);
        end
        
       R_final(avg_sim_ind) = R_Temp;
        
    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 
fprintf('Total time taken = %f\n', sum(time));


function R_F = SumCapacity_ZFBF_WOC(sec, n_t, n_r, SNR, users_num)

alpha = 0.62;
R_F = zeros(1, length(users_num));
time = zeros(1, length(users_num));
num_sim = 1000;
for mi = 1:length(users_num)
    nusers = users_num(mi);
    str = [' Channel_Mat/','H_all_',num2str(n_t),'_',num2str(n_r),'_', num2str(nusers), '.mat'];
    eval(['load' str]);
    R_final = zeros(num_sim,1);    
    tic
    for avg_sim_ind = 1:num_sim
        H = H_all(:,:,:,avg_sim_ind);
%         K = ceil(n_t / n_r);
%         if K > nusers, K = nusers;end;
%         T_set = 1:nusers;
        T_set = 1:nusers*n_r;
        if n_r ~= 1
            H_use = zeros(1, n_t, nusers*n_r);
            sz = 1;
            for j = 1:nusers
                [~,G,V] = svd(H(:,:,j));
                for i = 1:n_r
                    H_use(:,:,sz) = G(i,i)*V(:,i)';
                    sz = sz + 1;
                end
            end
        else
            H_use = H;
        end
        FrobTemp = 0;
        FrobNormC = zeros(length(T_set),1);
        G_mat = zeros(1, n_t, n_t);
        for i = T_set
            FrobNormC(i) = norm(H_use(:,:,i));
            if FrobNormC(i) > FrobTemp
                FrobTemp = FrobNormC(i);
                maxValInd = i;
            end
        end
        G_set = maxValInd;
        G_mat(:,:,1) = H_use(:,:,maxValInd);
        T_set(T_set == maxValInd) = [];
        T_set_temp = [];
        for i = T_set
            Frob_Val = abs(H_use(:,:,i)*G_mat(:,:,1)') / ...
                (FrobNormC(i)*FrobNormC(maxValInd));
            if Frob_Val < alpha
                T_set_temp = [T_set_temp, i];
            end
        end
        T_set = T_set_temp;
        
        for i = 2:n_t
            
            FrobTemp = 0;
            maxValInd = [];
            for k = T_set
                mat = eye(n_t);
                for j = 1:i-1
                    mat = mat - (G_mat(:,:,j)'*G_mat(:,:,j) / ...
                        (G_mat(:,:,j)*G_mat(:,:,j)'));
                end
                mat = H_use(:,:,k)*mat;
                Frob_Val = norm(mat);
                if Frob_Val > FrobTemp
                    FrobTemp = Frob_Val;
                    maxValInd = k;
                    G_mat_bk = mat;
                end
            end
            G_set = [G_set, maxValInd];
            G_mat(:,:,i) = G_mat_bk;
            T_set(T_set == maxValInd) = [];
            
            T_set_temp = [];
            for k = T_set
                Frob_Val = abs(H_use(:,:,k)*G_mat(:,:,i)') / ...
                    (FrobNormC(k)*norm(G_mat(:,:,i)));
                if Frob_Val < alpha
                    T_set_temp = [T_set_temp, k];
                end
            end
            T_set = T_set_temp;
            if isempty(T_set), break;end
        end
        
        if strcmp(sec, 'regular')
            
            R_final(avg_sim_ind) = SumCapacityofZFBF(G_set, H_use, SNR, n_t);
        elseif strcmp(sec, 'RxTx')
            
              G_set_use = ceil(G_set/n_r);
              G_set = unique(G_set_use);
              L_set = histc(G_set_use,G_set);
              if length(G_set)~= 1
                  L_set = L_set(L_set~= 0);
                  [R, T] = getPostPreMatrices(H, G_set, L_set, n_t, n_r);
                  R_final(avg_sim_ind) = SumCapacityForJMode(H, R, T, SNR, G_set, L_set, n_r);
              else
                  R_final(avg_sim_ind) = SumCapacityofZFBF(G_set, H_use, SNR, n_t);
              end
         end
    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 
fprintf('Total time taken = %f\n', sum(time));


function R_S = SumCapacityofZFBF(S_set, H, SNR, n_t)

K = length(S_set);
H_S = zeros(K, n_t);
sz = 1;
for i = 1:K
    H_S(sz, :) = H(:,:,S_set(i));
    sz = sz + 1;
end
Gamma = inv((H_S*H_S'));
lambda = 1 ./ diag(Gamma);
pl = Water_filling(lambda, SNR);
R_S = sum(log2(1 + pl .* lambda));


function R_F = SumCapacity_TDMA(n_t, n_r, SNR, users_num)

R_F = zeros(1, length(users_num));
time = zeros(1, length(users_num));
num_sim = 1000;
for mi = 1:length(users_num)
    nusers = users_num(mi);
    str = [' Channel_Mat/','H_all_',num2str(n_t),'_',num2str(n_r),'_', num2str(nusers), '.mat'];
    eval(['load' str]);
    R_final = zeros(num_sim,1);    
    tic
    for avg_sim_ind = 1:num_sim
        H = H_all(:,:,:,avg_sim_ind);
        R = zeros(1, nusers);
        for i = 1:nusers
            lambda = real(eig(H(:,:,i)*H(:,:,i)'));
            pl = Water_filling(lambda, SNR);
            R(i) = sum(log2(1 + pl.*lambda));
        end
        R_final(avg_sim_ind) = max(R);
    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 
fprintf('Total time taken = %f\n', sum(time));


function R_F = SumCapacity_gg_SZF(n_t,n_r,SNR,users_num)

R_F = zeros(1, length(users_num));
time = zeros(1, length(users_num));
num_sim = 1000;
for mi = 1:length(users_num)
    nusers = users_num(mi);
    str = [' Channel_Mat/','H_all_',num2str(n_t),'_',num2str(n_r),'_', num2str(nusers), '.mat'];
    eval(['load' str]);
    R_final = zeros(num_sim,1);    
%     countg = 0;
    tic
    for avg_sim_ind = 1:num_sim
%         H = (randn(n_r, n_t, nusers) + sqrt(-1) * randn(n_r, n_t, nusers)) / sqrt(2);
        H = H_all(:,:,:,avg_sim_ind);
        K = ceil(n_t / n_r);
        if K > nusers, K = nusers;end
        T_set = 1:nusers;
        
%        GG algorithm

        ohm = (SNR/n_t) * eye(n_t);
        R_Temp = 0;
%         G_set = [];
        ohm_n = ohm;
        ohm_s = zeros(n_t,n_t,K);
        Ent_Temp = 0;
        for i = T_set
            Ent = real(det(eye(n_r) + H(:,:,i)*ohm_n*H(:,:,i)'));
            if Ent > Ent_Temp
                Ent_Temp = Ent;
                maxValIndFirst = i;
            end
        end
        G_set = maxValIndFirst;
        T_set(T_set == maxValIndFirst) = [];
        ohm_s(:,:,1) = ohm_n;
        ohm_n = ohm_n - ohm_n * H(:,:,maxValIndFirst)' *...
            ((eye(n_r) + H(:,:,maxValIndFirst)*ohm_n*H(:,:,maxValIndFirst)') \...
            H(:,:,maxValIndFirst)*ohm_n);
        
        for i = 2:K
            Ent_Temp = 0;
            for k = T_set
                Hk = H(:,:,k);
                Ent_total = log2(real(det(eye(n_r) + Hk*ohm_n*Hk')));
                ohm_Temp = zeros(n_t,n_t,K);
                for g = 1:length(G_set)
                    ohm_use = ohm_s(:,:,g);
                    ohm_Temp(:,:,g) = ohm_use - (ohm_use*Hk')*...
                        ((eye(n_r)+Hk*ohm_use*Hk') \ Hk *ohm_use);
                    Ent_total = Ent_total + log2(real(det(eye(n_r) + H(:,:,G_set(g))*ohm_Temp(:,:,g)*H(:,:,G_set(g))')));
                end
                if Ent_total > Ent_Temp
                    Ent_Temp = Ent_total;
                    maxValInd = k;
                    ohm_sTemp = ohm_Temp;
                end
            end
            G_setTemp = [G_set,maxValInd];
            R = SumCapacityOfUserSet_SZF(G_setTemp, H, SNR, n_t, n_r);
            if R < R_Temp
%                 countg  = countg + 1;
                break
            else
                ohm_s = ohm_sTemp;
                R_Temp = R;
                G_set = G_setTemp;
                T_set(T_set == maxValInd) = [];
            end 
            ohm_s(:,:,i) = ohm_n;
            ohm_n = ohm_n - ohm_n * H(:,:,maxValInd)' *...
                ((eye(n_r) + H(:,:,maxValInd)*ohm_n*H(:,:,maxValInd)') \...
                H(:,:,maxValInd)*ohm_n);
        end
        
       R_final(avg_sim_ind) = R_Temp;
        
    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 
fprintf('Total time taken = %f\n', sum(time));
% countg


function R_F = SumCapacity_Frob_iter(n_t,n_r,SNR,users_num)

R_F = zeros(1, length(users_num));
time = zeros(1,length(users_num));
for mi = 1:length(users_num)
    nusers = users_num(mi);
    R_final = zeros(1000,1);    
    str = [' Channel_Mat/','H_all_',num2str(n_t),'_',num2str(n_r),'_', num2str(nusers), '.mat'];
    eval(['load' str]);
% % % % % % % % % % % % % % CORRELATION CODE % % % % % % % % % % % % % % % % 
%     rho_m = 0.5;
%     R_Tx = zeros(n_t);
%     for sim = 1:1000
%         for use = 1:nusers
% %             H_tmp = R_kron * G(:,:,j,i);
% %             H_all(:,:,j,i) = reshape(H_tmp,n_r,n_t);
%             rho_phi = exp(1i*2*pi*rand);
%             for i = 1:n_t
%                 for j = 1:n_t
%                     R_Tx(i,j) = rho_m^abs(i-j) * rho_phi^(i-j);
%                 end
%             end
%             R = sqrtm(R_Tx);
% %             R = chol(R_Tx,'lower');
%             H_all(:,:,use,sim) = H_all(:,:,use,sim)*R;
%         end
%     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    
    tic
    for avg_sim_ind = 1:1000
        H = H_all(:,:,:,avg_sim_ind);
        K = ceil(n_t / n_r);
        if K > nusers, K = nusers;end
        T_set = 1:nusers;
        Norm_temp = 0;
        for k = T_set
            Norm_pdt = real(prod(diag(H(:,:,k)*H(:,:,k)')));
            if Norm_pdt > Norm_temp
                Norm_temp = Norm_pdt;
                maxValInd = k;
            end
        end
        T_set(T_set == maxValInd) = [];
        S_set = maxValInd;
        [Qr,~] = qr(H(:,:,maxValInd)');
        Q = Qr(:,n_r+1:end);        
        B = cell(1,nusers);
        B{maxValInd} = eye(n_t);
        H_eff = cell(1,nusers);
        for use = 1:nusers
            H_eff{use} = H(:,:,use);
        end
        G = cell(1,K);
        for i = 2:K
            Norm_temp = 0;
            H_eff_bak = H_eff(S_set);
            for m = T_set
                B{m} = Q;
                H_eff{m} = H(:,:,m)*Q;
                Norm_pdt = real(prod(diag(H_eff{m}*H_eff{m}')));
                for k = 1:length(S_set)
                    mat_use = H(:,:,m)*B{S_set(k)};
                    [Qr,~] = qr(mat_use');
                    G{k} = Qr(:,n_r+1:end);
                    H_eff{S_set(k)} = H_eff{S_set(k)}*G{k};
                    Norm_pdt = Norm_pdt * real(prod(diag(H_eff{S_set(k)}*...
                        H_eff{S_set(k)}')));
                end
                if Norm_pdt > Norm_temp
                    Norm_temp = Norm_pdt;
                    G_un = G;
                    H_eff_max_bak = H_eff(S_set);
                    maxValInd = m;
                end
                H_eff(S_set) = H_eff_bak;
            end
            H_eff(S_set) = H_eff_max_bak;
            for k = 1:length(S_set)
                B{S_set(k)} = B{S_set(k)} * G_un{k};
            end
            B{maxValInd} = Q;
            mat_use = H(:,:,maxValInd)*Q;
            [Qr,~] = qr(mat_use');
            Q = Q * Qr(:,n_r+1:end);
            T_set(T_set == maxValInd) = [];
            S_set = [S_set,maxValInd];
        end
%         H = H_eff(S_set);
        H = H(:,:,S_set);
        S_set = 1:K;
        
        G_set = [];
        Ctemp = 0;
        for i = 1:K
            R_temp = 0;
            for k = S_set
                G_temp = [G_set,k];
%                 R_S = SumCapacityOfUserSet_F_iter(G_temp, H, SNR);
                R_S = SumCapacityOfUserSet_opt(G_temp, H, SNR,n_t,n_r);
                if R_S > R_temp
                    R_temp = R_S;
                    maxValInd = k;
                end
            end
            if R_temp < Ctemp
                break;
            else
                Ctemp = R_temp;
                G_set = [G_set,maxValInd];
                S_set(S_set == maxValInd) = [];
            end
        end
        if Ctemp > R_final(avg_sim_ind)
            R_final(avg_sim_ind) = Ctemp;
        end
    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 
fprintf('Total time taken = %f\n', sum(time));


function R_S = SumCapacityOfUserSet_F_iter(comb, H, SNR)

set_card = length(comb);
[m,n] = size(H{1});
H_blkD = zeros(m*set_card,n*set_card);
sz_r = 1;
sz_t = 1;
% lambda = zeros(set_card*m,1);
% sz = 1;
for k = comb
%     lambda(sz:sz+m-1) = eig(H{k}*H{k}');
%     sz = sz + m;
    H_blkD(sz_r:sz_r+m-1,sz_t:sz_t+n-1) = H{k};
    sz_r = sz_r + m;
    sz_t = sz_t + n;
end
% pl = Water_filling(lambda,SNR);
% R_S = sum(log2(1 + pl .* lambda));
R_S = Sum_Capacity(H_blkD, SNR);


function R_F = SumCapacity_CapacityUpperbound(n_t,n_r,SNR,users_num)

R_F = zeros(1, length(users_num));
time = zeros(1,length(users_num));
for mi = 1:length(users_num)
    nusers = users_num(mi);
    R_final = zeros(1000,1);    
    str = [' Channel_Mat/','H_all_',num2str(n_t),'_',num2str(n_r),'_', num2str(nusers), '.mat'];
    eval(['load' str]);
    tic
    for avg_sim_ind = 1:1000
%         H = (randn(n_r, n_t, nusers) + sqrt(-1) * randn(n_r, n_t, nusers)) / sqrt(2);
        H = H_all(:,:,:,avg_sim_ind);
        K = ceil(n_t / n_r);
        if K > nusers, K = nusers;end
        T_set = 1:nusers;
        S_set = [];
        ohm = SNR / n_t * eye(n_t);
        R = 0;
        
        for i = 1:K
            det_temp = 0;
            for t = T_set
                det_now = det(eye(n_r) + H(:,:,t) * ohm * H(:,:,t)');
                if det_now > det_temp
                    det_temp = det_now;
                    maxValInd = t;
                end
            end
            Stemp_set = [S_set, maxValInd];
            R_temp = SumCapacityOfUserSet(Stemp_set, H, SNR, n_t, n_r);
            if R_temp < R
                break;
            else
                S_set = Stemp_set;
                R = R_temp;
                T_set(T_set == maxValInd) = [];
            end
            ohm = ohm - ohm * H(:,:,maxValInd)' *...
                (eye(n_r) + H(:,:,maxValInd)*ohm*H(:,:,maxValInd)')^-1 *...
                H(:,:,maxValInd)*ohm;
        end
        R_final(avg_sim_ind) = R;
    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 
fprintf('Total time taken = %f\n', sum(time));


function R_F = SumCapacity_Chordal(n_t, n_r, SNR, users_num)

R_F = zeros(1, length(users_num));
time = zeros(1, length(users_num));
alpha = input('Input the value of alpha: ', 's');
[alpha,status] = str2num(alpha); %#ok<*ST2NM>
if ~status
    fprintf('Wrong value input, program will terminate!\n');
    return
end

for mi = 1:length(users_num)
    nusers = users_num(mi);
    R_final = zeros(1000,1);    
    str = [' Channel_Mat/','H_all_',num2str(n_t),'_',num2str(n_r),'_', num2str(nusers), '.mat'];
    eval(['load' str]);
    K = ceil(n_t / n_r);
    if K > nusers, K = nusers;end
    tic
    for avg_sim_ind = 1:1000
%         H = (randn(n_r, n_t, nusers) + sqrt(-1) * randn(n_r, n_t, nusers)) / sqrt(2);
        H = H_all(:,:,:,avg_sim_ind);
        
        T_set = 1:nusers;
        FrobTemp = 0;
        FrobNormC = zeros(length(T_set),1);
        for i = T_set
            FrobNormC(i) = trace(H(:,:,i) * H(:,:,i)');
            if FrobNormC(i) > FrobTemp
                FrobTemp = FrobNormC(i);
                maxValInd = i;
            end
        end
        T_set(T_set == maxValInd) = [];
        S_set = maxValInd;
        G = zeros(n_t,n_r,nusers);
        for i = 1:nusers
            G(:,:,i) = H(:,:,i).';
        end
        P_o = [];R = 0;
        for i = 2:K
            P_o = GSO([P_o, G(:,:,S_set(i-1))]);
            A = P_o * P_o';
            FrobTemp = 0;
            for k = T_set
                G_o = GSO(G(:,:,k));
                FrobMat = A - (G_o * G_o');
%                 FrobNorm = (FrobNormC(k) ^ alpha) * ...
%                     trace(FrobMat * FrobMat');
                FrobNorm = trace(FrobMat * FrobMat');
                if FrobNorm > FrobTemp
                    FrobTemp = FrobNorm;
                    maxValInd = k;
                end
            end
            Stemp_set = [S_set, maxValInd];
            R_temp = SumCapacityOfUserSet(Stemp_set, H, SNR, n_t, n_r);
            if R_temp < R
                break;
            else
                S_set = Stemp_set;
                R = R_temp;
                T_set(T_set == maxValInd) = [];
            end
        end
        R_final(avg_sim_ind) = R;
    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 
fprintf('Total time taken = %f\n', sum(time));


function R_F = SumCapacity_BDFrobenius(n_t, n_r, SNR, users_num)

R_F = zeros(1, length(users_num));
time = zeros(1, length(users_num));
for mi = 1:length(users_num)
    nusers = users_num(mi);
    R_final = zeros(1000,1);    
    str = [' Channel_Mat/','H_all_',num2str(n_t),'_',num2str(n_r),'_', num2str(nusers), '.mat'];
    eval(['load' str]);
% % % % % % % % % % % % % % CORRELATION CODE % % % % % % % % % % % % % % % % 
%     rho_m = 0.5;
%     R_Tx = zeros(n_t);
%     for sim = 1:1000
%         for use = 1:nusers
% %             H_tmp = R_kron * G(:,:,j,i);
% %             H_all(:,:,j,i) = reshape(H_tmp,n_r,n_t);
%             rho_phi = exp(1i*2*pi*rand);
%             for i = 1:n_t
%                 for j = 1:n_t
%                     R_Tx(i,j) = rho_m^abs(i-j) * rho_phi^(i-j);
%                 end
%             end
%             R = sqrtm(R_Tx);
% %             R = chol(R_Tx,'lower');
%             H_all(:,:,use,sim) = H_all(:,:,use,sim)*R;
%         end
%     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    tic
    for avg_sim_ind = 1:1000    
%         H = (randn(n_r, n_t, nusers) + sqrt(-1) * randn(n_r, n_t, nusers)) / sqrt(2);
        H = H_all(:,:,:,avg_sim_ind);
        K = ceil(n_t / n_r);
        if K > nusers, K = nusers;end
        FrobTemp = 0;
        O_set = 1:nusers;
        for i = 1:nusers
            FrobVal = trace(H(:,:,i) * H(:,:,i)');
            if FrobVal > FrobTemp
                FrobTemp = FrobVal;
                maxValInd = i;
            end
        end
        O_set(O_set == maxValInd) = [];
        G_set = maxValInd;
        Q = GSO(H(:,:,G_set(1)).');
        V = Q.';
        
        for i = 2:K
            F_Temp = 0;
            for k = O_set
                Hk_tilda = H(:,:,k) - H(:,:,k) * (V') * V;
                FrobNorm = trace(Hk_tilda * Hk_tilda');
                H_cap = zeros(n_r*(i-1), n_t);
                for s = G_set
                    count = 1;
                    for g = G_set
                        if g == s, continue;end
                        H_cap(n_r*(count-1)+1:n_r*count,:) = H(:,:,g);
                        count = count + 1;
                    end
                    H_cap(end-n_r+1:end,:) = H(:,:,k);
                    Q = GSO(H_cap.');
                    W = Q.';
                    Hs_tilda = H(:,:,s) - H(:,:,s) * (W') * W;
                    FrobNorm = FrobNorm + trace(Hs_tilda*Hs_tilda');
                end
                if FrobNorm > F_Temp
                    F_Temp = FrobNorm;
                    maxValInd = k;
                end
            end
            O_set(O_set == maxValInd) = [];
            G_set = [G_set,maxValInd];
            Q = GSO(H(:,:,maxValInd).');
            Vs = Q.';
            V = [V;Vs];
        end

        Ctemp = 0;
        S_set = [];
        for i = 1:K
            R_temp = 0;
            for k = G_set
                S_temp = [S_set,k];
                R_S = SumCapacityOfUserSet_opt(S_temp, H, SNR, n_t, n_r);
                if R_S > R_temp
                    R_temp = R_S;
                    maxValInd = k;
                end
            end
            if R_temp < Ctemp
                break;
            else
                Ctemp = R_temp;
                S_set = [S_set,maxValInd];
                G_set(G_set == maxValInd) = [];
            end
        end
        if Ctemp > R_final(avg_sim_ind)
            R_final(avg_sim_ind) = Ctemp;
        end
    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 
fprintf('Total time taken = %f\n', sum(time));


function R_F = SumCapacity_BDCapacity(n_t, n_r, SNR, users_num)

R_F = zeros(1, length(users_num));
time = zeros(1, length(users_num));
N_sim = 1000;

for mi = 1:length(users_num)
    nusers = users_num(mi);
    R_final = zeros(N_sim,1);    
    str = sprintf('Channel_Mat/H_all_%d_%d_%d',n_t,n_r,nusers);
    eval(['load ' str]);
% % % % % % % % % % % % % % CORRELATION CODE % % % % % % % % % % % % % % % % 
%     rho_m = 0.95;
%     R_Tx = zeros(n_t);
%     for sim = 1:1000
%         for use = 1:nusers
% %             H_tmp = R_kron * G(:,:,j,i);
% %             H_all(:,:,j,i) = reshape(H_tmp,n_r,n_t);
%             rho_phi = exp(1i*2*pi*rand);
%             for i = 1:n_t
%                 for j = 1:n_t
%                     R_Tx(i,j) = rho_m^abs(i-j) * rho_phi^(i-j);
%                 end
%             end
%             R = sqrtm(R_Tx);
% %             R = chol(R_Tx,'lower');
%             H_all(:,:,use,sim) = H_all(:,:,use,sim)*R;
%         end
%     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    
    tic
    for avg_sim_ind = 1:N_sim
        Ctemp = 0;
%         H = (randn(n_r, n_t, nusers) + sqrt(-1) * randn(n_r, n_t, nusers)) / sqrt(2);
        H = H_all(:,:,:,avg_sim_ind); %#ok<*NODEF>
        K = ceil(n_t / n_r);
        if K > nusers, K = nusers;end
        
        T_set = 1:nusers;
        G_set = [];
        for i = 1:K
            R_temp = 0;
            for k = T_set
                G_temp = [G_set,k];
%                 R_S = SumCapacityOfUserSet(G_temp, H, SNR, n_t, n_r);
                R_S = SumCapacityOfUserSet_opt(G_temp, H, SNR, n_t, n_r);
                if R_S > R_temp
                    R_temp = R_S;
                    maxValInd = k;
                end
            end
            if R_temp < Ctemp
                break;
            else
                Ctemp = R_temp;
                G_set = [G_set,maxValInd];
                T_set(T_set == maxValInd) = [];
            end
        end
        if Ctemp > R_final(avg_sim_ind)
            R_final(avg_sim_ind) = Ctemp;
        end
    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 


function R_F = SumCapacity_DSearch(n_t, n_r, SNR, users_num)

R_F = zeros(1, length(users_num));
time = zeros(1, length(users_num));
for mi = 1:length(users_num)
    nusers = users_num(mi);
    N_sim = 1000;
    R_final = zeros(N_sim,1);    
    str = [' Channel_Mat/','H_all_',num2str(n_t),'_',num2str(n_r),'_', num2str(nusers), '.mat'];
    eval(['load' str]);
%     K = ceil(n_t / n_r);
%     if K > nusers, K = nusers;end;
    tic
    for avg_sim_ind = 1:N_sim
        
%         H = (randn(n_r, n_t, nusers) + sqrt(-1) * randn(n_r, n_t, nusers)) / sqrt(2);
        H = H_all(:,:,:,avg_sim_ind);
        
        
        T_set = 1:nusers;
        R_temp1 = 0;
        FrobTemp = 0;
        for i = T_set
            FrobNorm = trace(H(:,:,i)*H(:,:,i)');
            if FrobNorm > FrobTemp
                FrobTemp = FrobNorm;
                maxValInd = i;
            end
        end
        T_set(T_set == maxValInd) =[];
        G_set = maxValInd;
        
        for k = T_set
            G_temp = [G_set,k];
            R_S = SumCapacityOfUserSet(G_temp, H, SNR, n_t, n_r);
            if R_S > R_temp1
                R_temp1 = R_S;
                maxValInd = k;
            end
        end
        
        R_temp2 = 0;
        FrobTemp = 0;
        for j = T_set
            FrobNorm = trace(H(:,:,j)*H(:,:,j)');
            if FrobNorm > FrobTemp
                FrobTemp = FrobNorm;
                maxValInd = j;
            end
        end
        T_set(T_set == maxValInd) =[];
        G_set = maxValInd;
        
        for k = T_set
            G_temp = [G_set,k];
            R_S = SumCapacityOfUserSet(G_temp, H, SNR, n_t, n_r);
            if R_S > R_temp2
                R_temp2 = R_S;
                maxValInd = k;
            end
        end
        
        R_final(avg_sim_ind) = max(R_temp1, R_temp2);

    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 
fprintf('Total time taken = %f\n', sum(time));


function R_F = SumCapacity_BDOptimal(n_t, n_r, SNR, users_num)

R_F = zeros(1, length(users_num));
time = zeros(1, length(users_num));
for mi = 1: length(users_num)
    
    nusers = users_num(mi);
    R_final = zeros(1000,1);    
    str = [' Channel_Mat/','H_all_',num2str(n_t),'_',num2str(n_r),'_', num2str(nusers), '.mat'];
    eval(['load' str]);
    tic
    for avg_sim_ind = 1:1000
        
%         H = (randn(n_r, n_t, nusers) + sqrt(-1) * randn(n_r, n_t, nusers)) / sqrt(2);
        H = H_all(:,:,:,avg_sim_ind);
        K = ceil(n_t / n_r);
        if K > nusers, K = nusers;end
        for i = 1:K    
            comb = combntns(1:nusers, i);
            for j = 1:size(comb,1)
                R_S = SumCapacityOfUserSet_opt(comb(j,:), H, SNR, n_t, n_r);
                if R_S > R_final(avg_sim_ind), R_final(avg_sim_ind) = R_S;end
            end
        end
    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 
fprintf('Total time taken = %f\n', sum(time));


function R_F = SumCapacity_BDOptimal_SZF(n_t, n_r, SNR, users_num)

R_F = zeros(1, length(users_num));
time = zeros(1, length(users_num));
for mi = 1: length(users_num)
    
    nusers = users_num(mi);
    R_final = zeros(1000,1);    
    str = [' Channel_Mat/','H_all_',num2str(n_t),'_',num2str(n_r),'_', num2str(nusers), '.mat'];
    eval(['load' str]);
    tic
    for avg_sim_ind = 1:1000
        
%         H = (randn(n_r, n_t, nusers) + sqrt(-1) * randn(n_r, n_t, nusers)) / sqrt(2);
        H = H_all(:,:,:,avg_sim_ind);
        K = ceil(n_t / n_r);
        if K > nusers, K = nusers;end
        for i = 1:K    
            comb = combntns(1:nusers, i);
            for j = 1:size(comb,1)
                perm = perms(comb(j,:));
                for g = 1:size(perm,1)
%                     R_S = SumCapacityOfUserSet_SZF(perm(g,:), H, SNR, n_t, n_r);
                    R_S = SumCapacityOfUserSet_DPC(perm(g,:), H, SNR, n_t, n_r);
                    if R_S > R_final(avg_sim_ind), R_final(avg_sim_ind) = R_S;end
                end
            end
        end
    end
    R_F(mi) = mean(R_final);
    time(mi) = toc;
    fprintf('Simulation for no. of users = %d done time taken = %f\n', users_num(mi), time(mi));
end 
fprintf('Total time taken = %f\n', sum(time));


function R_S = SumCapacityOfUserSet_SZF(comb, H, SNR, n_t, n_r)

R_S = 0;
set_card = length(comb);
P = getCovMat(comb, H, SNR, set_card, n_t, n_r);
Q_bar = getMACToBCCovMat(comb, P, H, set_card, n_t, n_r);
Q = getFinalCovMat(comb, Q_bar, H, SNR, set_card, n_t, n_r);

for j = 1:set_card
    mat = zeros(n_t,n_t);
    for i = 1:j-1
        mat = mat + Q(:,:,i);
    end
    mat_num = mat + Q(:,:,j);
    num = real(det(eye(n_r) + H(:,:,comb(j))*mat_num*H(:,:,comb(j))'));
    den = real(det(eye(n_r) + H(:,:,comb(j))*mat*H(:,:,comb(j))'));
    R_S = R_S + log2(num/den);
end


function Q = getFinalCovMat(comb, Q_bar, H, Pow, K, n_t, n_r)

V_bar_indM1 = cell(1,K);
Q = zeros(n_t,n_t,K);
V_bar_indM1{1} = eye(n_t);
Q(:,:,1) = Q_bar(:,:,1);
for j = 2:K
    H_bar_indM1 = zeros(n_r*(j-1),n_t);
    for i = 1:j-1
        H_bar_indM1(n_r*(i-1)+1:n_r*i,:) = H(:,:,comb(i));
    end
    [~,G,V] = svd(H_bar_indM1);
    rank = length(find(G));
    V_bar_indM1{j} = V(:,rank+1:end);
    if j == K, break;end
    Q(:,:,j) = V_bar_indM1{j}*V_bar_indM1{j}'*Q_bar(:,:,j)*V_bar_indM1{j}...
    *V_bar_indM1{j}';
end

mat = zeros(n_t,n_t);
for i = 1:K-1
    mat = mat + Q(:,:,i);
end
H_eff = ((eye(n_r) + H(:,:,comb(K))*mat*H(:,:,comb(K))')^-0.5) * ...
    H(:,:,comb(K))*V_bar_indM1{K}*V_bar_indM1{K}';

P = Pow;
for i = 1:K-1
    P = P - real(trace(Q(:,:,i)));
end
[V,D] = eig(H_eff'*H_eff);
lambda = diag(D);
pl = Water_filling(lambda, P);
Q_bar(:,:,K) = V*diag(pl)*V';
Q(:,:,K) = V_bar_indM1{K}*V_bar_indM1{K}'*Q_bar(:,:,K)*...
    V_bar_indM1{K}*V_bar_indM1{K}';


function Q_bar = getMACToBCCovMat(comb, P, H, K, n_t, n_r)

Q_bar = zeros(n_t,n_t,K);

for j = 1:K
    mat = zeros(n_t,n_t);
    for l = 1:j-1
        mat = mat + Q_bar(:,:,l);
    end
    
    D = eye(n_r) + H(:,:,comb(j))*mat*H(:,:,comb(j))';
    E = eye(n_t);
    for l = j+1 : K
        E = E + H(:,:,comb(l))' * P(:,:,l) * H(:,:,comb(l));
    end
    [F,~,G] = svd(E^-0.5 * H(:,:,comb(j))' * D^-0.5, 0);
    Q_bar(:,:,j) = (E^-0.5)*F*(G')*(D^0.5)*P(:,:,j)*(D^0.5)*G*(F')*(E^-0.5);
end


function P = getCovMat(comb,H, Pow, K, n_t, n_r)

G = zeros(n_r,n_t,K);
P = zeros(n_r,n_r,K);
for j = 1:K
    P(:,:,j) = eye(n_r)*Pow/(K*n_r);
end
for n = 1:75
    for j = 1:K
        mat = eye(n_t);
        for i = 1:K
            if i == j, continue;end
            mat = mat + H(:,:,comb(i))' *P(:,:,i)* H(:,:,comb(i));
        end
        G(:,:,j) = H(:,:,comb(j)) * (mat^-0.5);
    end
    S = getWaterFillCovMat(G,n_r,K, Pow);
    P_nM1 = P;
    MSE = zeros(1,K);
    for j = 1:K
        if K <= 2
            P(:,:,j) = S(:,:,j);
        else
            P(:,:,j) = 1/K * S(:,:,j) + (K-1)/K * P(:,:,j);
        end
        MSE(j) = sum(sum((P(:,:,j) - P_nM1(:,:,j)).^2))/(n_t*n_r);
    end
    if sum(MSE)/K < 1e-7, break;end
end


function S = getWaterFillCovMat(G,n_r,K, Pow)

S = zeros(n_r,n_r,K);
lambda = zeros(n_r*K,1);
for i = 1:K
    lambda((i-1)*n_r+1:i*n_r,1) = real(eig(G(:,:,i)*G(:,:,i)'));
end

pl = Water_filling(lambda, Pow);

for i = 1:K
    [U,~] = eig(G(:,:,i)*G(:,:,i)');
    Lam = diag(pl((i-1)*n_r+1:i*n_r));
    S(:,:,i) = U*Lam*U';
end

function R_S = SumCapacityOfUserSet_DPC(comb, H, SNR, n_t, n_r)

set_card = length(comb);
P = getCovMat(comb, H, SNR, set_card, n_t, n_r);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% Note: While solving dual MAC problem it is assumed that decoding order is
% 1,2,...,K while the eqn for BC capacity (3) in "Duality, Achievable Rates,
% and Sum-Rate Capacity of Gaussian MIMO Broadcast Channels" is written
% assuming encoding order 1,2,...,K. Therefore here after solving dual
% problem the encoding order will reverse i.e. in BC K,K-1,..,1 and the
% capacity eqn will change as below:

% Q = getMACToBCCovMat(comb, P, H, set_card, n_t, n_r);
% for j = 1:set_card
%     mat = zeros(n_t,n_t);
%     for i = 1:j-1
%         mat = mat + Q(:,:,i);
%     end
%     mat_num = mat + Q(:,:,j);
%     num = real(det(eye(n_r) + H(:,:,comb(j))*mat_num*H(:,:,comb(j))'));
%     den = real(det(eye(n_r) + H(:,:,comb(j))*mat*H(:,:,comb(j))'));
%     R_S = R_S + log2(num/den);
% end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

mat = eye(n_t);
for j = 1:set_card
    mat = mat + H(:,:,comb(j))'*P(:,:,j)*H(:,:,comb(j));
end
R_S = log2(real(det(mat)));



function R_S = SumCapacityOfUserSet(comb, H, SNR, n_t, n_r)

R_S = 0;
set_card = length(comb);

for k = comb
    H_final = H(:, :, k);
    if set_card~=1
        H_tilda = zeros(n_r * (set_card-1), n_t);
        count = 1;
        for l = comb
            if l == k, continue;end
                H_tilda((count-1)*n_r+1:count*n_r,:) = H(:,:,l);
                count = count + 1;
        end
        [~, G, V] = svd(H_tilda);
        rank = length(find(G));
        T = V(:,rank+1:end);
        H_final = H(:,:,k) * T;  % H_bar
    end
    SNR_final = SNR / set_card;
    cwf = Sum_Capacity(H_final, SNR_final);   
    R_S = R_S + cwf;
end


function R_S = SumCapacityOfUserSet_opt(comb, H, SNR, n_t, n_r)

set_card = length(comb);
H_blkD = zeros(n_r*set_card,n_t*set_card);
sz_r = 1;
sz_t = 1;
lambda = [];
for k = comb
    H_final = H(:, :, k);
    if set_card~=1
        H_tilda = zeros(n_r * (set_card-1), n_t);
        count = 1;
        for l = comb
            if l == k, continue;end
                H_tilda((count-1)*n_r+1:count*n_r,:) = H(:,:,l);
                count = count + 1;
        end
        [~, G, V] = svd(H_tilda);
        rank = length(find(G));
        T = V(:,rank+1:end);
        H_final = H(:,:,k) * T;  % H_bar
    end
    H_blkD(sz_r:sz_r+n_r-1,sz_t:sz_t+size(H_final,2)-1) = H_final;
    sz_r = sz_r + n_r;
    sz_t = sz_t + size(H_final,2);
    lambda = [lambda ; eig(H_final*H_final')];
end
H_blkD(:,sz_t:end) = [];
% R_S = Sum_Capacity(H_blkD, SNR);   
R_S = Sum_Capacity_tmp(lambda, SNR);   


function cwf = Sum_Capacity_tmp(lambda, P)

% global data_gg
% lambda = eig(H * H');
% lambda = lambda(find(lambda > 0));      % ignoring non-positive eigenvalues
pl = Water_filling(lambda, P);
% data_gg = [data_gg ; {pl'}];
% lambda = lambda(1:length(pl));
cwf = sum(log2(1 + pl .* lambda));


function cwf = Sum_Capacity(H, P)

% global data_gg
lambda = eig(H * H');
% lambda = lambda(find(lambda > 0));      % ignoring non-positive eigenvalues
pl = Water_filling(lambda, P);
% data_gg = [data_gg ; {pl'}];
% lambda = lambda(1:length(pl));
cwf = sum(log2(1 + pl .* lambda));


function R_S = SumCapacityForJMode(H, R, T, SNR, G_set, L_set, n_r)

K = length(G_set);
lambda = zeros(n_r*K,1);
sz = 1;
for i = 1:K
    mat = R{i}'*H(:,:,G_set(i))*T{i};
    lambda(sz:sz+L_set(i)-1,1) = real(eig(mat*mat'));
    sz = sz + L_set(i);
end
lambda(sz:end) = [];
pl = Water_filling(lambda, SNR);
R_S = sum(log2(1 + pl .* lambda));


function [R, T] = getPostPreMatrices(H, G_set, L_set, n_t, n_r)

K = length(G_set);
if K == 1
    disp('Single user, optimization can''t be done');
    return;
end
T = cell(1,K);
R = cell(1,K);
for i = 1:K
    R{i} = zeros(n_r,L_set(i));
    R{i}(1:L_set(i), :) = eye(L_set(i));
end
Q = cell(1,K);
He_t_Te_temp = 0;
count_gg = 0;
while(1)
    
    for k = 1:K
        H_e_m = zeros(n_r*(K-1), n_t);
        sz = 1;
        for l = 1:K
            if G_set(l) == G_set(k), continue;end
            H_e_m(sz:sz+size(R{l},2)-1,:) = R{l}'*H(:,:,G_set(l));
            sz = sz + size(R{l},2);
        end
        H_e_m(sz:end,:) = [];
        [~, G, V] = svd(H_e_m);
        rank = length(find(G));
        Q{k} = V(:,rank+1:end); 
    end
    for k = 1:K
        [U, ~, V] = svd(H(:,:,G_set(k))*Q{k});
        R{k} = U(:,1:L_set(k));
        T{k} = Q{k} * V(:,1:L_set(k));
    end
    H_e = zeros(n_r*K, n_t);
    T_e = zeros(n_t, n_r*K);
    szH = 1;
    szT = 1;
    for k = 1:K    
        H_e(szH:szH+size(R{k},2)-1,:) = R{k}'*H(:,:,G_set(k));
        T_e(:,szT:szT+size(T{k},2)-1) = T{k};
        szH = szH + size(R{k},2);
        szT = szT + size(T{k},2);
    end
    H_e(szH:end,:) = [];
    T_e(:,szT:end) = [];
%     err = trace(H_e*T_e*(T_e')*H_e');
    mat_use = diag(H_e*T_e);
    err = sum(abs(mat_use-He_t_Te_temp).^2);
%     err = sum(abs(diag(H_e*T_e)).^2);
%     abs(err-errp)
    if err <= 1e-12 || count_gg == 75,break;end
    He_t_Te_temp = mat_use;
    count_gg = count_gg + 1;
end


function [pl] = Water_filling(lambda, P)

[lambda idx] = sort(lambda, 'descend');
lambda = lambda(find(lambda > 1e-7));      % ignoring non-positive eigenvalues
pl = -1;
try
    while (min(pl) < 0)
        mu = (P + sum(1 ./ lambda)) / length(lambda);
        pl = mu - 1 ./ lambda;
        lambda = lambda(1:end-1);
    end
catch %#ok<*CTCH>
    disp('There exists no water filling level for the input eigenvalues. Check your data and try again');
end
pl = [pl; zeros(length(idx) - length(pl), 1)]; % assigning zero power for weak eigen-modes
pl(idx) = pl; % rearranging the power levels


function[Q] =  GSO(A)

[~,n] = size(A);
% compute QR using Gram-Schmidt
for j = 1:n
   v = A(:,j);
   for i=1:j-1
        R(i,j) = Q(:,i)'*A(:,j);
        v = v - R(i,j)*Q(:,i);
   end
   R(j,j) = norm(v);
   Q(:,j) = v/R(j,j);
end