close all;

%Parameter && Data Initialization
m1Real = [7 12];
m2Real = [12 7];
m3Real = [14 15];
C1Real = [1 0;0 1];
C2Real = [3 1;1 3];
C3Real = [3 1;1 3];
priorProbReal = [0.5 0.2 0.3];
N = 5000;
X_1 = mvnrnd(m1Real,C1Real,N*priorProbReal(1));
X_2 = mvnrnd(m2Real,C2Real,N*priorProbReal(2));
X_3 = mvnrnd(m3Real,C3Real,N*priorProbReal(3));
X=[X_1;X_2;X_3];
X = X(randperm(size(X,1)),:);
%Parameter && Data Initialization

%plot the intial mixture of data clusters
subplot(4,3,1);
plot(X(:, 1), X(:, 2), 'bx');


%Randomize different prior parameters
PIk = [1/3 1/3 1/3];
%TRICK: m is supposed to be initialized differently for 3 distributions.
m = [max(X(:, 2)) / 4 + 3 * min(X(:, 2)) / 4 (min(X(:, 2)) + max(X(:, 2))) / 2];
m = [m; 2 * max(X(:, 2)) / 4 + 2 * min(X(:, 2)) / 4 (min(X(:, 2)) + max(X(:, 2))) / 2];
m = [m; 3 * max(X(:, 2)) / 4 + 1 * min(X(:, 2)) / 4 (min(X(:, 2)) + max(X(:, 2))) / 2];
C = [1 0;0 1];
C(:, :, 2) = [1 0;0 1];
C(:, :, 3) = [1 0;0 1];


iteration = 11;
for k = 1:iteration
    %%%E STEP%%%%%%%%%%%%%%%
    %calculate the posterior probability of X for each cluster
    for i = 1 : length(priorProbReal)
        numer(:, i) = PIk(i) * mvnpdf(X, m(i, :), C(:, :, i));
    end
    denom = sum(numer, 2);
    PostProb = bsxfun(@rdivide, numer, denom); % N X 3
    %%%E STEP%%%%%%%%%%%%%%%
    
    %%%M STEP%%%%%%%%%%%%%%%
    Nk = sum(PostProb, 1); % 1 X 3
    PIk = Nk / N;
    
    m = PostProb' * X; % 3 X 2
    m = m ./ repmat((sum(PostProb, 1))', 1, size(m, 2));

    for j = 1 : length(priorProbReal)
        %%for each class j,  cov = (x - m)' * (x - m)
        vari = repmat(PostProb(:, j), 1, size(X, 2)) .* (X - repmat(m(j, :), size(X, 1), 1)); 
        cov = vari' * vari;
        C(:, :, j) = cov / Nk(j);      
    end
    %%%M STEP%%%%%%%%%%%%%
    [foo estimate] = max(PostProb,[],2);
    one = find(estimate==1); 
    two = find(estimate == 2);
    three = find(estimate == 3);
    subplot(4,3,k+1) 
    plot(X(one, 1), X(one, 2), 'x',X(two, 1), X(two, 2), 'o', X(three, 1), X(three, 2), '<');
  
end

PIk
m
C
