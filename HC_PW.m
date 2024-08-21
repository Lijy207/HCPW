function [preR,num,time] = HC_PW(X,Xtest,para)
%% HC_Piecewise weighting
% by Jiaye Li 2022-06-16
% X = NormalizeFea(X,0);
% Xtest = NormalizeFea(Xtest,0);
[n,d] = size(X);
[m,~] = size(Xtest);
alpha = para.alpha;
beta = para.beta;
miu = para.miu;
fea = para.fea;
kvalue = para.kvalue;
% lambda1 = para.lambda1;
lambda2 = para.lambda2;
d1 = ones(d,1);
b =rand(n,1);
I = eye(d);
lambda1 = zeros(d,1);
%% Initialization
W = rand(d,n);
theta = rand(d,1);
sumtheta = sum(theta);
theta = theta./sumtheta;
Theta = diag(theta);
v = theta;
iter = 1;
flagL = 'g';
switch flagL
    case 'g'
        options = [];
        options.NeighborMode = 'KNN';
        options.k = 5;
        options.WeightMode = 'HeatKernel';
        options.t = 1;
        S = constructW_xf(X,options);
        S = max(S,S');
        L = diag(sum(S,2)) - S;
    case 'h' 
        Weight = ones(n,1);
        options = [];
        options.NeighborMode = 'KNN';
        options.k = 5;
        options.WeightMode = 'HeatKernel';
        options.t = 1;
        options.bSelfConnected = 1;
        W = constructW_xf(X',options);   
        Dv = diag(sum(W)'); 
        De = diag(sum(W,2)); 
        invDe = inv(De);
        DV2 = full(Dv)^(-0.5);
        L = eye(n) - DV2 * W * diag(Weight) * invDe * W' *DV2;        
end

for j =1:d
    K(:,j) = X'*X(:,j);
end
while 1
%% Update W    
for i =1:n
    Dn{i} = 0.5./(sqrt(sum(W(:,i).*W(:,i),2)+eps));
    D{i} = diag(Dn{i});
    XT =X';
    W(:,i) = pinv(Theta*K*K'*Theta'+alpha*D{i}+beta*Theta*X'*L*X*Theta')*(Theta*K*XT(:,i)-Theta*K*d1*b(i));   
end 
%% Update b
b = (1/n)*(X*d1-W'*Theta*K*d1);

%% Update Theta
H = I-(1/d)*d1*d1';
P = (K*H*K')'.*(W*W');
Q = (X'*L*X)'.*(W*W');
R = diag(2*K*H*X'*W');
F = 2*P+2*beta*Q+miu*I+miu*d1*d1';
g = miu*v + miu*d1 - lambda2*d1 - lambda1 + R;
theta = pinv(F)*g;
Theta = diag(theta);
v = theta + (1/miu)*lambda1;
v(find(v < 0)) = 0;

term1 = norm(W'*Theta*K + b*d1' - X, 'fro')^2;
term2 = alpha*abs(sum(sum(W)));
term3 = beta*trace(W'*Theta*X'*L*X*Theta'*W);
obj(iter) = term1 + term2 + term3;

iter = iter+1;
if  iter == 21,    break,     end

end 
plot(obj)
L1 = mean(W,2);
% W(find(W<mean(W,2))) = 0;
WW = [];
for j=1:d
    Wj = W(j,:);
    Wj(find(Wj<L1(j))) = 0;
   Wj = Wj./sum(Wj);
    WW = [WW;Wj];
    clear Wj
end

similarity = [];
for j =1:d
    for i = 1:n
        
   for i1 = 1:n
       if i~=i1
     similarity(i,i1) = abs(WW(j,i) - WW(j,i1));  
       end
   end
    end
   Sim{j} = similarity;
     similarity = [];
end

    for j =1:d
        R1 = triu(Sim{j});
[idx1,idx2] = find(R1 <= 0.001 & R1~=0);
R1 = [];
    WW(j,idx1) = WW(j,idx1)+ WW(j,idx2);
WW(j,idx2) = WW(j,idx1);
    end




tic;
preR = Xtest;
num = 0;
for testi = 1:m

    [idx, dist] = knnsearch(X.*WW',Xtest(testi,:),'k',kvalue);
    if kvalue == 1
        minDistance1 = X(idx,:);
    else
        minDistance1 = mean(X(idx,:));
    end
    

    for j=fea:d
        if Xtest(testi,j)==0 & minDistance1(1,j)~=0
            num=num+1;
            preR(testi,j) = minDistance1(1,j);
        end
    end
    
end
time = toc;
end

