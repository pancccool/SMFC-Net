%% Note:
% The operation of WSR requires LeastR_Weight_YRP.m and sll_opts.m.
% sll_opts.m is from the Sparse Learning via Efficient Projection (SLEP) matlab toolbox.
% LeastR_Weight_YRP.m is modified from LeastR.m in the SLEP toolbox.
% The SLEP toolbox can be obtained from http://www.yelab.net/software/SLEP or https://github.com/jiayuzhou/SLEP

%% load BOLD signals
% load('data\boldsignals.mat')

% A random number matrix is generated to simulate the BOLD signal.
subjects = 125;
times = 140;
ROIs = 120;
boldsignals=cell(1,subjects);
for i = 1:subjects
    subject_every =rand(times,ROIs);
    boldsignals{1,i}=subject_every;
end

BOLD = boldsignals;
Min_param=-5;Interval=1;Max_param=5;
lambda=2.^[Min_param:Interval:Max_param];

%% Basic parameter 
nTime=size(BOLD{1},1);
nSubj=length(BOLD);
RegionNum=120;%size(BOLD{1},2);
%% Normalized data
Total_Data=zeros(nSubj,nTime,RegionNum);
Corr_Record=zeros(nSubj,RegionNum,RegionNum-1);
for SubjectID=1:nSubj
    R=zeros(RegionNum,RegionNum-1);
    tmp=BOLD{SubjectID};
    subject=tmp(:,1:RegionNum);
    [r,~]=corrcoef(subject);
    for j=1:RegionNum
        Index=1:RegionNum;
        Index(j)=[];
        R(j,:)=r(j,Index);
    end
    %% mean std (0 1)
    subject=subject-repmat(mean(subject),nTime,1);
    subject=subject./(repmat(std(subject),nTime,1));
    
    Total_Data(SubjectID,:,:)=subject;
    Corr_Record(SubjectID,1:RegionNum,1:RegionNum-1)=R;
    clear tmp;
end
%% Network construction
BrainNetSet=cell(length(lambda),1);
opts=[];
opts.init=2;% Starting point: starting from a zero point here
opts.tFlag=0;% termination criterion
% abs( funVal(i)- funVal(i-1) ) ¡Ü .tol=10e?4 (default)
%For the tFlag parameter which has 6 different termination criterion.
% 0 ? abs( funVal(i)- funVal(i-1) ) ¡Ü .tol.
% 1 ? abs( funVal(i)- funVal(i-1) ) ¡Ü .tol ¡Á max(funVal(i),1).
% 2 ? funVal(i) ¡Ü .tol.
% 3 ? kxi ? xi?1k2 ¡Ü .tol.
% 4 ? kxi ? xi?1k2 ¡Ü .tol ¡Á max(||xi||_2, 1).
% 5 ? Run the code for .maxIter iterations.
opts.nFlag=0;% normalization option: 0-without normalization
opts.rFlag=0;% regularization % the input parameter 'rho' is a ratio in (0, 1)
opts.rsL2=0; % the squared two norm term in min  1/2 || A x - y||^2 + 1/2 rsL2 * ||x||_2^2 + z * ||x||_1
%fprintf('\n mFlag=0, lFlag=0 \n');
opts.mFlag=0;% treating it as compositive function
opts.lFlag=0;% Nemirovski's line search
for SubjectID=1:nSubj
    r=zeros(RegionNum,RegionNum-1);
    r(:,:)=abs(Corr_Record(SubjectID,:,:));

    r_std=0.26;
    w=exp(-(r.^2)/r_std);
   %% weighted sparse representation
    for l1=1:size(lambda,2)
        param=lambda(l1);
        BrainNet=zeros(RegionNum,RegionNum);
        for j=1:RegionNum
            Index=1:RegionNum;
            Index(j)=[];
            Cube=zeros(nTime,RegionNum-1);
            Region=zeros(nTime,1);
            Cube(:,:)=Total_Data(SubjectID,:,Index);
            Region(:,1)=Total_Data(SubjectID,:,j);
            Weight=w(j,:);
            [x, ~, ~]= LeastR_Weight_YRP(Cube, Region, param, Weight',opts);
            BrainNet(j,Index)=x;
            clear  Weight x ;
        end
        BrainNetSet{l1,1}(SubjectID,:,:)=BrainNet;
        fprintf('Done the %d subject WSR networks with lamda1 equal to %d!\n',SubjectID,l1);
    end
end
save('data\BrainNetSet_SZ_WSR.mat','BrainNetSet');

