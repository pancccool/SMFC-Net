function [BrainNetSet]=WSR(BOLD,lambda)
%% Note:
%  The Sparse Learning via Efficient Projection (SLEP) matlab toolbox is required.
%% Basic parameter 
nTime=size(BOLD{1},1);
nSubj=length(BOLD);
RegionNum=120;
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
opts.init=2;
opts.tFlag=0;
opts.nFlag=0;
opts.rFlag=0;
opts.rsL2=0;
opts.mFlag=0;
opts.lFlag=0;
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

