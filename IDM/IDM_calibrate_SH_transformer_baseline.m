clear all

tic
options = gaoptimset('Generations',300,'StallGenLimit',100,'PopulationSize',300,'StallTest','geometricWeighted','UseParallel', true); %using parallel computing in the ga algorithm


train = load('train_SH_info.mat','carFolEventInfoTrans');
train_data = train.carFolEventInfoTrans;

val = load('val_SH_info.mat','carFolEventInfoTrans');
val_data = val.carFolEventInfoTrans;

calibrateSeq = (1:size(train_data,1))';
validateSeq = (1:size(val_data, 1))';
% [calibrateSeq,validateSeq, testingSeq] = selectEventForDriver(driverID);

f = @(x) IDMOneDriver(x,train_data,calibrateSeq);
%进行多次标定，缺误差最小的那次

totalCount = 4;
%IDM模型的上下限
%     desiredSpd = x(1); %in m/s  【1，150/3.6】
%     desiredTimeHdw = x(2); % in seconds  【0.1 ，5】
%     maxAcc = x(3); % m/s^2 【0.1 5】
%     comfortAcc = x(4); % m/s^2  【0.1 5】
%     beta =x(5);  【1 10】  %整数
%     jamSpace = x(6); % in meters  【0.1 10】
LB = [1 0.1 0.1 0.1 1 0.1];
UB = [150/3.6 5 5 5 10 10];

errMat = zeros(totalCount,1);
xMat = zeros(totalCount,6);

for i =1:totalCount

    [x,fval] = ga(f,6,[],[],[],[],...
        LB,UB,[],[5],options);

    xMat(i,:)=x;
    
    % validation part
    f2 = @(x) IDMOneDriver(x, val_data, validateSeq);
    validateErr = f2(x);
    errMat(i)= validateErr;
end
[minValidateErr,index]= min(errMat);
bestX = xMat(index,:);

% %testing part
% f3 = @(x) IDMOneDriver(x,carFolEventInfoTrans,testingSeq);
% testingErr = f3(bestX);

save(['IDM', '_SH_transformer.mat'],'bestX', 'minValidateErr', 'errMat', 'xMat');
    
% save('IDMAll.mat');

toc
% %发送结果到邮箱
%  mailTome('跟车模型验证',['The best X is ',num2str(bestX),' with error ',num2str(minErr)],'IDMAll.mat');

% % shut down
% % system('shutdown -s');