% 根据IDM()函数改变，用于计算一个驾驶员的数据的目标函数值

function [sse,sseV,numOfCollision,numOfBack,svSpd,lvSpd,space,svSpd_sim,space_sim] = IDMOneDriver(x,carFolEventInfo,calibrateSeq)

desiredSpd = x(1); %in m/s
desiredTimeHdw = x(2); % in seconds
maxAcc = x(3); % m/s^2
comfortAcc = x(4); % m/s^2
beta =x(5);
jamSpace = x(6); % in meters

% collsion penalty
Penalty = 1e6;%1e6;  %当space <= 0时，进行惩罚

%构造临时变量，存储每步得到的结果，累加便于计算最后的误差指标
%sse = sse+sqrt(sum((space-space_sim).^2)/sum(space.^2)) +sqrt(sum((svSpd-svSpd_sim).^2)/sum(svSpd.^2));
space_temp=0;
spaceDiff_temp=0;
svSpd_temp =0;
svSpdDiff_temp=0;
numOfCollision=0;
numOfBack = 0;

%对该驾驶员的片段进行逐个计算
for kk = 1:size(calibrateSeq,1)
    
    %data structure
    % smsstring = 'select [vtti_timestamp],[SMS_Object_ID_T0],[SMS_X_Velocity_T0],[SMS_X_Range_T0],[SMS_Y_Range_T0],[FOT_Control_Speed],[System_video_frame],[Head_Unit_Time_Of_Day],[IMU_Accel_X]';
    
    % get data of a specific car-following event
    carFolID = calibrateSeq(kk); %49
    data =carFolEventInfo{carFolID};
    % newly added on 2020.11.19
    MAX_LEN = 200;
    len = min(MAX_LEN, size(data, 1));
    data = data(1:len,:);
    
    space = data(:,1);
    svSpd =data(:,2);
    lvSpd = data(:,3);%FilterZmx(data(:,3), 10);
%     lvSpd = FilterZmx(lvSpd, 15);
    xSv_data = data(:,4);
    xLv =data(:,5);
    
    % 进行后车加速度推导
    % 初始化
    xSv_sim =xSv_data;
    svSpd_sim = svSpd;
    
    % RT = 1;
    sampleInteval = 0.1; %采样间隔，一般为0.1s
    shiftIndex = 0;
    
    %循环推导
    num = size(space,1);
    
    isCollision=0; %碰撞
    isBackward=0;  %后车倒退
    
    for i =shiftIndex+1:num-1
        %GHR model format:  acc_x(t) = a*deltaV(t-RT)*(svSpd(t))^z /(space(t-RT))^l  a,l,z is parameters that requires calibration
        deltaV = lvSpd(i-shiftIndex) -svSpd_sim(i-shiftIndex); % 此处有风险，前提是： 总是按0.1 s的间隔进行采样， 后续改进
        deltaS = xLv(i-shiftIndex) - xSv_sim(i-shiftIndex);
        
        if deltaS<=0
            numOfCollision=numOfCollision+1;
            isCollision =1;
            break;
        end
        
        velocity =svSpd_sim(i);
        
        %***** IDM 模型******
        % acc = maxAcc(1-(v/vDesiered)^beta-(spaceDesired/space)^2 )
        % spaceDesired = SpaceJam +
        % v*DesirdHdw-v*deltaV/(2*sqrt(maxAcc*comfortableAcc));
        
        desiredSpace = jamSpace+max(0,desiredTimeHdw*velocity-velocity*deltaV/(2*sqrt(maxAcc*comfortAcc)));
        acc = maxAcc*(1-(velocity/desiredSpd)^beta-(desiredSpace/deltaS)^2);
        
        %进行速度、位移反推
        svSpd_sim(i+1) = max(0.001, svSpd_sim(i)+sampleInteval*acc);
        
        xSv_sim(i+1) =xSv_sim(i) + sampleInteval*svSpd_sim(i);
    end
    space_sim = xLv - xSv_sim;
    
    %指标累加
    space_temp=space_temp+sum(space.^2);
    spaceDiff_temp=spaceDiff_temp+sum((space-space_sim).^2)+isCollision*Penalty; 
    svSpd_temp =svSpd_temp+sum(svSpd.^2);
    svSpdDiff_temp=svSpdDiff_temp+sum((svSpd-svSpd_sim).^2)+isCollision*Penalty;
end
sse = sqrt(spaceDiff_temp/space_temp);
sseV = sqrt(svSpdDiff_temp/svSpd_temp);
