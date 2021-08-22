% ����IDM()�����ı䣬���ڼ���һ����ʻԱ�����ݵ�Ŀ�꺯��ֵ

function [sse,sseV,numOfCollision,numOfBack,svSpd,lvSpd,space,svSpd_sim,space_sim] = IDMOneDriver(x,carFolEventInfo,calibrateSeq)

desiredSpd = x(1); %in m/s
desiredTimeHdw = x(2); % in seconds
maxAcc = x(3); % m/s^2
comfortAcc = x(4); % m/s^2
beta =x(5);
jamSpace = x(6); % in meters

% collsion penalty
Penalty = 1e6;%1e6;  %��space <= 0ʱ�����гͷ�

%������ʱ�������洢ÿ���õ��Ľ�����ۼӱ��ڼ����������ָ��
%sse = sse+sqrt(sum((space-space_sim).^2)/sum(space.^2)) +sqrt(sum((svSpd-svSpd_sim).^2)/sum(svSpd.^2));
space_temp=0;
spaceDiff_temp=0;
svSpd_temp =0;
svSpdDiff_temp=0;
numOfCollision=0;
numOfBack = 0;

%�Ըü�ʻԱ��Ƭ�ν����������
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
    
    % ���к󳵼��ٶ��Ƶ�
    % ��ʼ��
    xSv_sim =xSv_data;
    svSpd_sim = svSpd;
    
    % RT = 1;
    sampleInteval = 0.1; %���������һ��Ϊ0.1s
    shiftIndex = 0;
    
    %ѭ���Ƶ�
    num = size(space,1);
    
    isCollision=0; %��ײ
    isBackward=0;  %�󳵵���
    
    for i =shiftIndex+1:num-1
        %GHR model format:  acc_x(t) = a*deltaV(t-RT)*(svSpd(t))^z /(space(t-RT))^l  a,l,z is parameters that requires calibration
        deltaV = lvSpd(i-shiftIndex) -svSpd_sim(i-shiftIndex); % �˴��з��գ�ǰ���ǣ� ���ǰ�0.1 s�ļ�����в����� �����Ľ�
        deltaS = xLv(i-shiftIndex) - xSv_sim(i-shiftIndex);
        
        if deltaS<=0
            numOfCollision=numOfCollision+1;
            isCollision =1;
            break;
        end
        
        velocity =svSpd_sim(i);
        
        %***** IDM ģ��******
        % acc = maxAcc(1-(v/vDesiered)^beta-(spaceDesired/space)^2 )
        % spaceDesired = SpaceJam +
        % v*DesirdHdw-v*deltaV/(2*sqrt(maxAcc*comfortableAcc));
        
        desiredSpace = jamSpace+max(0,desiredTimeHdw*velocity-velocity*deltaV/(2*sqrt(maxAcc*comfortAcc)));
        acc = maxAcc*(1-(velocity/desiredSpd)^beta-(desiredSpace/deltaS)^2);
        
        %�����ٶȡ�λ�Ʒ���
        svSpd_sim(i+1) = max(0.001, svSpd_sim(i)+sampleInteval*acc);
        
        xSv_sim(i+1) =xSv_sim(i) + sampleInteval*svSpd_sim(i);
    end
    space_sim = xLv - xSv_sim;
    
    %ָ���ۼ�
    space_temp=space_temp+sum(space.^2);
    spaceDiff_temp=spaceDiff_temp+sum((space-space_sim).^2)+isCollision*Penalty; 
    svSpd_temp =svSpd_temp+sum(svSpd.^2);
    svSpdDiff_temp=svSpdDiff_temp+sum((svSpd-svSpd_sim).^2)+isCollision*Penalty;
end
sse = sqrt(spaceDiff_temp/space_temp);
sseV = sqrt(svSpdDiff_temp/svSpd_temp);
