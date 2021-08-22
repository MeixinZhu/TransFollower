% transform data into format required by the GA calibration code, revised
% on 07/22/2019  

load('train_SH.mat', 'data');
carFolEventInfo = data;
carFolEventInfoTrans = cell(size(carFolEventInfo,2), 1);
%对该驾驶员的片段进行逐个计算
for kk = 1:size(carFolEventInfo,2)
% get data of a specific car-following event
data =carFolEventInfo{1,kk};
% data in the format [space, svSpd, relSpd, lvSpd]

dd = 150;%size(data, 1);

deltaV = smooth(data(:,3)); 
deltaV = deltaV(1:dd);

space = smooth(data(:,1));
space = space(1:dd);

svSpd = smooth(data(:,2));
svSpd = svSpd(1:dd);

lvSpd = svSpd + deltaV;
lvSpd = lvSpd(1:dd); % fliter has a minimum size of a

timeDiff = ones(dd-1,1)*0.1;

xSv_data = [0;cumsum(timeDiff.*svSpd(1:end-1))];
xLv = xSv_data+space;

carFolEventInfoTrans{kk,1}=[space,svSpd,lvSpd,xSv_data,xLv];
end

% save the mat
save('train_SH_info.mat','carFolEventInfoTrans');


