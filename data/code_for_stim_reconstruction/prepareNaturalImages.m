function prepareNaturalImages()
%PREPARENATURALIMAGES 
% Generates a folder with .raw files withing the path that contains the
% images

%--------------------------------------------------------------------------
options.sqside  = 512; % change this number for images of different size
options.imgmean = 0.5;
options.imgstd  = 0.2;
%--------------------------------------------------------------------------
%select the images
fstr={'*.jpg;*.tif;*.imc','All allowed image files';...
    '*.jpg', 'Berkeley image files'; ...
    '*.tif', 'McGill image files';...
    '*.imc', 'vanHateren image files'};

[filenames, pathname]= uigetfile(fstr,'Select the images you want to convert',...
    'MultiSelect', 'on'); 

if isequal(filenames,0); error('No images selected! Program is terminating...');
else, disp('Images selected, proceeding to loading...'); end

filenames=cellstr(filenames);
%--------------------------------------------------------------------------
%create saving folder
rawFolderName = fullfile(pathname, 'rawFolder');
if ~exist(rawFolderName,'dir'); mkdir(rawFolderName); end
%--------------------------------------------------------------------------
for ii=1:numel(filenames)
    disp(['Loading and processing image ' filenames{ii} '...']);tic;
    
    imMat  = getProperGrayImg(fullfile(pathname, filenames{ii}), options.sqside);
    valMat = double(imMat)/255;    
    %--------------------------------------------------------------------------
    %Scaling and shifting
    reMat   = options.imgmean + (valMat-mean(valMat(:)))*options.imgstd/std(valMat(:));
    saveMat = imcast(reMat,'uint8');
    
    %save original
    name_raw = fullfile(rawFolderName, [strtok([ filenames{ii}], '.') '.raw']);
    fid = fopen(name_raw,'w+'); 
    fwrite(fid, saveMat, 'uint8'); 
    fclose(fid);
    %--------------------------------------------------------------------------
end
end
%--------------------------------------------------------------------------
function imgMat=getProperGrayImg(imgpath, tsize)

[~,~,ext] = fileparts(imgpath);

switch ext
case '.imc'; imtype='vanHateren';
case '.jpg'; imtype='Berkeley';
case '.tif'; imtype='McGill';
end

switch imtype
    case 'vanHateren'
        w = 1536; h = 1024;
        f1    = fopen(imgpath, 'rb', 'ieee-be');
        imRaw = fread(f1, [w, h], 'uint16');
        fclose(f1); clear f1; 
        
        imRaw = imRaw'/4095; 
        imRaw(imRaw>1) = 1;
        imRaw = imcast(imRaw,'uint8');
        ih = size(imRaw,1); iw = size(imRaw,2); ifin = 1000;
    otherwise
        imRaw = imread(imgpath);
        ih    = size(imRaw,1); 
        iw    = size(imRaw,2); 
        ifin  = min([ih iw]);
end
        imRawClipped = imRaw(round((ih-ifin)/2+1:(ih+ifin)/2),...
            round((iw-ifin)/2+1:(iw+ifin)/2), :);
        imRawResized = uint8(imageresize(imRawClipped, tsize/ifin));
        imgMat       = imRawResized(1:tsize, 1:tsize,:);
        if ndims(imgMat)>2; 
            imgMat=rgb2gray(imgMat); 
        end
end

