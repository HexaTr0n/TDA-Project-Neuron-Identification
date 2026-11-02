function [stimMat, cta, ctb, screenMat, spaceVecX, spaceVecY] =...
    findLocallySparseSubunitFlashOrder(nframes, stimdesc)
%FINDLOCALLYSPARSESUBUNITFLASHORDER 

Nx = stimdesc.Nx;
Ny = stimdesc.Ny;

screenNx=Nx*stimdesc.nstixelsx;
screenNy=Ny*stimdesc.nstixelsy;
%think of screen coordinates again
spaceVecX=stimdesc.lmargin+0.5+stimdesc.stixelwidth*(0:screenNx-1)+stimdesc.stixelwidth/2;
spaceVecY=stimdesc.bmargin+0.5+stimdesc.stixelheight*(0:screenNy-1)+stimdesc.stixelheight/2;

gapx=stimdesc.gapwidth; gapy=stimdesc.gapheight;
seedlocation=stimdesc.seedlocation;
seedcombination=stimdesc.seedcombination;

locationlist = (0:Nx*Ny-1)';
neighbourlist = getneighbourlist(Nx, Ny, gapx, gapy);
[cta, ctb] = genLists(stimdesc); ncombis=numel(cta);

stimMat=zeros(nframes, Ny*Nx); 

possibleXYlocs = zeros(floor(numel(locationlist)/min(cellfun(@numel,neighbourlist))),1);
%outside loop is faster
[randvals, ~]=ran1(seedlocation, nframes*numel(possibleXYlocs)); randIdx=1;

for ii = 1:nframes
    availablelist = locationlist;
    currXYlocs = possibleXYlocs;
    idx = 1;
    while numel(availablelist)>0
        
        randval=randvals(randIdx); randIdx=randIdx+1;
        stixelInd=floor(randval * size(availablelist,1))+1;
        toadd=availablelist(stixelInd);
        toremove = neighbourlist{Ny * mod(toadd,Nx) + floor(toadd/Nx)+1};
        
        %using the mex function for speed
        availablelist(ismembc(availablelist,sort(toremove))) = [];
        
        currXYlocs(idx) = toadd+1;
        idx = idx+1;
    end
    [randomvalues, seedcombination]=ran1(seedcombination, idx-1);
    
    stimMat(ii, currXYlocs(1:idx-1))=ceil(randomvalues*ncombis);
end
stimMat=reshape(stimMat, nframes, Nx, Ny);
stimMat=flip(permute(stimMat, [1 3 2]),2);
stimMat=sparse(reshape(stimMat,nframes,[]));
%stimMat=ndSparse(stimMat);

screenMat=zeros(nframes,screenNy, screenNx);

n=max([stimdesc.nstixelsy, stimdesc.nstixelsx]);
p = mod(1 : n, 2); C = bsxfun(@xor, p', p); 
C=C(1:stimdesc.nstixelsy, 1:stimdesc.nstixelsx);

for icon=1:numel(cta)
   
    [fInd,yInd,xInd]=ind2sub([nframes Ny Nx],find(stimMat==icon));
    
    stixelMat=zeros(stimdesc.nstixelsy, stimdesc.nstixelsx);
    
    stixelMat(~C)=ctb(icon);
    stixelMat(C)=cta(icon);

    for ii=1:numel(fInd)
        screenMat(fInd(ii), ...
            (yInd(ii)-1)*stimdesc.nstixelsy+(1:stimdesc.nstixelsy),...
            (xInd(ii)-1)*stimdesc.nstixelsx+(1:stimdesc.nstixelsx))=stixelMat;
    end
end

end


function neighbourlist = getneighbourlist(nx, ny, gapx, gapy)

neighbourlist = cell(nx*ny,1);
idx = 1;
for x = 0: nx-1
    for y = 0: ny-1
        localneigbour=[];
        for currX = x-gapx : x+gapx
            for currY = y-gapy : y+gapy
                if (currX >=0 && currY >= 0 )
                    localneigbour = [localneigbour, nx * currY + currX];    %#ok
                end
            end
        end
        neighbourlist{idx} = localneigbour;
        idx = idx+1;
    end
end

end


function [cta, ctb] = genLists(stimdesc)
    nAngles = stimdesc.nangles;
    minCon = stimdesc.mincontrast;
    maxCon = stimdesc.maxcontrast;
    nContrasts = stimdesc.ncontrasts;
    
    angles = (0:(nAngles-1)) * 2*pi/nAngles;
    cosAngs = cos(angles);
    sinAngs = sin(angles);
    
    radii = minCon + (0:(nContrasts-1)) * (maxCon - minCon)/(nContrasts-1);
    
    cta = zeros( sum(radii >= 1e-5)*nAngles, 1 );
    ctb = zeros( sum(radii >= 1e-5)*nAngles, 1 );
    
    idx = 1; % Start from 2nd element of (cta,ctb) [first element = (0,0)]
    for ii = 1:nAngles
        cosAng = cosAngs(ii);
        sinAng = sinAngs(ii);
        
        for jj = 1:nContrasts
            r = radii(jj);
            if r < 1e-5; continue; end
            cta(idx) = r * cosAng;
            ctb(idx) = r * sinAng;
            
            idx = idx + 1;
        end
    end
end