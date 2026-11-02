function [ cta, ctb, totalflashes, prs ] = findNewOrder( stimdesc, npulses, maxrepeats, seed )
    %FINDNEWORDER Recover the order of cta,ctb from the new SubUnitFlash stimulus
    %    The new stimulus only has radial components and more
    %    parameters such as range of radial search, number of angles,
    %    number of contrasts, etc. It also uses a different shuffling which
    %    might give a different ordering than the one Michael has used
    %    before.
    %
    %    [ cta, ctb, totalflashes, prs ] = findNewOrder( stimdesc, npulses, maxrepeats, seed )
    %
    %  Input:
    %       stimdesc: struct containing description of stimulus
    %                    (mincontrast, maxcontrast, ncontrasts, nangles)
    %        npulses: number of pulses recorded from the experiment
    %     maxrepeats: only analyze until the maxrepeats-th repeat
    %                    (use 'inf' for analyzing all the repeats)
    %           seed: seed for ran1 (should be the same as in the experiment)
    %
    %  Output:
    %             cta: contrast at subtiles A, ctb: contrast at subtiles B
    %    totalflashes: total number of flashes in one repeat
    %             prs: order in which the contrasts are used
    %
    
    
    [cta, ctb] = genLists(stimdesc);
    
    totalflashes = numel(cta);
    nrepeats = floor(npulses/totalflashes); 
    nrepeats = min(nrepeats, maxrepeats); % Obey maxrepeats
    
    prs = zeros( totalflashes * nrepeats, 1 );
    
    for ii = 1:nrepeats
        [order, seed] = shuffleOrder(totalflashes, seed);
        prs( (ii-1)*totalflashes + (1:totalflashes) ) = order;
    end
end

function [order, seed] = shuffleOrder(nelem, seed)
   % Fisher-Yates shuffle (initializes the vector in a random order)
   
   order = ones(nelem, 1);
   
   order(2) = 2;
   for ii = 3:nelem
       [jj, seed] = ran1(seed);
       jj = floor(2 + (ii-1)*jj);
       
       order(ii) = order(jj);
       order(jj) = ii;     
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
    %radii = (0:(nContrasts)) * (maxCon - minCon)/(2*nContrasts);
   
    cta = zeros( sum(radii >= 1e-5)*nAngles + 1, 1 );
    ctb = zeros( sum(radii >= 1e-5)*nAngles + 1, 1 );
    
    idx = 2; % Start from 2nd element of (cta,ctb) [first element = (0,0)]
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
