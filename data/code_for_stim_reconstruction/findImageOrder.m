function [ prs ] = findImageOrder( nimages, npulses, maxrepeats, seed )
    %FINDIMAGEORDER Recover the order of images in an ImageSequence stimulus
    %    [prs ] = findNewOrder( nimages, npulses, maxrepeats, seed )
    %
    %  Input:
    %       nimages: struct containing description of stimulus
    %                (mincontrast, maxcontrast, ncontrasts, nangles)
    %       npulses: number of pulses recorded from the experiment
    %       maxrepeats: only analyze until the maxrepeats-th repeat
    %                  (use 'inf' for analyzing all the repeats)
    %       seed: seed for ran1 (should be the same as in the experiment)
    %
    %  Output:
    %        prs: order in which the images were presented, where 1
    %             marks the presentation of a gray sceen and 2,...,nimages+1 
    %             mark the presentation of the 1st, ..., nth image of the
    %             list
    
    
    nrepeats = floor(npulses/nimages); 
    nrepeats = min(nrepeats, maxrepeats); % Obey maxrepeats
    
    prs = zeros( (nimages+1) * nrepeats, 1 );
    
    for ii = 1:nrepeats
        [order, seed] = shuffleOrder(nimages+1, seed);
        prs( (ii-1)*(nimages+1) + (1:nimages+1) ) = order;
    end
    prs = prs - 1;
end

function [order, seed] = shuffleOrder(nelem, seed)
   % Fisher-Yates shuffle (initializes the vector in a random order)
   
   order = ones(nelem, 1);
   
   order(1) = 1;
   for ii = 2:nelem
       [jj, seed] = ran1(seed);
       jj = floor(2 + (ii-1)*jj);
       
       order(ii) = order(jj);
       order(jj) = ii;     
   end
   
end
