function [saccades, stimall, fixstim, transstim] = generateSaccadeSeq(stimdesc,frameN)

saccades = zeros (8,2);
for ii = 1:4
    saccades(2*ii-1,1:2) = [ii,5];
    saccades(2*ii,1:2) = [ii,6];
end
saccades = [repmat(saccades,4,1), kron(1:4,ones(1,8))'];

% calculate the correct order of stimulus.

seedvals= ran1(stimdesc.seed,2*frameN);
seedvals=reshape(seedvals,2,frameN);
fixstim=floor(4 * seedvals(1,:))';
transstim=floor(2 * seedvals(2,:))';
stimall = zeros(1,2*frameN);
stimall(1:2:end)=transstim+5; stimall(2:2:end)=fixstim+1;

end
