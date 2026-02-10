f = imread('assets/fillings.tif');
SE = strel("disk", 1);
SE_2 = strel("disk", 3);
% Noise removal
fd = imdilate(f, SE);
fe = imerode(fd, SE);
fc = imclose(fe, SE_2);
fo = imopen(fc, SE_2);

% Gamma correction
low = 0.80
g = imadjust(fo, [low 1], [0 1], 1);
g2 = imadjust(g, [low 1], [0 1], 0.5);

% Threshold
level = graythresh(g2);
BW = imbinarize(g2, level);

% Labelling and measuring
CC = bwconncomp(BW);
stats = regionprops(CC, 'Area', 'Centroid');
areas = [stats.Area]


montage({g, g2, BW}, "size", [1 3])