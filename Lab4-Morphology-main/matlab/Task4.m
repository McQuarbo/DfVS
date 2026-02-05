f = imread('assets/fingerprint.tif');
f = imcomplement(f);
level = graythresh(f);
BW = imbinarize(f, level);
SE = strel("disk", 3);
g1 = bwmorph(BW, "thin");
g2 = bwmorph(g1, "thin");
g3 = bwmorph(g2, "thin");
g4 = bwmorph(g3, "thin");
g5 = bwmorph(g4, "thin");
g6 = bwmorph(BW, "thin", inf);
montage({g1, g2, g3, g4, g5, g6}, "Size", [2 3])

% try thickening with inverse image