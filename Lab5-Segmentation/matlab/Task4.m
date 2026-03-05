I = im2double(imread('assets/yeast-cells.tif'));
I = imgaussfilt(I, 1);

w = 35;
h = fspecial('average', w);
m  = imfilter(I, h, 'replicate');
m2 = imfilter(I.^2, h, 'replicate');
sigma = sqrt(max(m2 - m.^2, 0));

% Lower threshold for "body"
k1 = 0.04;   c1 = 0.05;
Tlow  = m*(1 + k1) + c1*sigma;

% Higher threshold for "core"
k2 = 0.05;   c2 = 0.18;
Thigh = m*(1 + k2) + c2*sigma;

BW_body = I > Tlow;
BW_core = I > Thigh;

BW_body = imfill(BW_body,'holes');
BW_body = bwareaopen(BW_body, 200);

BW_core = bwareaopen(BW_core, 30);

BW_recon = imreconstruct(BW_core, BW_body);
BW_recon = imfill(BW_recon,'holes');
BW_recon = bwareaopen(BW_recon, 200);

montage({BW_body, BW_core, BW_recon}, 'Size',[1 3])