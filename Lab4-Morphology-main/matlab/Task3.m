I = imread('assets/blobs.tif');
I = imcomplement(I);
level = graythresh(I);
BW = imbinarize(I, level);

w_gauss = fspecial('Gaussian', [5,5], 1.5);
g_gauss = imfilter(BW, w_gauss, 0);
SE = strel("disk", 1);

fe = imerode(BW, SE);
fd = imdilate(fe, SE);
fo = imopen(fd, SE);
fc = imclose(fo, SE);

ie = imerode(g_gauss, SE);
If = g_gauss - ie;

montage({I, g_gauss, If}, "size", [1 3])