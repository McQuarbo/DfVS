f = imread('assets/fingerprint-noisy.tif');
SE = strel("disk", 1);
fe = imerode(f, SE);
fd = imdilate(fe, SE);
fo = imopen(fd, SE);
fc = imclose(fo, SE);
% montage({f, fe, fd, fo}, "size", [2 2])
% montage({fo, fc}, "size", [1 2])

w_gauss = fspecial('Gaussian', [10,10], 1.5);
g_gauss = imfilter(f, w_gauss, 0);

montage({f, fc, g_gauss}, "Size", [1 3])

