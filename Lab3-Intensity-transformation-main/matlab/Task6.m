clear all
close all
f = imread('assets/moon.tif');

w_lap = fspecial('laplacian', 1.0)
w_sob = fspecial('sobel')

g_lap = imfilter(f, w_lap, 0);
g_sob = imfilter(f, w_sob, 0);

g_unsh = imsharpen(f, "Radius", 1.2, "Amount", 1.0, "Threshold", 0.02)

size(f)
size(g_lap)
size(g_sob)
size(g_unsh)

M = cat(4, f, g_lap, g_sob, g_unsh);
figure; montage(M);