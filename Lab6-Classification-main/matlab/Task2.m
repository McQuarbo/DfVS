clear all; close all;
f = imread('assets/salvador_grayscale.tif');
w = imread('assets/template1.tif');
c = normxcorr2(w, f);
figure(1)
surf(c)
shading interp

[ypeak, xpeak] = find(c==max(c(:)));
yoffSet = ypeak-size(w,1);
xoffSet = xpeak-size(w,2);
figure(2)
imshow(f)
axis on;
drawrectangle(gca,'Position', ...
    [xoffSet,yoffSet,size(w,2),size(w,1)], 'FaceAlpha',0);