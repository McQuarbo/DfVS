clear all
% imfinfo('assets/breastXray.tif')

f = imread('assets/breastXray.tif');
% imshow(f)

% f(3,10)             % print the intensity of pixel(3,10)
% imshow(f(1:241,:))  % display only top half of the image

% [fmin, fmax] = bounds(f(:))

g1 = imadjust(f, [0 1], [1 0])
% figure              % open a new figure window
% montage({f, g1})

g2 = imadjust(f, [0.5 0.75], [0 1]);
g3 = imadjust(f, [ ], [ ], 2);
figure
montage({g2,g3})