% Lab 6 Task 1 - Solutions
% This Matlab script firstly decimate the image by a factor of 2
% .. to form a multiresolution pyramid by two methods discussed 
% .. in the lecture 
clear all; close all;
f0 = imread('assets/cafe_van_gogh.jpg');

% Decimation by dropping samples & display
f1 = f0(1:2:end,1:2:end,:);
f2 = f1(1:2:end,1:2:end,:);
f3 = f2(1:2:end,1:2:end,:);
f4 = f3(1:2:end,1:2:end,:);
f5 = f4(1:2:end,1:2:end,:);
figure(1)
montage({f0,f1,f2,f3,f4,f5},'Size',[2 3]);
title('Wrong way of subsampling', 'FontSize', 14);

% Use imresize which first filter the image before dropping samples
f_1 = imresize(f0, 0.5);
f_2 = imresize(f_1, 0.5);
f_3 = imresize(f_2, 0.5);
f_4 = imresize(f_3, 0.5);
f_5 = imresize(f4, 0.5);
figure(2)
montage({f0,f_1,f_2,f_3,f_4,f_5},'Size',[2 3]);
title('Correct way of subsampling', 'FontSize', 14);

% Compare original with level 4 image
figure(3)
imshow(f0)
title('Original Image', 'FontSize', 14);
figure(4)
imshow(f_3);
title('1/8 image (filtered)', 'FontSize', 11);
figure(5)
imshow(f3);
title('1/8 image (drop samples)', 'FontSize', 11);