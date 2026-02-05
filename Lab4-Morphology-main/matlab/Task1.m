A = imread('assets/text-broken.tif');
B1 = [0 1 0;
     1 1 1;
     0 1 0];    % create structuring element

B2 = ones(3,3);

Bx = [1 0 1;
      0 1 0;
      1 0 1];

A1 = imdilate(A, B1);
A2 = imdilate(A, B2);
A3 = imdilate(A, Bx);
montage({A, A1, A2, A3})

SE = strel('disk',4);
SE.Neighborhood

clear all
close all
A = imread('assets/wirebond-mask.tif');
SE2 = strel('disk',2);
SE10 = strel('disk',10);
SE20 = strel('disk',20);
E2 = imerode(A,SE2);
E10 = imerode(A,SE10);
E20 = imerode(A,SE20);
montage({A, E2, E10, E20}, "size", [2 2])
