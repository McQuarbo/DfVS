f = imread('assets/normal-blood.tif');

% Checking all 4 channels
%R = f(:,:,1);
%G = f(:,:,2);
%B = f(:,:,3);
%A = f(:,:,4);
%montage({R, G, B, A})
%imshow(A, [])
%minA = min(A(:));
%maxA = max(A(:));
%disp([minA maxA])

% Removing the 4th blank channel
f = f(:,:,1:3);

% converting to BW binary image
BW = rgb2gray(f);
level = graythresh(BW);
BW = imbinarize(BW, level);

% inverting image
cells = ~BW;

% filling cells
cellsFilled = imfill(cells, 'holes');

% separating cells that are only just touching
cellsFIlled = imopen(cellsFilled, strel('disk', 1));

% label and count
CC = bwconncomp(cellsFilled);
numCells = CC.NumObjects;
fprintf('Estimated RBC count: %d\n', numCells);

montage({f, BW, cells, cellsFilled})