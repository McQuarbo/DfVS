imfinfo('peppers.png')
RGB = imread('peppers.png');

[R,G,B] = imsplit(RGB);
montage({R, G, B},'Size',[1 3])