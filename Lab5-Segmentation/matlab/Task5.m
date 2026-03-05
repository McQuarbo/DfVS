clear; close all; clc;

% Images
img1 = imread('assets/baboon.png');
img2 = imread('assets/peppers.png');

k_values = [2 3 5 10 25];

%% --------- BABOON ---------
figure('Name','Baboon k-means Results');

results_baboon = cell(1, length(k_values)+1);
results_baboon{1} = img1;   % first image = original

for i = 1:length(k_values)
    k = k_values(i);
    
    [L, centers] = imsegkmeans(img1, k);
    J = label2rgb(L, im2double(centers));
    
    results_baboon{i+1} = J;
end

montage(results_baboon, 'Size', [2 3])

%% --------- PEPPERS ---------
figure('Name','Peppers k-means Results');

results_peppers = cell(1, length(k_values)+1);
results_peppers{1} = img2;   % first image = original

for i = 1:length(k_values)
    k = k_values(i);
    
    [L, centers] = imsegkmeans(img2, k);
    J = label2rgb(L, im2double(centers));
    
    results_peppers{i+1} = J;
end

montage(results_peppers, 'Size', [2 3])