f1 = imread('assets/circuit.tif');
f2 = imread('assets/brain_tumor.jpg');

%[g11, t11] = edge(f1, 'Sobel', 0.02);
%[g112, t112] = edge(f1, 'Sobel', 0.05);
%[g113, t113] = edge(f1, 'Sobel', 0.08);
%[g114, t114] = edge(f1, 'Sobel', 0.11);
%montage({g11, g112, g113, g114}, 'Size', [1, 4])



%[g12, t12] = edge(f1, 'Log', []);
%[g121, t121] = edge(f1, 'log', 0.004, 1.5);
%[g122, t122] = edge(f1, 'log', 0.006, 1.5);
%[g123, t123] = edge(f1, 'log', 0.008, 1.5);
%[g124, t124] = edge(f1, 'log', 0.004, 2);
%[g125, t125] = edge(f1, 'log', 0.006, 2);
%[g126, t126] = edge(f1, 'log', 0.008, 2);
%[g127, t127] = edge(f1, 'log', 0.004, 2.5);
%[g128, t128] = edge(f1, 'log', 0.006, 2.5);
%[g129, t129] = edge(f1, 'log', 0.008, 2.5);
%montage({g121, g122, g123, g124, g125, g126, g127, g128, g129}, 'Size', [3,3])

%[g13, t13] = edge(f1, 'Canny', []);
% High thresholds to test
highVals = [0.08 0.12 0.16];

% Sigma values to test
sigmaVals = [1.5 2 2.5];

results = cell(3,3);

for i = 1:3
    for j = 1:3
        
        highT = highVals(j);
        lowT  = 0.4 * highT;   % typical ratio
        
        results{i,j} = edge(f1, 'Canny', [lowT highT], sigmaVals(i));
        
    end
end

montage(results, 'Size', [3 3])

%[g21, t21] = edge(f2, 'Sobel', []);
%[g22, t22] = edge(f2, 'Log', []);
%[g23, t24] = edge(f2, 'Canny', []);

%montage({g11, g12, g13, g21, g22, g23}, 'size', [2 3])
