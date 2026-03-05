% Lab 6 Task 6
% Continuous object recognition using webcam

camera = webcam;
net = googlenet;

inputSize = net.Layers(1).InputSize(1:2);

figure

while true
    
    I = snapshot(camera);
    f = imresize(I,inputSize);
    
    tic
    [label, score] = classify(net,f);
    elapsedTime = toc;
    
    confidence = max(score);
    
    textLabel = sprintf('%s (%.2f)', char(label), confidence);
    
    I = insertText(I,[10 10],textLabel,...
        'FontSize',30,...
        'BoxColor','yellow',...
        'TextColor','black');
    
    imshow(I)
    title(['Processing time: ' num2str(elapsedTime) ' s'])
    drawnow
    
end