% binarization for hebrew papyrus 
close all; 
% take image that you want to use 
img = imread(fullfile('data','00260608_341_2188.jpg'));

num_classes = 3;
min_cc_size = 100; 
% img = cat(3, exp(im2double(img)), im2double(img), exp(-im2double(img)));
% img = cat(3, tan(im2double(img)), img, -tan(im2double(img)));
img = cat(3, img, img, img);

% quantize into num_classes number of bins using 
% minimum variance quantization (explanation in MATLAB documentation)
% each pixel is assigned a class with an index
% I found using higher number of classes is better 
[quantized_img, map] = rgb2ind(img, num_classes);

% lowest class is usually letters (darkest parts of text)
bw = quantized_img == 1;

% get rid of small blobs 
bw = bwareaopen(bw, min_cc_size, 4); 
bw = imopen(bw, strel('disk',1)); 

% label different connected components with color
[labeledImage, numberOfBlobs] = bwlabel(bw, 8);
coloredLabelsImage = label2rgb(labeledImage, 'hsv', 'k', 'shuffle');

% theta = 90;
% [R,xp] = radon(bw,theta);
% plot(xp,R(:,1));

figure(); imshow([img, cat(3,im2uint8(bw),im2uint8(bw),im2uint8(bw)), coloredLabelsImage]); 

