clear; close all;
DATA_DIR = fullfile('data','binarization_test'); 

images = dir(fullfile(DATA_DIR, '*.png'));

for i=1:size(images,1)
    close all; 
    img = imread(fullfile(DATA_DIR, images(i).name));
    img = rgb2gray(img);
    original_img = img; 
    % img = double(img)./255;
    % img = uint8(exp(-img));
    % img = img./max(img(:));
%     img = bwareaopen(img, 20);
%     skel = bwmorph(img, 'skel', Inf);
%     dilated = imdilate(skel, strel('disk',3));
%     endpoints = bwmorph(skel, 'endpoints');
%     figure(); imshow([img, skel, dilated, endpoints]);
    figure(); imshow(original_img);
    theta = 0:90;
    [R,xp] = radon(img,theta);
%     for j=1:size(theta,2)
%         figure(); 
%         plot(xp,R(:,j));
%         title(num2str(theta(j)))
%         close;
%     end
    imagesc(theta,xp,R);
    title('R_{\theta} (X\prime)');
    xlabel('\theta (degrees)');
    ylabel('X\prime');
    set(gca,'XTick',0:1:90);
    colormap(hot);
    colorbar
end