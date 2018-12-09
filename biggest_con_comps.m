function [im_labels,last_label,bounding_rects,sorted_areas,origin_labels,im_all_labels,centroid,stats] = biggest_con_comps(im_binary, min_area_ratio,flag_use_filled_area,flag_smooth)
%The 4 coordinates are: top-left row and column, bottom right row and column
if nargin < 2
    min_area_ratio = 0.03;
end
if nargin < 3
    flag_use_filled_area = true; % use filled area instead of area with holes.
end
if nargin<4
    flag_smooth = true;
end
im_labels=[];
last_label=[];
bounding_rects=[];
sorted_areas=[];
origin_labels=[];
im_all_labels=[];
centroid=[];
convex_images=[];

holes_max_radius = 2; %radius of maximum holes to close in the components
[rows,cols,d] = size(im_binary);
area_thresh = min_area_ratio*rows*cols;
L = bwlabel(im_binary,4);
if nargout>5
    im_all_labels = L;
end
clear im_binary;
areas = histc(L(:),0.5:1:max(L(:))+0.5);
areas(end)=[];
origin_labels = find(areas>=area_thresh);
if isempty(origin_labels)
    return;
end
im_labels = zeros(rows,cols,'double');
for n=1:length(origin_labels)
    im_labels(L==origin_labels(n)) = n;
end
L=im_labels;
if flag_use_filled_area
    stats = regionprops(L,'BoundingBox','FilledArea','Centroid','PixelIdxList');
else
    stats = regionprops(L,'BoundingBox','Area','Centroid','PixelIdxList');
end
max_label = length(stats);
areas=zeros(max_label,1);
bounding_rects = zeros(max_label,4);
for n=1:max_label
    if flag_use_filled_area
        areas(n) = stats(n).FilledArea;
    else
        areas(n) = stats(n).Area;
    end
    bounding_rects(n,1:2) = ceil(stats(n).BoundingBox([2 1]));
    bounding_rects(n,3:4) = bounding_rects(n,1:2) + stats(n).BoundingBox([4 3]) -1;
    centroid(n,:)=stats(n).Centroid;
end
[sorted_areas,idxs] = sort(areas,'descend');
larger_than_thresh = sorted_areas >= area_thresh;
sorted_areas = sorted_areas(larger_than_thresh);
origin_labels = origin_labels(idxs);
origin_labels = origin_labels(larger_than_thresh);
idxs = idxs(larger_than_thresh);
bounding_rects = bounding_rects(idxs,:);
centroid = centroid(idxs,:);
last_label = length(sorted_areas);
idxs = idxs(1:last_label);
im_labels = zeros(rows,cols,'double');
for n=1:last_label
    im_labels(L==idxs(n)) = n;
end
% closes small holes in the components.
if flag_smooth
    se = strel('disk',holes_max_radius);
    im_labels = imdilate(im_labels,se);
    im_labels = imerode(im_labels,se);
end

%im_labels= imfill(im_labels,'holes');