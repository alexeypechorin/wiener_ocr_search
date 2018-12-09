root_directory='data/wiener_zahi/images';
root_directory_out='data/wiener_zahi_cropped';

D1=dir(root_directory);
for d1=3:7% numel(D1)

    if strfind(D1(d1).name,'.DS_Store')
        continue;
    end
    % imname='../images/00090006.png';
    imname=D1(d1).name;
    fullimname=fullfile(root_directory,imname);
    fprintf('Process image %s\n',fullimname);
    im=imread(fullimname);

    im_binary=imbinarize(im);
    im_binary=im_binary(:,:,1);

    [im_labels,last_label,bounding_rects,sorted_areas,origin_labels,im_all_labels,centroid,stats] = ...
        biggest_con_comps(im_binary);

    x=bounding_rects(1,2);
    y=bounding_rects(1,1);

    w=bounding_rects(1,4)-bounding_rects(1,2);
    h=bounding_rects(1,3)-bounding_rects(1,1);

    mask = imcrop(im_binary,[x,y,w,h]);

    fragment=imcrop(im,[x,y,w,h]);

    if ~exist(fullfile(root_directory_out,curdir),'dir')
        fprintf('create dir %s\n',fullfile(root_directory_out,curdir))
        mkdir(fullfile(root_directory_out,curdir));
    end
    suffix=['_',num2str(x),'_',num2str(y), '_', num2str(w),'_',num2str(h)];
    outimname=fullfile(root_directory_out,curdir,[imname(1:end-4),suffix]);
    imwrite(fragment,[outimname,'.jpg'])
end