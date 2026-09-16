%% download_test_images.m
%   Downloads standard USC-SIPI test images from Wikimedia Commons
%   (public domain) and saves them to data/test_images/
%
%   Run this once before using any figure scripts or benchmarks.

clear; clc;

out_dir = fullfile(fileparts(mfilename('fullpath')),'..','data','test_images');
if ~exist(out_dir,'dir'); mkdir(out_dir); end

images = {
    'lena.png',      'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png';
    'baboon.png',    'https://upload.wikimedia.org/wikipedia/commons/a/a8/Mandrill_original.jpg';
    'peppers.png',   'https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg';
    'cameraman.png', 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Grayscale_8bits_palette_sample_image.png/256px-Grayscale_8bits_palette_sample_image.png';
};

fprintf('Downloading USC-SIPI test images...\n');
for k = 1:size(images,1)
    fname  = fullfile(out_dir, images{k,1});
    if exist(fname,'file')
        fprintf('  %s already exists — skipping\n', images{k,1});
        continue;
    end
    try
        img    = imread(images{k,2});
        img_g  = uint8(mean(double(img),3));
        img_r  = imresize(img_g,[256 256]);
        imwrite(img_r, fname);
        fprintf('  Downloaded: %s\n', images{k,1});
    catch e
        fprintf('  FAILED: %s  (%s)\n', images{k,1}, e.message);
    end
end

% Also try MATLAB built-in images
builtin_map = {'cameraman.tif','cameraman.png'; ...
               'rice.png','rice.png'; ...
               'coins.png','coins.png'};
for k = 1:size(builtin_map,1)
    src = builtin_map{k,1}; dst = fullfile(out_dir, builtin_map{k,2});
    if ~exist(dst,'file')
        try
            img = imread(src);
            imwrite(img, dst);
            fprintf('  Copied built-in: %s\n', builtin_map{k,2});
        catch; end
    end
end

fprintf('\nDone. Images saved to: %s\n', out_dir);
fprintf('Verify with: imshow(imread(''%s/lena.png''))\n', out_dir);
