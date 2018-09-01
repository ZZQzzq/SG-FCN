%%
clear;
clc;
addpath( genpath( '..' ) );
root_path = '.';

%% Parameter initialization
param.valScale = 30;
param.alpha = 0.04;
param.color_size = 5;
param.gradLambda = 0.5;
param.blurfrac = 0.05;
param.modelpath = fullfile(root_path,'caffe','models');
param.videofolder = fullfile(root_path,'data'); 
param.outputfolder = fullfile(param.videofolder,'output'); 
param.mean_pix = [104.00698793,  116.66876762,  122.67891434];
param.IMAGE_DIM = 500;
param.sigma = 5;
%%  iInitialize the model
use_gpu = 1;
if exist('use_gpu', 'var') && use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

sgf3_model = fullfile(param.modelpath, 'SGF3','deploy.prototxt');
sgf3_weights = fullfile(param.modelpath,'SGF3','sgf3.caffemodel');
sgfe_model = fullfile(param.modelpath, 'SGFE','sgfe.prototxt');
sgfe_weights = fullfile(param.modelpath,'SGFE','sgfe.caffemodel');
sgfe2_model = fullfile(param.modelpath, 'SGFE','sgf_3in','sgf_3in.prototxt');
sgfe2_weights = fullfile(param.modelpath,'SGFE','sgf_3in','sgf_3in_30000.caffemodel');
phase = 'test';   

videos = dir(fullfile(param.videofolder,'*.avi'));
video_num = length(videos);
for vindex = 1:video_num
    video_name=videos(vindex).name;
    param.salfolder = fullfile( param.outputfolder,video_name);  

    if( ~exist( param.salfolder, 'dir' ) )
        mkdir( param.salfolder ),
    else
        continue;
    end
    % get video information
    [data.rawframes, data.frames,data.names,height,width,nframe ]= LoadAllFrames( param,video_name ); 
    % Get optical flow
    data.flow = loadFlow(param, video_name);
    if( isempty( data.flow ) )
         fprintf( 'compute flow for %s\n',video_name);
         data.flow = computeOpticalFlow( param, data.frames,video_name);  
    end

    % Get superpixel
    data.superpixels = loadSuperpixels( param,video_name);
    if( isempty( data.superpixels ) )
        fprintf( 'compute superpixels for %s\n',video_name);
        data.superpixels = computeSuperpixels( param, data.frames,video_name );  
    end
 
    [ superpixels, ~, bounds, labels ] = makeSuperpixelIndexUnique( data.superpixels );  
    [ meanColoursMxArray, massCentresMxArray, sizesMxArray ] = getSuperpixelStats( data.frames(1:nframe-1), superpixels, double(labels) ); 
    % meanColoursMxArray superpixels center_color
    % massCentresMxArray superpixels center_location
    % sizesMxArray superpixels size

    %% get saliency object boundary
    MovObjVal = cell( nframe-1, 1 );  
    if ~exist(strcat(param.salfolder ,'/Bmap/'),'dir')
        mkdir(strcat(param.salfolder ,'/Bmap/'));
    end
    if ~exist(strcat(param.salfolder ,'/Bsal/'),'dir')
        mkdir(strcat(param.salfolder ,'/Bsal/'));
    end
    Bsal = cell(nframe-1, 1);
    for index = 1:nframe-1
            frame = data.frames{index}; % current frame
            % superpixel segementation
            Label = data.superpixels{index}.Label;   % get Label，label:h*w
            Label = reshape(Label,height*width,1);       
            frameVal = meanColoursMxArray(bounds(index):bounds(index+1)-1,:);
            seg_frame = uint8(reshape(superpixel2pixel(double(data.superpixels{index}.Label),double(frameVal)),height ,width,3)); % superpixel segmentation visualization
            framex=imfilter(seg_frame,fspecial('average',3),'same','replicate');  
            G = edge_detect(framex);  

            % 1.get optical gradient height×width×2(x-wise,y-wise)
            gradient = getFlowGradient( data.flow{index} );  
            % 2.magnitude=sqrt( gradient( :, :, 1 ).^2 + gradient( :, :, 2 ).^2 )
            magnitude = getMagnitude( gradient );  
            if index>1              
                mask = imdilate((MovObjVal{index-1}>0.3),strel('disk',20))+0.3;           
                mask(mask(:)>1)=1;
                magnitude = magnitude.*mask;
                G = G.*mask;
            end
            gradBoundary = 1 - exp( -param.gradLambda * magnitude );
            
            if (max(magnitude(:))<10) 
                gradBoundary = gradBoundary + 0.01;
            end
            G = G.*( gradBoundary );  % spatio-temporal gradient 

            %% saliency via gradient flow
            [V_Energy1 ,H_Energy1 ,V_Energy2, H_Energy2] = energy_map(double(G));       
            if index ==1 
                Energy = min(min(min(H_Energy1,V_Energy1),H_Energy2),V_Energy2);       
            else 
                mask = int32(imdilate((Energy>0.2),strel('disk',20)));
                mask = ~mask;
                Energymap = (Energy<0.05).*mask; 
                Energymap = ~Energymap;
                Energy = Energy*0.3+(Energymap.*min(min(min(H_Energy1,V_Energy1),H_Energy2),V_Energy2))*0.7; % considering saliency of prior frame
            end
            Energy = Energy/max(Energy(:));  
            
            MovObjVal{index} = Energy;
            heatmap = mat2gray(Energy);
            if ( param.blurfrac > 0 )
                k = mygausskernel( max(size(heatmap)) * param.blurfrac , 2);
                heatmap = myconv2(myconv2( heatmap , k ),k');
                heatmap = mat2gray(heatmap);
            end
            heatmap = double(heatmap) / max(heatmap(:));
            data.Bsal{index} = heatmap;
            imwrite(Energy,strcat(param.salfolder ,'/Bmap/',data.names{index,1},'.png'),'png');
            imwrite(heatmap,strcat(param.salfolder ,'/Bsal/',data.names{index,1},'.png'),'png');
    end
    clear frameEnergy MovObjVal
    fprintf( 'Successfully get Bsal,Bmap\n');
    
    %% get static saliency
    data.StaticSal = cell(nframe, 1);
    if ~exist(strcat(param.salfolder ,'/StaticSal/'),'dir')
        mkdir(strcat(param.salfolder ,'/StaticSal/'));
    end 
    sgf3 = caffe.Net(sgf3_model, sgf3_weights, phase);
    for i = 1:nframe
        frame = data.rawframes{i}; 
        im = single(frame);
        im = imresize(im, [param.IMAGE_DIM param.IMAGE_DIM], 'bilinear');
        im = im(:, :, [3 2 1]);% RGB -> BGR 
        for c = 1:3  % mean BGR pixel subtraction
            im(:, :, c, :) = im(:, :, c, :) - param.mean_pix(c);
        end
        input_data={im};
        scores=sgf3.forward(input_data);
        temp=scores{1};
        salmap=permute(temp,[2,1,3]); 
        salmap = mat2gray(temp);
        gausFilter = fspecial('gaussian', [10,10], param.sigma);
        gaus= imfilter(salmap, gausFilter, 'replicate');
        gaus=imresize(gaus,[height,width]);
        salmap=uint8(gaus*255);
        data.StaticSal{i}= salmap;
        imwrite(salmap,strcat(param.salfolder ,'/StaticSal/',data.names{i,1},'.png'),'png');
    end  
    fprintf( 'Successfully get Static Saliency\n');
    caffe.reset_all();
    
    %% get final spatiotemporal saliency
    STSal = cell(nframe, 1);
    if ~exist(strcat(param.salfolder ,'/STSal/'),'dir')
        mkdir(strcat(param.salfolder ,'/STSal/'));
    end 
    for i = 1:nframe
        frame = data.rawframes{i}; 
        presal = data.StaticSal{i};
        if i>=2
            Bsal = data.Bsal{i-1};
            Bsal = imresize(Bsal, [param.IMAGE_DIM param.IMAGE_DIM], 'bilinear');
            Bsal = permute(Bsal,[2,1]);
        end
        im = single(frame);
        im = imresize(im, [param.IMAGE_DIM param.IMAGE_DIM], 'bilinear');
        im = im(:, :, [3 2 1]);% RGB -> BGR 
        for c = 1:3  % mean BGR pixel subtraction
            im(:, :, c, :) = im(:, :, c, :) - param.mean_pix(c);
        end
        
        presal = single(presal);
        presal = imresize(presal, [param.IMAGE_DIM param.IMAGE_DIM], 'bilinear');
        images = zeros(param.IMAGE_DIM, param.IMAGE_DIM, 4, 1, 'single');
        images(:,:,1:3,1) = permute(im,[2 1 3]);
        images(:,:,4,1) = permute(presal,[2,1]);        
        if i>=2
            sgfe2 = caffe.Net(sgfe2_model, sgfe2_weights, phase);
            input = cell(2,1);
            input{1} = {images};
            input{2} = {single(Bsal)};
            scores=sgfe2.forward({images,single(Bsal)});
            temp=scores{1};
            salmap=permute(temp,[2,1,3]);
            salmap = mat2gray(salmap); 
			caffe.reset_all();  
        else
            sgfe1 = caffe.Net(sgfe_model, sgfe_weights, phase);
            scores=sgfe1.forward({images});
            temp=scores{1};
            salmap=permute(temp,[2,1,3]);
            salmap = mat2gray(salmap);
            caffe.reset_all();  
        end
        gausFilter = fspecial('gaussian', [10,10], param.sigma);
        gaus= imfilter(salmap, gausFilter, 'replicate');
        gaus=imresize(gaus,[height,width]);
        salmap=uint8(gaus*255);
        imwrite(salmap,strcat(param.salfolder ,'/STSal/',data.names{i,1},'.png'),'png');
    end  
    clear data
    fprintf( 'Successfully get SpatioTemporal Saliency\n');
    caffe.reset_all();  
end

            
