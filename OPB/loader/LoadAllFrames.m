function  [raw, frames, names, height,width, numofframe ] = LoadAllFrames(param,video_name)
     video = VideoReader(fullfile(param.videofolder,video_name));
     numofframe = video.NumberOfFrames;
     height = video.Height;
     width = video.Width;   
     frames = cell( numofframe, 1 );
     raw = cell( numofframe, 1 );
     names = cell( numofframe, 1 );
     for i = 1: numofframe
         frame = read(video,i);
         raw{ i } = double(frame);
         frameName =  num2str(i);
         frame = imfilter(frame,fspecial('gaussian',7,1.5),'same','replicate');
         frames{ i } = double(frame);
         names{ i } =  frameName;
     end
     
end
