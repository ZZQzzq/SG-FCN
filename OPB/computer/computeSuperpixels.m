% Function to compute some given superpixel method for a given shot
%
%    Copyright (C) 2013  Anestis Papazoglou
%
%    You can redistribute and/or modify this software for non-commercial use
%    under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%    For commercial use, contact the author for licensing options.
%
%    Contact: a.papazoglou@sms.ed.ac.uk

function superpixels = computeSuperpixels(param,frames,video_name )
    
    superpixelfolder = fullfile( param.salfolder, 'superpixels' ); 
    if( ~exist( superpixelfolder, 'dir' ) )
        mkdir( superpixelfolder );
    end

    fprintf( 'computeSuperpixels: \n');
    filename = fullfile( superpixelfolder,strcat(video_name, '_superpixels.mat') );
    
    if( exist( filename, 'file' ) )  
        % Shot already processed, skip
        fprintf( 'computeSuperpixels: Data processed, skipping...\n' );
        superpixels = loadSuperpixels( param );
        return;     
    else 
%         superpixels = computeSLIC( param, frames );   
   
        nframes = length(frames)-1; 
        totalTimeTaken = 0;
        superpixels = cell( nframes, 1 );  
        for index = 1: nframes 
           tic;  
           
           fprintf( 'SLIC: Processing frame %i/%i <====> ', ...
               index, nframes );  
           frame = frames{index}; 
           [ height,width ] = size(frame(:,:,1)); 
           PixNum = height*width;   
           frameVecR = reshape( frame(:,:,1)', PixNum, 1);  
           frameVecG = reshape( frame(:,:,2)', PixNum, 1);
           frameVecB = reshape( frame(:,:,3)', PixNum, 1); 
           frameAttr=[ height ,width, 500, 30, PixNum ];  
           [ Label, Sup1, Sup2, Sup3, ~ ] = SLIC( double(frameVecR), double(frameVecG), double(frameVecB), frameAttr );
           Label = int32(reshape(Label,width,height)');
           superpixels{ index }.Label = Label+1;
           superpixels{ index }.Sup1 = Sup1;
           superpixels{ index }.Sup2 = Sup2;
           superpixels{ index }.Sup3 = Sup3;
           %superpixels{ index } = SLIC_mex( frame, 1500, 30 );
           timeTaken = toc;
           totalTimeTaken = totalTimeTaken + timeTaken;


           fprintf( 'time taken: %.2f seconds\n', timeTaken );
        end
 
        fprintf( 'SLIC: Total time taken: %.2f sec\n', totalTimeTaken );
        fprintf( 'SLIC: Average time taken per frame: %.2f sec\n', ...
               totalTimeTaken / nframes );

        save( filename, 'superpixels', '-v7' );
    end
    
    fprintf( 'computeSuperpixels: finished processing\n' );
    
end
