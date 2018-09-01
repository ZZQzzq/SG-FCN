% Wrapper to compute some given optical flow method
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

function flow = computeOpticalFlow( param, frames,video_name)
    
    flowfolder = fullfile( param.salfolder, 'flow');  
    if( ~exist( flowfolder, 'dir' ) )
        mkdir( flowfolder );
    end

    fprintf( 'computeOpticalFlow: \n');
    filename = fullfile( flowfolder, strcat(video_name, '_flow.mat' ));
    
    if( exist( filename, 'file' ) )  
        % Shot already processed, skip
        fprintf( 'computeOpticalFlow: Data processed, skipping...\n' );
        flow = loadFlow( param );
        return;
    else   
        flowframes = length(frames)-1;  
        flow = cell( 1, flowframes);  

        totalTimeTaken = 0; 
        parfor i =  1: flowframes 
            tic  
            currImage = frames{i};  
            if( size( currImage, 3 ) == 1 )
                currImage = gray2rgb( currImage ); 
            end
            currImage = double( currImage );  
            
            nextImage = frames{i+1}; 
            if( size( nextImage, 3 ) == 1 )
                nextImage = gray2rgb( nextImage );
            end
            nextImage = double( nextImage );
 
            fprintf( 'computeBroxPAMI2011Flow: Computing optical flow of pair: %i of %i... ', ...
                    i, length(frames)-1);
            flowframe = mex_LDOF( im2double(currImage), im2double(nextImage) );  
            flow{ i }( :, :, 1 ) = flowframe( :, :, 2 );  
            flow{ i }( :, :, 2 ) = flowframe( :, :, 1 );
            timeTaken = toc;
            totalTimeTaken = totalTimeTaken + timeTaken;


            fprintf( 'done. Time taken: %.2f sec\n', timeTaken );

        end

        fprintf( 'computeBroxPAMI2011Flow: Total time taken: %.2f sec\n', ...
                totalTimeTaken );
        fprintf( 'computeBroxPAMI2011Flow: Average time taken per frame: %.2f sec\n', ...
                totalTimeTaken / flowframes );
        save( filename, 'flow', '-v7' );
    end    
    fprintf( 'computeOpticalFlow finished\n');
end
