% Wrapper to compute the SLIC superpixels of a given shot
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

function superpixels = computeSLIC( param, frames )
    
    nframes = length(frames)-1; 

    totalTimeTaken = 0;
    
    superpixels = cell( nframes, 1 );
    
    for( index = 1: nframes )

        tic;
        if( param.vocal )
            fprintf( 'computeSLIC: Processing frame %i/%i... ', ...
            index, nframes );
        end
        
        frame = frames{index}; 
        % param.segnum =500; param.m = 20;
        superpixels{ index } = SLIC_mex( 3, frame, 500, 30 );
        
        timeTaken = toc;
        totalTimeTaken = totalTimeTaken + timeTaken;
        
        if( param.vocal )
            fprintf( 'time taken: %.2f seconds\n', timeTaken );
        end
        
    end
    
    if( param.vocal )
        fprintf( 'computeSLIC: Total time taken: %.2f sec\n', totalTimeTaken );
        fprintf( 'computeSLIC: Average time taken per frame: %.2f sec\n', ...
            totalTimeTaken / frames );
    end
    
end
