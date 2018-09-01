% Function to load the stored optical flow based on some given method
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

function result = loadFlow( param,video_name)

    file = fullfile( param.salfolder, 'flow',strcat(video_name, '_flow.mat' ));
    if( exist( file, 'file' ) ) 
        flow = load( file );
        result=flow.flow;
        fprintf( 'Successfully load flow\n');
    else
        result = [];
    end

end
