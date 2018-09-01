import sys     

import surgery, score  
      
import numpy as np  
import os  
import sys
sys.path.append("yourpathtocaffe/caffe/python")
sys.path.append("yourpathtocaffe/SG-FCN/caffe/caffe")
import caffe
try:  
    import setproctitle  
    setproctitle.setproctitle(os.path.basename(os.getcwd()))  
except:  
    pass  
      
vgg_weights = 'vgg.caffemodel'
vgg_proto = 'vgg.prototxt'
 
      
# init  
caffe.set_mode_gpu()   
caffe.set_device(0)  
      
 
solver = caffe.SGDSolver('solver.prototxt')
vgg_net=caffe.Net(vgg_proto, vgg_weights, caffe.TRAIN)
surgery.transplant(solver.net, vgg_net)
del vgg_net  

# surgeries  
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]  
surgery.interp(solver.net, interp_layers)  
    
solver.solve()  


 
