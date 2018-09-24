# SG-FCN （TC2018）
Now in experimental release, suggestions welcome

## Paper

M.J. Sun, Z.Q. Zhou, Q. H. Hu, and Z. Wang, SG-FCN: A Motion and Memory-Based Deep Learning Model 
for Video Saliency Detection, IEEE Transactions on Cybernetics.

[Paper][1],  [Code][2].  

[1]: https://arxiv.org/abs/1809.07988     "Paper" 
[2]: https://github.com/ZZQzzq/SG-FCN#sg-fcn/  "Code" 

```
    @ARTICLE{8365810, 
    author={M. Sun and Z. Zhou and Q. Hu and Z. Wang and J. Jiang}, 
    journal={IEEE Transactions on Cybernetics}, 
    title={SG-FCN: A Motion and Memory-Based Deep Learning Model for Video Saliency Detection}, 
    year={2018}, 
    volume={}, 
    number={}, 
    pages={1-12}, 
    keywords={Computational modeling;Saliency detection;Predictive models;Feature extraction;Video sequences;Visualization;Training;Eye fixation detection;fully convolutional neural networks;video saliency}, 
    doi={10.1109/TCYB.2018.2832053}, 
    ISSN={2168-2267}, 
    month={},}
```

## Frame Work

### 1. SG-FCN
<div align=center>
    <img src="https://github.com/ZZQzzq/SG-FCN/blob/master/figs/sg-fcn.png"/>
</div>
Flow chart of our proposed model, in which we use the proposed model SGF for capturing the spatial and temporal information simultaneously.
SGF(3) is used to handle the first frame because neither motion nor temporal information is available. From the next frame onward, the SGF(E) model
takes EF(1) from SGF(3), a fast moving object edge map B(2) from the OPB algorithm, and the current frame (2) as the input, and directly outputs the
spatiotemporal prediction EF(2).

### 2. SGF(E)

<div align=center>
 <img src="https://github.com/ZZQzzq/SG-FCN/blob/master/figs/sgfe.png"/>
</div>


Structure of model SGF(E). As shown in the flowchart, the input data is a tensor of h × w × 4. At the top of the model, we add an Eltwise layer with function SUM [big map(i), boundary map(i)] before Sigmoid function.

### 3. OPB Algorithm
details of the OPB algorithm can be found in './OPB/'


## Implement

Please first download and install caffe. [caffe][5]

[5]: https://github.com/BVLC/caffe

The model weights trained on HOLLYWOOD2 and UCF-Sports datasets can be downloaded from

Baidu Wangpan: https://pan.baidu.com/s/1bgu80UOKJOXN2OvhAx_2sw 

password: lfi8

Please put the download 'caffe' folder under the main branch, and put './models/' under the folder './caffe/', then run main.m.

In order to better eliminate the checkerboard effect, we make an adjustment to the parameters of the deconvolution layer.

Our results are sightly improved over the original scores reported in our paper:

| dataset | CC | SIM | NSS | EMD | AUC|
| --- | --- |  --- | --- | --- | --- |
| HOLLYWOOD2 | 0.6181 | 0.5161 |1.5158 |0.8984 |0.8936 |
| UCF-Sports | 0.5985 | 0.4524 |1.4582 |0.8301 |0.9065 |

***
If you find our method useful in your research, please consider citing:
```
M. Sun, Z. Zhou, Q. Hu, Z. Wang, and J. Jiang, “Sg-fcn: A motion and memory-based deep learning model for video saliency detection,” IEEE Transactions on Cybernetics, vol. PP, no. 99, pp. 1–12, 2018.
```
***

***
Welcome to our LAB:

https://zhengwangtju.github.io/

http://www.escience.cn/people/sunmeijun/index.html

Any questions, please contact me via ziqizhou@tju.edu.cn
***

