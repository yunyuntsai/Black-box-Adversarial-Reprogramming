# Black-box-Adversarial-Reprogramming

This is the repo for **Transfer Learning without Knowing: Reprogramming Black-box Machine Learning Models with Scarce Data and Limited Resources, Yun-Yun Tsai, Pin-Yu Chen, Tsung-Yi Ho**, in Proceeding of **[International Conference on Machine Learning (ICML)](https://icml.cc/), 2020**. Our code is implemented in Python 3.6 and Tensorflow 1.14. 

The following figure illustrates the framework for our proposed black-box adversarial reprogramming method (BAR): <br/>
![Alt text](https://user-images.githubusercontent.com/20013955/89761762-b2e55880-db21-11ea-93f8-db0cef7800c3.png)<br/>
<br/>
(a) Generate adversarial program. <br/>
(b) Find _q_ pertubed adversarial programs with vectors that are uniformly drawn at random
from a unit Euclidean sphere. <br/>
(c) Estimate gradient with zeroth-order gradient estimator. The corresponding algorithmic convergence guarantees have been proved in both the convex loss and non-convex loss settings [(Liu et al., 2018; 2019)](https://arxiv.org/pdf/1805.10367.pdf).<br/> 
(d) Optimize adversarial program’s parameters _W_. <br/>


For more detail, please refer to our [main paper](https://proceedings.icml.cc/static/paper_files/icml/2020/3642-Paper.pdf), and [video on slideslive!](https://slideslive.com/38928106/transfer-learning-without-knowing-reprogramming-blackbox-machine-learning-models-with-scarce-data-and-limited-resources?ref=speaker-31425-latest).

