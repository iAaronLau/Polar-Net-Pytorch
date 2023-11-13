
## [Polar-Net: A Clinical-Friendly Model for Alzheimer’s Disease Detection in OCTA Images](https://arxiv.org/abs/2311.06009) (MICCAI 2023)
[![DOI](https://img.shields.io/badge/DOI-10.1007/978--3--031--43990--2__57-darkyellow)](https://doi.org/10.1007/978-3-031-43990-2_57)
[![SharedIt](https://img.shields.io/badge/SharedIt-rdcu.be/dnwMc-darkyellow.svg)](https://rdcu.be/dnwMc)
[![arXiv](https://img.shields.io/badge/arXiv-2311.06009-darkyellow.svg)](https://arxiv.org/abs/2311.06009)

By [Shouyue Liu](https://github.com/iAaronLau), [Jinkui Hao](https://scholar.google.com/citations?user=XQqCo8QAAAAJ&hl=zh-CN), [Yanwu Xu](https://scholar.google.com/citations?user=0jP8f7sAAAAJ&hl=zh-CN), [Huazhu Fu](https://scholar.google.com/citations?user=jCvUBYMAAAAJ&hl=zh-CN), [Xinyu Guo](https://github.com/Mr-Guowang), [Jiang Liu](https://scholar.google.com/citations?user=NHt3fUcAAAAJ&hl=zh-CN), [Yalin Zheng](https://scholar.google.com/citations?user=nKCHXTAAAAAJ&hl=zh-CN), [Yonghuai Liu](https://scholar.google.com/citations?user=8J-qVlQAAAAJ&hl=zh-CN), [Jiong Zhang](https://scholar.google.com/citations?user=UJKsxKkAAAAJ&hl=zh-CN), and [Yitian Zhao](https://scholar.google.com/citations?user=8mULu94AAAAJ&hl=zh-CN)

![image](https://github.com/iAaronLau/Polar-Net-Pytorch/blob/master/images/figflowchart5.png "Flowchart")



### Contents
1. [Abstract](#Abstract)
2. [Requirements](#Requirements)
3. [Dataset](#Dataset)
4. [Training&Testing](#Training&Testing)
5. [Citing](#Citing)


### Abstract

Optical Coherence Tomography Angiography (OCTA) is a promising tool for detecting Alzheimer's disease (AD) by imaging the retinal microvasculature. Ophthalmologists commonly use region-based analysis, such as the ETDRS grid, to study OCTA image biomarkers and understand the correlation with AD. However, existing studies have used general deep computer vision methods, which present challenges in providing interpretable results and leveraging clinical prior knowledge. To address these challenges, we propose a novel deep-learning framework called Polar-Net. Our approach involves mapping OCTA images from Cartesian coordinates to polar coordinates, which allows for the use of approximate sector convolution and enables the implementation of the ETDRS grid-based regional analysis method commonly used in clinical practice. Furthermore, Polar-Net incorporates clinical prior information of each sector region into the training process, which further enhances its performance. Additionally, our framework adapts to acquire the importance of the corresponding retinal region, which helps researchers and clinicians understand the model's decision-making process in detecting AD and assess its conformity to clinical observations. Through evaluations on private and public datasets, we have demonstrated that Polar-Net outperforms existing state-of-the-art methods and provides more valuable pathological evidence for the association between retinal vascular changes and AD. In addition, we also show that the two innovative modules introduced in our framework have a significant impact on improving overall performance.

### Requirements

1. System Requirements:
	- NVIDIA GPUs, CUDA supported.
	- Ubuntu 20.04 workstation or server
	- Anaconda environment
	- Python 3.9
	- PyTorch 2.0 
	- Git

2. Installation:
   - `git clone https://github.com/iAaronLau/Polar-Net-Pytorch.git`
   - `cd ./Polar-Net-Pytorch`
   - `conda env create -f environment.yaml`


### Dataset

Please put the root directory of your dataset into the folder ./data. The root directory contain the two subfolder now: AD and control. The most convenient way is to follow the sample file structure, as follows:

```
|-- data
    |-- root directory
        |-- AD
        |-- control
            |-- ID_name
                |-- macular3_3 or 3x3
                    |-- *SVC.png
                    |-- *DVC.png
                    |-- *choriocapillaris.png or *CC.png
                    |-- ... 
```

You can also change the file structure. Note that you need to change the data processing function to ensure the data can be obtained correctly. 

Due to the method need the multiple inputs, i.e., SVC, DVC and choriocapillaris, so the most important thing is that you need specify the filter words for file name of SVC, DVC, and choriocapillaris. Please make sure the three filter words are in the right order.

### Training&Testing
You can change the experiment parameters by modifying the configuration file, Config.py,  and then come to train the model.

```
python start_e200_polarnet.py
```

The results will be automatically saved in the ./save folder.

### Citing 

If you find our paper useful in your research, please consider citing:

```
@inproceedings{liu2023polar,
  title={Polar-Net: A Clinical-Friendly Model for Alzheimer’s Disease Detection in OCTA Images},
  author={Liu, Shouyue and Hao, Jinkui and Xu, Yanwu and Fu, Huazhu and Guo, Xinyu and Liu, Jiang and Zheng, Yalin and Liu, Yonghuai and Zhang, Jiong and Zhao, Yitian},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={607--617},
  year={2023},
  organization={Springer}
}
```

### License
MIT License
