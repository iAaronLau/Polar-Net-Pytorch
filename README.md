
## [Polar-Net: A Clinical-Friendly Model for Alzheimer’s Disease Detection in OCTA Images](https://arxiv.org/abs/xxxx) (MICCAI 2023)
[![DOI](https://img.shields.io/badge/DOI-10.1007/978--3--031--43990--2__57-darkyellow)](https://doi.org/10.1007/978-3-031-43990-2_57)
[![SharedIt](https://img.shields.io/badge/SharedIt-rdcu.be/dnwMc-darkyellow.svg)](https://rdcu.be/dnwMc)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx-darkyellow.svg)](https://arxiv.org/abs/xxxx)

By [Shouyue Liu](https://github.com/iAaronLau), Jinkui Hao, Yanwu Xu, Huazhu Fu, Xinyu Guo, Jiang Liu, Yalin Zheng, Yonghuai Liu, Jiong Zhang, and Yitian Zhao

![image](https://github.com/iAaronLau/Polar-Net-Pytorch/blob/master/images/figflowchart5.png "Flowchart")



### Contents
1. [Requirements](#Requirements)
2. [Dataset](#Dataset)
3. [Training&Testing](#Start_training_and_evaluation)
4. [Notes](#Citing)


### Requirements

1. System Requirements:
	- `cd xx`
	- `conda env create -f environment.yml`

2. Installation:
	- `cd xx`
	- `conda env create -f environment.yml`


### Dataset

Please put the root directory of your dataset into the folder ./data. The root directory contain the two subfolder now: AD and control. The most convenient way is to follow the sample file structure, as follows:

```
|-- .data
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

### Start training and evaluation
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
