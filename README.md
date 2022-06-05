# A Text Attention Network for Spatial Deformation Robust Scene Text Image Super-resolution (CVPR2022)

https://arxiv.org/abs/2203.09388

_Jianqi Ma, Zhetong Liang, [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang)_  
_Department of Computing, [The Hong Kong Polytechnic University](http://www.comp.polyu.edu.hk), Hong Kong, China_ & OPPO Research

#### Recovering TextZoom samples
![TATT visualization](./visualizations/TATT-vis.jpg)

## Environment:


![python](https://img.shields.io/badge/python-v3.8-green.svg?style=plastic)
![pytorch](https://img.shields.io/badge/pytorch-v1.8-green.svg?style=plastic)
![cuda](https://img.shields.io/badge/cuda-v11.0-green.svg?style=plastic)
![numpy](https://img.shields.io/badge/numpy-1.18-green.svg?style=plastic)

```
Other possible python packages like pyyaml, cv2, Pillow and imgaug
```

## Main idea
### The pipeline
<img src="./visualizations/TATT_pipeline_v2.jpg" width="720px"/> 

### TP Interpreter
<img src="./visualizations/TATT-TP_Interpreter.jpg" width="720px"> 

## Configure your training
### Download the pretrained recognizer from: 

	Aster: https://github.com/ayumiymk/aster.pytorch  
	MORAN:  https://github.com/Canjie-Luo/MORAN_v2  
	CRNN: https://github.com/meijieru/crnn.pytorch

Unzip the codes and walk into the 'TATT_ROOT/', place the pretrained weights from recognizer in 'TATT_ROOT/'.

### Download the TextZoom dataset:

	https://github.com/JasonBoy1/TextZoom

### Train the corresponding model (e.g. TPGSR-TSRN):
```
chmod a+x train_TATT.sh
./train_TATT.sh
```

### Run the test-prefixed shell to test the corresponding model.
```
Adding '--go_test' in the shell file
```
## Cite this paper:

	@article{ma2021text,
  	title={A Text Attention Network for Spatial Deformation Robust Scene Text Image Super-resolution},
  	author={Ma, Jianqi and Zhetong, Liang and Zhang, Lei},
  	journal={},
  	year={2022}
	}


