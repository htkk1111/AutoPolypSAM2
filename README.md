<h1 align="center">AutoPolypSAM2: Auto Prompt Encoding Adaptation of Medical SAM2 for Polyp Segmentation</h1>

## ● Overall Architecture
 <div align="center"><img src="https://github.com/htkk1111/AutoPolypSAM2/blob/main/vis/model.png"></div>

## ● Results
 You could download our results from [goole-drive](https://drive.google.com/file/d/15yMO2jgs0LAAeKvGUCwoL85lzUvjj0A5/view?usp=sharing).
 
 You can download our weights and the pre-trained weights of HardNet85 from [goole-drive](https://drive.google.com/file/d/1F_x8mebWuwkKZtlY-x75BQZXBiwi9FOC/view?usp=drive_link).
## ● Usage
### Recommended environment:
```
python 3.12.x
cuda 12.x
```
### Data preparation
Please organize the training and inference data in the following format.
```
data/
│
├── Frame/
│   ├── sub1/
│   │   ├── frame1.jpg
│   │   ├── frame2.jpg
│   │   └── ...
│   └── sub2/
│       ├── frame3.jpg
│       ├── frame4.jpg
│       └── ...
│
└── GT/
    ├── sub1/
    │   ├── frame1.png
    │   ├── frame2.png
    │   └── ...
    └── sub2/
        ├── frame3.png
        ├── frame4.png
        └── ...

```
## Acknowledgement
We are very grateful for these excellent works [MedSAM2](https://github.com/SuperMedIntel/Medical-SAM2), [AutoSAM](https://github.com/talshaharabany/AutoSAM) and [PolypPVT](https://github.com/DengPingFan/Polyp-PVT), which have provided the basis for our framework.

