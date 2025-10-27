# Topology-Aware Hierarchical Mamba for Salient Object Detection in Remote Sensing Imagery

⭐ This code has been completely released ⭐ 

⭐ our [article](https://ieeexplore.ieee.org/document/11180127) ⭐ 

If our code is helpful to you, please cite:

```
@ARTICLE{11180127,
  author={Yang, Wei and Yi, Zhiqi and Huang, Andong and Wang, Ying and Yao, Yongxiang and Li, Yansheng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Topology-Aware Hierarchical Mamba for Salient Object Detection in Remote Sensing Imagery}, 
  year={2025},
  volume={63},
  number={},
  pages={1-16},
  keywords={Remote sensing;Topology;Semantics;Shape;Object detection;Feature extraction;Image edge detection;Accuracy;Transformers;Optical sensors;Mamba;remote sensing images (RSI);salient object detection (SOD);topology enhancement},
  doi={10.1109/TGRS.2025.3614376}}
```
# 📖 Introduction
<span style="font-size: 125%">
Salient object detection (SOD) plays a crucial role in the intelligent interpretation of remote sensing tasks. Significant advancements have been made with SOD methods based on convolutional neural networks (CNNs) and Transformers. However, the structural differences and complex topological relationships presented by diverse remote sensing scenes often limit SOD performance. To address these challenges, this study proposes a novel method for SOD in remote sensing images (RSI) called topology-aware hierarchical Mamba network (THMNet), which fully utilizes the Mamba’s advantages in global modeling and linear computation. Specifically, a grouped Mamba topology enhancement module is designed to extract high-level semantic features and improve the global representation of structural information for salient objects. Furthermore, a hierarchy Mamba segmentation optimization (HMSO) module is introduced, which employs the prior guidance to adaptively focus on the foreground and background at different levels. This module reduces the processing of noncritical information within complex topological relationships and enhances the capture and depiction of intricate objects. Finally, experimental results demonstrate that our method outperforms mainstream CNNs- and Transformers-based SOD techniques, effectively detecting salient objects in complex backgrounds.
</span>
<p align="center"> <img src="Fig/Fig 2.png" width=90%"></p>


# Saliency maps
We provide saliency maps of our and compared methods at [here](https://pg6e1IlXNqGdg?pwd=hmpg) on two datasets (ORSSD and EORSSD).


## EORSSD complex-scene dataset
EORSSD complex-scene will be released soon.


## Time

**2024.11.13** Upload code

**2025.9.8** Upload README.md

**2025.10.27** Upload Fig 

**2025.10.27** Upload README.md