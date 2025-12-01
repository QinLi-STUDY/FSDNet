**FSDNet: A Feature-Aware Attention Selection Network for Anomaly Detection on Printed Circuit Boards**

## Abstract
Self-supervised anomaly detection has emerged as a research hotspot in intelligent manufacturing and quality inspection, holding significant practical value in industrial applications.  However, anomaly detection in real-world printed circuit board (PCB) production environments remains challenging.  Existing methods often exhibit limited generalization when facing diverse anomaly types and environmental disturbances.  In addition, high model complexity and insufficient capability for multi-scale fine-grained defect recognition constrain their practical deployment. To address these issues, this paper proposes a novel self-supervised anomaly detection framework (FSDNet).  First, this paper proposes an Anomalous Sample Synthesizer Based on Diffusion Model (AnoDiff), which generates diverse and controllable anomalous samples to improve model generalization.  Second, this paper designs an Anomaly Feature Perception Module (AFPM) that selects discriminative channels from pretrained features, thereby reducing model complexity while enhancing detection performance.  Third, this paper proposes a Multi-scale Residual Reconstruction Network (MRRN) is developed to aggregate multi-scale features, improving sensitivity to fine-grained anomalies.  Finally, this paper proposes two novel attention-based modules: a Top-k Sparse \& Space Attention Module (TSSM) and a Gated Feature Enhancement Module (GFEM), both of which strengthen the discriminability and robustness of anomaly features. Experimental results demonstrate that the proposed method significantly outperforms state-of-the-art approaches on MVTec-AD, ViSA, and a self-constructed PCB dataset in terms of detection accuracy and robustness, validating its effectiveness and practical utility.
## 1. Installation

First create a new conda environment

    conda create -n fsdnet python=3.9
    conda activate fsdnet
    pip install -r requirements.txt
## 2.Dataset
### 2.1 MVTec-AD
- **Create the MVTec-AD dataset directory**. Download the MVTec-AD dataset from [MVTec-AD.zip](https://pan.baidu.com/s/1Q5lmmb9733ihJHhaAFQtHg?pwd=vuer). Unzip the file and move them to `./training/MVTec-AD/`. The MVTec-AD dataset directory should be as follows. 

```
|-- training
    |-- MVTec-AD
        |-- mvtec_anomaly_detection
            |-- bottle
                |-- ground_truth
                    |-- broken_large
                        |-- 000_mask.png
                    |-- broken_small
                        |-- 000_mask.png
                    |-- contamination
                        |-- 000_mask.png
                |-- test
                    |-- broken_large
                        |-- 000.png
                    |-- broken_small
                        |-- 000.png
                    |-- contamination
                        |-- 000.png
                    |-- good
                        |-- 000.png
                |-- train
                    |-- good
                        |-- 000.png
        |-- train.json
        |-- test.json
```

### 2.2 VisA
- **Create the VisA dataset directory**. Download the VisA dataset from [VisA.tar](https://pan.baidu.com/s/1Q5lmmb9733ihJHhaAFQtHg?pwd=vuer). Unzip the file and move them to `./training/VisA/`. The VisA dataset directory should be as follows. 

```
|-- training
    |-- VisA
        |-- visa
            |-- candle
                |-- Data
                    |-- Images
                        |-- Anomaly
                            |-- 000.JPG
                        |-- Normal
                            |-- 0000.JPG
                    |-- Masks
                        |--Anomaly 
                            |-- 000.png        
        |-- visa.csv
```

## 3. Training diffusion model
- Train the diffusion model on the MVTec-AD dataset: 
```
$ python -m torch.distributed.launch --nproc_per_node=2  train_diffusion.py --dataset MVTec-AD 

## 4. Strength-controllable Diffusion Anomaly Synthesis

Sample anomaly images using `1*RTX3090 GPU`:  
```
$ python -m torch.distributed.launch --nproc_per_node=1  sample.py --dataset MVTec-AD
``` 

## 5. Training FSDNet   
  
Train RealNet using `1*RTX3090 GPU`:  
```
$ python -m torch.distributed.launch --nproc_per_node=1  train_fsdnet.py --dataset MVTec-AD --class_name bottle
```

## 6. Evaluating FSDNet  
  
Calculating Image AUROC, Pixel AUROC, and PRO, and generating qualitative results for anomaly localization:
```
$ python  evaluation_fsdnet.py --dataset MVTec-AD --class_name bottle
```  

## Acknowledgements
We thank the great works [DiAD](https://lewandofskee.github.io/projects/diad) and [LDM](https://github.com/CompVis/latent-diffusion) for providing assistance for our research.
