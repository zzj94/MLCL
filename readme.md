# Official implementation for MLCL

## [Exploiting Multi-level Consistency Learning for Source-free Domain Adaptation]



### Prerequisites:
- python == 3.7.16
- pytorch ==1.10.0
- torchvision == 0.11.1
- numpy, scipy, sklearn, PIL, argparse, math, random

### Dataset:

- Please manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-2017](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification), [DomainNet](https://paperswithcode.com/dataset/domainnet) from the official websites, and modify the path of images in each '.txt' under the folder './data/'.


### Training:
2. ##### Unsupervised Closed-set Domain Adaptation (UDA) on the Office/ Office-Home dataset
	- Train model on the source domain **A** (**s = 0**)
    ```python
   cd object/
   python mlcl_source.py --trte val --da uda --output ckps/source/ --gpu_id 0 --dset office --max_epoch 100 --s 0
    ```
	
	- Adaptation to other target domains **D and W**, respectively
    ```python
   python mlcl_target.py --da uda --dset office --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/ --cross_par 0.1 --aug_par 0.3 --psva 0.6 --ent_par 1
    ```
   
3. ##### Unsupervised Closed-set Domain Adaptation (UDA) on the VisDA-C dataset
	- Synthetic-to-real 
    ```python
    cd object/
	 python mlcl_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset VISDA-C --net resnet101 --lr 1e-3 --max_epoch 10 --s 0
	python mlcl_target.py --da uda --dset VISDA-C --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/ --net resnet101 --lr 1e-3 --cross_par 0.1 --aug_par 0.3 --psva 0.1 --ent_par 1
	 ```
	
