# bms-molecular-translation

## Description
* Train object detection model with RDKit

Detect atoms and bonds from image using Faster RCNN with ResNet50 backbone.  
Use RDKit and Torchvision to train and infer the correponding InChI string.

* main.py - use DDP to train and infer the image
* data_loader.py - prepare train, valid, and test dataset and loader 
* models.py - get Faster RCNN model 
* inference.py - infer the molecular graph and InChI form from the prediction of detection model
* detection_label_generator.py - generate noisy molecular images and annotations with InChI text input using RDKit 
* test_image_resizer.py - resize test images to fit 300 x 300 size preserving original aspect ratio

## Command

* Environment setting

`conda env create -f bms_env.yaml`

`conda activate bms`

RDKit should be installed individually from [here](https://www.rdkit.org/docs/Install.html).

* Generating image dataset and labels for training

`python3 detection_label_generator.py`

* Train

`python3 -m torch.distributed.launch --nproc_per_node=NUMBER_OF_PROCESSES --use_env main.py`

Make sure to use proper data path in `data_loader.py`.

* Resize test images into 300 x 300

`python3 test_image_resizer.py`

* Inference

`python3 main.py --test --load_path PATH_OF_TRAINED_MODEL`

## Dependency

- Python 3.8.5
- PyTorch 1.8.1+cu102
- RDKit 2020.09.1

all other requirements are described in `requirements.txt`
