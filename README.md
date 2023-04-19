# comp2field
**Basic requirements**
- Python3
- Tensorflow=2.0.0
- CUDA 10.0
- Other requirements can be found at requirement.txt

**Image requirements**
- For both training and testing, geometry image (256, 256) and field image (256, 256) should be stitched into one image with size (256, 512).
- If images with size (512, 512) are used, replace the corresponding python codes with codes in "HIGH_RESO"
- Make sure the solid black background is in good proportion in order to match pixes.
- Large deformance is not recommended. 
- Make sure they are named in order. e.g. 1.png, 2.png,...

**Before training or testing**
- Specify the directory path which contains data set in config.py. e.g. PATH="./MISES/"
- split your data set into training set and test set under dataset directory:
```
cd MISES
mkdir train ### training set dir
mkdir test ### test set dir
## move you data into these two folders (You can also not use two commands below and customize)##
cp split.sh ./MISES
bash split.sh
```
- Make sure images in training set and test set are named in order separately. e.g. 1.png, 2.png,...

**Training**
- Specify epochs in config.py. e.g. EPOCHS = 150
- Start training by typing
```
python train.py
```
- The training checkpoints will be stored
- Checking the training status using Tensorboard:
```
tensorboard --logdir logs/fit
```

**Testing/prediction**
- The models ara stored with training checkpoints during training, all checkpoints of the pretrained models are stored in link:[https://www.dropbox.com/sh/4g8hyh6lbc5tn07/AACDMQZKtsiniZXpz7peyxk-a?dl=0]. Please check "Dataset availablity" for available pretrained model
- Copy training checkpoint(e.g. MISES stress field):
```
mkdir training_checkpoints
mkdir predict
cp DOWNLOAD_PATH/ckpt/MISES/* training_checkpoints
```
- Specify the number of data in test set in test.py. e.g. num_data = 5, so in test set, there will be 5 images: 1.png,...,5.png
- Run prediction on test set by typing:
```
python test.py
```
- The predcition will be in folder "./predict", stitched image from left to right is geometry, ground truth and prediction
- To change the checkpoint used for predictions, open "./training_checkpoints/checkpoint" to specify the checkpoint

**Dataset availablity**
- Available dataset in https://www.dropbox.com/sh/ffcks4lm85x440r/AAANq4qYBlTb8KeOTavsJ9jta?dl=0 and training checkpoints in https://www.dropbox.com/sh/4g8hyh6lbc5tn07/AACDMQZKtsiniZXpz7peyxk-a?dl=0:

| field description | shape of units | boundary condition embedded | loading condition | strain magnitude | materials property | ratio of young's modulus | resolution | dataset folder | checkpoints folder |
| ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |-------------|-------------|-------------|
| MISES stress  | square | no | uniaxial compression in x | 10% | crushable foam | 4.75 | 8 * 8 | /dataset/MISES | /ckpt/MISES  |
| S11 | square | no | uniaxial compression in x | 10% | crushable foam | 4.75 | 8 * 8 | /dataset/S11 | /ckpt/S11  |
| S12 | square | no | uniaxial compression in x | 10% | crushable foam | 4.75 | 8 * 8 | /dataset/S12 | /ckpt/S12  |
| PE11 | square | no | uniaxial compression in x | 10% | crushable foam | 4.75 | 8 * 8 | /dataset/PE11 | /ckpt/PE11  |
| PE12 | square | no | uniaxial compression in x | 10% | crushable foam | 4.75 | 8 * 8 | /dataset/PE12 | /ckpt/PE12  |
| MISES stress | square| yes | uniaxial compression in x and nanoindentation (spherical indentor)| 5% | crushable foam | 4.75 | 8 * 8 | /dataset/BC | /ckpt/BC |
| MISES stress | hexagon | no | uniaxial compression in x | 10% | crushable foam | 4.75 | 8 * 8 | /dataset/HEXAGON | /ckpt/HEXAGON  |
| MISES stress | triangle| no | uniaxial compression in x | 5% | crushable foam | 4.75 |  8 * 8 | /dataset/TRIANGLE | /ckpt/TRIANGLE |
| MISES stress | square | no | uniaxial compression in x | 5% | crushable foam | 4.75 | 32 * 32 | /dataset/HIGH_RESO| /ckpt/HIGH_RESO |
| S11 | square | no | uniaxial tensile in x | 10% | linear elasticity | 4.75 | 32 * 32 | /dataset/ELASTIC_TENSOR/ELASTIC_TENSOR/S11 | /ckpt/ELASTIC_TENSOR/ELASTIC_TENSOR/S11 |
| S12 | square | no | uniaxial tensile in x | 10% | linear elasticity | 4.75 | 32 * 32 | /dataset/ELASTIC_TENSOR/ELASTIC_TENSOR/S12 | /ckpt/ELASTIC_TENSOR/ELASTIC_TENSOR/S12 |
| S22 | square | no | uniaxial tensile in x | 10% | linear elasticity | 4.75 | 32 * 32 | /dataset/ELASTIC_TENSOR/ELASTIC_TENSOR/S22 | /ckpt/ELASTIC_TENSOR/ELASTIC_TENSOR/S22 |
| S33 | square | no | uniaxial tensile in x | 10% | linear elasticity | 4.75 | 32 * 32 | /dataset/ELASTIC_TENSOR/ELASTIC_TENSOR/S33 | /ckpt/ELASTIC_TENSOR/ELASTIC_TENSOR/S33 |
| LE11 | square | no | uniaxial tensile in x | 10% | linear elasticity | 4.75 | 32 * 32 | /dataset/ELASTIC_TENSOR/ELASTIC_TENSOR/LE11 | /ckpt/ELASTIC_TENSOR/ELASTIC_TENSOR/LE11 |
| LE12 | square | no | uniaxial tensile in x | 10% | linear elasticity | 4.75 | 32 * 32 | /dataset/ELASTIC_TENSOR/ELASTIC_TENSOR/LE12 | /ckpt/ELASTIC_TENSOR/ELASTIC_TENSOR/LE12 |
| LE22 | square | no | uniaxial tensile in x | 10% | linear elasticity | 4.75 | 32 * 32 | /dataset/ELASTIC_TENSOR/ELASTIC_TENSOR/LE22 | /ckpt/ELASTIC_TENSOR/ELASTIC_TENSOR/LE22 |
| S11 | square | no | uniaxial tensile in y | 10% | linear elasticity | 4.75 | 32 * 32 | /dataset/ELASTIC_TENSOR/TENSILE_Y/S11 | /ckpt/ELASTIC_TENSOR/TENSILE_Y/S11_BC |
| S22 | square | no | uniaxial tensile in y | 10% | linear elasticity | 4.75 | 32 * 32 | /dataset/ELASTIC_TENSOR/TENSILE_Y/s22 | /ckpt/ELASTIC_TENSOR/TENSILE_Y/S22_BC |
| S11_ratio (uniform ratio) | square | no | uniaxial tensile | 10% | linear elasticity | 4.75 | 32 * 32 | /dataset/ELASTIC_TENSOR/UNIFORM_RATIO/S11 | /ckpt/ELASTIC_TENSOR/UNIFORM_RATIO/S11_ratio |
| S11_hier (hierarchical) | square | no | uniaxial tensile | 10% | linear elasticity | 4.75 | 32 * 32 | /dataset/ELASTIC_TENSOR/HIERARCHICAL/S11 | /ckpt/ELASTIC_TENSOR/HIERARCHICAL/8_16_32_3000data|
| LE11_crack | square | no | uniaxial tensile | 5% | linear elasticity | 100:10:1 | 32 * 32 | /dataset/CRACK/LE11_1_10_100 | /ckpt/CRACK/LE11_1_10_100 |
| LE11_crack | square | no | uniaxial tensile | 5% | linear elasticity | 100:10:0.1 | 32 * 32 | /dataset/CRACK/LE11_0.1_10_100 | /ckpt/CRACK/LE11_0.1_10_100 |
| S11_Field2geo | square | no | uniaxial tensile | 10% | linear elasticity | 4.75 | 32 * 32 | /dataset/FIELD2GEO/S11 | /ckpt/FIELD2GEO/S11 |
