# End-to-end Prediction of Various Mechanical Fields from Hierarchical Composite Geometry using Adverserial Neural Networks

Yang, Z., Yu, C. H., & Buehler, M. J. (2021). Deep learning model to predict complex stress and strain fields in hierarchical composites. Science Advances, 7(15), eabd7416, 2021, https://www.science.org/doi/10.1126/sciadv.abd7416

Materials-by-design is a paradigm to develop previously unknown high-performance materials. However, finding materials with superior properties is often computationally or experimentally intractable because of the astronomical number of combinations in design space. Here we report an AI-based approach, implemented in a game theory–based conditional generative adversarial neural network (cGAN), to bridge the gap between a material’s microstructure—the design space—and physical performance. Our end-to-end deep learning model predicts physical fields like stress or strain directly from the material microstructure geometry, and reaches an astonishing accuracy not only for predicted field data but also for derivative material property predictions. Furthermore, the proposed approach offers extensibility by predicting complex materials behavior regardless of component shapes, boundary conditions, and geometrical hierarchy, providing perspectives of performing physical modeling and simulations. The method vastly improves the efficiency of evaluating physical properties of hierarchical materials directly from the geometry of its structural makeup.

![Overall workflow](https://github.com/lamm-mit/FieldPredictorGAN/blob/main/flow_chart.png)

**Basic requirements**
- Python3
- Tensorflow=2.0.0
- CUDA 10.0
- Tensorboard=2.6.0
- Other requirements can be found at requirement.txt and installation can be done via:
```
pip install -r requirement.txt
```

**Image dataset requirements**
- For both training and testing, geometry image (256, 256) and field image (256, 256) need to be stitched into one image with size (256, 512).
- If images with size (512, 512) are used, replace the corresponding python codes with codes in "HIGH_RESO". 
- Make sure the solid black background is in good proportion in order to match pixes. 
- Make sure they are named in order. e.g. 1.png, 2.png,...

**Before training or testing**
- Hyperparameters are indicated in "config.py".
- Specify the directory path which contains data set in config.py. e.g. PATH="./MISES/"
- Split your data set into training set and test set under dataset directory:
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
- Specify epochs in "config.py". e.g. EPOCHS = 150
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

```
@article{YangYuBuehlerScienceAdv_2021,
    title   = {Deep learning model to predict complex stress and strain fields in hierarchical composites},
    author  = {Z. Yang, C.H. Yu, and M.J. Buehler},
    journal = {Science Advances},
    year    = {2021},
    volume  = {7},
    pages   = {eabd7416},
    url     = {https://www.science.org/doi/10.1126/sciadv.abd7416}
}

@article{YangYuBuehlerJMPS_2022,
    title   = {End-to-end deep learning method to predict complete strain and stress tensors for complex hierarchical composite microstructures},
    author  = {Z. Yang, C.H. Yu, K. Guo, and M.J. Buehler},
    journal = {Journal of the Mechanics and Physics of Solids},
    year    = {2021},
    volume  = {154},
    pages   = {104506},
    url     = {https://doi.org/10.1016/j.jmps.2021.104506}
}
```
