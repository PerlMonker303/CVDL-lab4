# Training attempts

Notes: 
- Each attempt (1-10) represents a different model and has a corresponding architecture (*results/model_n_ok.png*)
and the corresponding checkpoint (*saved/checkpoint_model_n_ok*).
- Images are resized using padding in order to contain the whole image without cropping parts of it

| Attempt  | Learning rate |   Regularization  | Epochs | Loss  |  Train Accuracy (top 1) | Val Accuracy (top 1) | Batch Size | Optimizer | Image size  | Filters   |  Classes
|----------|:-------------:|------------------:|-------:|------:|------------------------:|---------------------:|-----------:|----------:|-----------:|--------------:|---------:|
| 1        | 1e-4          |   -               | 100   | 0.45   |   81%                   | 72%                  |  128       | Adam      | 32x32      | 32,64,128     |    37
| 2        | 1e-2 -> 1e-6  |  BN, SGDR         | 50    | 0.4146 |   85.09%                | 65.62%               |  128       | Adam      | 32x32      | 32,64,128     |    37
| 3        | 1e-3 -> 1e-7  | Dropout, BN, SGDR | 200   | 0.25   |   90.62%                | 68.08%               |  128       | Adam      | 32x32      | 32,64,128     |    37
| 4        | 1e-4          |  Dropout, BN      | 15    | 0.5063 |   76.06%                | 68.75%               |  128       | Adam      | 32x32      |    -          |    37
| 5        | 1e-2          |  Dropout, BN, SGDR| 5     | 0.2234 |   67.77%                | 67.69%               |  4         | Adam      | 256x256    |16,32,64       |    2 (cat/dog)
| 6        | 1e-3          |  Dropout, BN, SGDR| 5     | 5.1946 |   67.77%                | 67.79%               |  4         | Adam      | 256x256    |16,32,64       |    2 (cat/dog)
| 7        | 1e-4          |  Dropout, BN      | 15    | 0.0416 |   93.91%                | 98.91%               |  4         | Adam      | 224x224    |    -          |    2 (cat/dog)
| 8        | 1e-4          |  Dropout, BN      | 15    | 0.0110 |   98.15%                | 99.13%               |  4         | Adam      | 224x224    |    -          |    2 (cat/dog)
| 10       | 1e-4          |  Dropout, BN      | 71    | 0.5432 |   85.66%                | 67.58%               |  128       | Adam      | 32x32      |    -          |    37
\*4,7 - Transfer Learning on EfficientNetB0; \*8,10 - Fine-tuning

# Transfer learning
I chose the architecture EfficientNet (B0), because, as the name suggests, it is an efficient model in terms of FLOPS for inference.
The authors who proposed this architecture explored multiple ways of rescaling network layers to obtain the best trade-off
between accuracy and network size. 
The key features of this architecture are:
- light-weight network with a small inference time
- uses compound scaling which uniformly scales the depth, width and resolution of the network through a compound coefficient (balances width, height and depth)
- comes in multiple flavours (B0-B7) based on the input size. (B7 achieves the best accuracy and beats SOTA on ImageNet)
- I chose the B0 version because, as they mention in the Keras documentation, tuning a larger model is harder when you have
a small number of classes and data (our scenario)

Fine tuning was performed by unfreezing the last 20 layers of EfficientNet, besides the batch normalization layers.

Resources used: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/


# Ensambles
For the ensamble network, the following models were used: 2, 3, 4, 10.
These models are used on the evaluation dataset, then their results are averaged together (mean).
Accuracy obtained: 71.32%, which is larger by more than 1% than any of the validation accuracies obtained
on the models 2, 3, 4 and 10 separately.
