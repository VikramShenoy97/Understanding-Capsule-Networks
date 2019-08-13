# Understanding Capsule Networks

Constructed and trained a capsule network to predict digits from the MNIST dataset.

## Overview

A Capsule Network is basically a neural network that tries to perform inverse graphics(Process of converting a visual image to some internal hierarchical representation of geometric data). It understands relative relationships between objects. Capsule Networks use vectors called capsules that incorporate all the important information about the state of the feature they are detecting. A capsule is any function that tries to predict the presence and instantiation parameters of a particular object at any given location. The architecture consists of an encoder network and a decoder network. The forward pass of the combined network is computed using the dynamic routing algorithm.

![Capsule_Network_Encoder](https://github.com/VikramShenoy97/Understanding-Capsule-Networks/blob/master/Media/Encoder_Capsnet.png)

*Fig 1. The CapsNet Architecture (Encoder) from the original paper by S Sabour et al., 2017.*


![Capsule_Network_Decoder](https://github.com/VikramShenoy97/Understanding-Capsule-Networks/blob/master/Media/Decoder_Capsnet.png)

*Fig 2. The CapsNet Architecture (Decoder) from the original paper by S Sabour et al., 2017.*


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

For using this project, you need to install PyTorch and Plotly.

```
pip install torch torchvision
pip install plotly
```

### Dataset

The [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.


## Training

Run the script *capsulenetwork.py* with mode="Train" in the terminal as follows.

```
Python capsulenetwork.py
```

### Training Performance

Each epoch takes about 87 seconds on average when using Google Colab's GPU.


The reconstructions for every 10th epoch are stored in the **Training** folder.

![training_epochs](https://github.com/VikramShenoy97/Understanding-Capsule-Networks/blob/master/Media/epochs.gif)

*Fig 3. Reconstructions for every 10th epoch.*

After 100 Epochs:
```
Final Training Accuracy = 99.91%
Final Training Loss = 0.4595620
```

### Training Performance Graph
![training_graph](https://github.com/VikramShenoy97/Understanding-Capsule-Networks/blob/master/Graphs/Training_Graph.png)

*Fig 4. Training Loss and Training Accuracy Graph*

## Testing

Run the script *capsulenetwork.py* with mode="Test" in the terminal as follows.
```
Python capsulenetwork.py
```

### Testing Performance

```
Test Set Accuracy = 98.80%
```

### Accuracy Graph

![accuracy_graph](https://github.com/VikramShenoy97/Understanding-Capsule-Networks/blob/master/Graphs/Accuracy_Graph.png)

*Fig 5. Training Accuracy vs Testing Accuracy Graph*

## Results

Fig. 6 Ground Truth Image        |  Fig 7. Reconstructed Image
:-------------------------:|:-------------------------:
![](https://github.com/VikramShenoy97/Understanding-Capsule-Networks/blob/master/Output_Images/Ground_Truth_Images.png)  |  ![](https://github.com/VikramShenoy97/Understanding-Capsule-Networks/blob/master/Output_Images/Reconstructed_Images.png)

## Understanding Dimensions of the Capsule Vector

Each capsule in the Digit Capsule Layer is a 16-Dimensional Vector. By holding 15 dimensions constant and slightly varying one dimension, we can understand the property captured by that dimension as shown below:


There is my interpretation of what some of these dimensions capture.

Fig. 8 Dimension 4 (Localised Skew)        |  Fig 9. Dimension 5 (Curvature)
:-------------------------:|:-------------------------:
![](https://github.com/VikramShenoy97/Understanding-Capsule-Networks/blob/master/Dimensional_Reconstructions/Pose_Reconstructions_for_dimension_4.png)  |  ![](https://github.com/VikramShenoy97/Understanding-Capsule-Networks/blob/master/Dimensional_Reconstructions/Pose_Reconstructions_for_dimension_5.png)


Fig. 10 Dimension 7 (Stroke & Thickness)   |  Fig 11. Dimension 9 (Edge Translation)
:-------------------------:|:-------------------------:
![](https://github.com/VikramShenoy97/Understanding-Capsule-Networks/blob/master/Dimensional_Reconstructions/Pose_Reconstructions_for_dimension_7.png)  |  ![](https://github.com/VikramShenoy97/Understanding-Capsule-Networks/blob/master/Dimensional_Reconstructions/Pose_Reconstructions_for_dimension_9.png)


## Built With

* [PyTorch](https://pytorch.org) - Deep Learning Framework
* [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb) - Cloud Service

## Authors

* **Vikram Shenoy** - *Initial work* - [Vikram Shenoy](https://github.com/VikramShenoy97)

## Acknowledgments

* Project is inspired by **Sara Sabour**, **Nicholas Frosst**, and **Geoffrey E Hinton**'s paper, [*Dynamic Routing Between Capsules*](https://arxiv.org/pdf/1710.09829.pdf)
* Initial understanding of Capsule Networks was made easy through **Aurélien Géron**'s [*Youtube Video on Capsule Networks*](https://www.youtube.com/watch?v=pPN8d0E3900).
* Procured an in-depth understanding of Capsule Networks and dynamic routing algorithm through **Max Pechyonkin**'s blog, [*Understanding Hinton’s Capsule Networks.*](https://medium.com/ai³-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b)
* Referenced **Gram.AI**'s [*code*](https://github.com/gram-ai/capsule-networks) for some details.
