# Learning to discover structure

This repository contains a PyTorch implementation of a model that combines the work from: 

**Neural relational inference for interacting systems.**  
Thomas Kipf*, Ethan Fetaya*, Kuan-Chieh Wang, Max Welling, Richard Zemel.  
https://arxiv.org/abs/1802.04687  (*: equal contribution)


and 

**Neural Ordinary Differential Equations**
Ricky T. Q. Chen*, Yulia Rubanova*, Jesse Bettencourt*, David Duvenaud.
https://arxiv.org/pdf/1806.07366.pdf (*: equal contribution)

#### Abstract: 
Reasoning about objects and relations comes naturally to humans.This makes sense, since this in many ways is how the world is struc-tured. We use this ability to make compositional models of the worldin order to make predictions. Common neural network architecturesdo not have the ability to reason about relations. A popular paperfrom ICML this year [1] introduces a framework for such relationalreasoning using graphs. Another well-received paper [2] introducesan intuitive framework for parameterizing ordinary differential equa-tions with neural networks. We argue that utilizing these frameworksin combination seems like a good way of learning e.g. physics di-rectly from data, and examine this in some detail.

This repository contains code from https://github.com/ethanfetaya/NRI that is some the official implementation of the Neural relational inference for interacting systems paper. Check out their repository to see their full work on it. Here is an extract of their README that explains how to generate data. 

### Data generation

To replicate the experiments on simulated physical data, first generate training, validation and test data by running:

```
cd data
python generate_dataset.py
```
This generates the springs dataset, use `--simulation charged` for charged particles.

### Graph ODE
Encoder ODE GNN with supervised training is in odegnn-supervised.py. The full NRI with continuous time message passing is on odegnn.py, but does not work well (and bugs out the torchdiffeq package if used with adaptive-time solvers).
