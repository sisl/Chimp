# Chimp

Chimp is a general purpose framework for deep reinforcement learning developed at the [Stanford Intelligent Systems Laboratory](http://sisl.stanford.edu/).
Chimp is based on a simple three-part architecture to allow plug-and-play like capabilities for deep reinforcement
learning experiments. 
This package was inspired by the Google DeepMind [paper](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) (V. Mnih, et al). 
Many of the architectural ideas were taken from DeepMind's
[GORILA](http://www.humphreysheil.com/blog/gorila-google-reinforcement-learning-architecture) framework and from the
[paper](http://arxiv.org/pdf/1508.04186.pdf) on distributed Deep Q-Learning by Ong, et al. 

# Architecture 
Chimp consists of three main modules: Learner, Simulator and Memory. 

# How it Works

What makes Chimp powerful is that it can handle arbitrary inputs to the deep Q-Netwrok. 
The user specifies the history lengths for observetations, actions and even rewards that they want to use as inputs to the DQN, and Chimp handles the rest. The specification is in the form of a tuple ```(s_size, a_size, r_size)```. For the DeepMind Atari experiments, this setting would look like (4,0,0), they used four frames per input and no action or reward history. 

# Dependencies

Chimp relies on existing deep learning back-ends. Currently only [Chainer](http://chainer.org/) is supported. Support
for tensor flow is planned.

The following Python packages are required to use PODRL.
* [Chainer](https://github.com/pfnet/chainer)
* NumPy

# Authors

The original authors of this software are: Yegor Tkachenko, Max Egorov, Hao Yi Ong.

# License

The software is distributed under the Apache License.
