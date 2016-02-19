# Chimp

Chimp is a general purpose framework for deep reinforcement learning developed at the [Stanford Intelligent Systems Laboratory](http://sisl.stanford.edu/).

# Architecture 
Chimp is based on a simple three-part architecture to allow plug-and-play like capabilities for deep reinforcement
learning experiments. 
This package was inspired by the Google DeepMind [paper](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) (V. Mnih, et al). 
Many of the architectural ideas were taken from DeepMind's
[GORILA](http://www.humphreysheil.com/blog/gorila-google-reinforcement-learning-architecture) framework and from the
[paper](http://arxiv.org/pdf/1508.04186.pdf) on distributed Deep Q-Learning by Ong, et al. 
Chimp consists of three main modules: Learner, Simulator and Memory. 

# Dependencies

Chimp relies on existing deep learning back-ends. Currently only [Chainer](http://chainer.org/) is supported. Support
for tensor flow is planned.

The following Python packages are required to use PODRL.
* [Chainer](https://github.com/pfnet/chainer)
* NumPy

The creators of this software are: Yegor Tkachenko, Max Egorov, Hao Yi Ong.

