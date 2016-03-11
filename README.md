# Chimp

Chimp is a general purpose framework for deep reinforcement learning developed at the [Stanford Intelligent Systems Laboratory](http://sisl.stanford.edu/).
Chimp is based on a simple three-part architecture to allow plug-and-play like capabilities for deep reinforcement
learning experiments. 
This package was inspired by the Google DeepMind [paper](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) (V. Mnih, et al). 
Many of the architectural ideas were taken from DeepMind's
[GORILA](http://www.humphreysheil.com/blog/gorila-google-reinforcement-learning-architecture) framework and from the
[paper](http://arxiv.org/pdf/1508.04186.pdf) on distributed Deep Q-Learning by Ong, et al. 

# Installation

ToDo

# Architecture 

Chimp consists of four main modules: Agent, Learner, Simulator and Memory. Such decomposition leads to a very powerful and flexible framework for reinforcement learning experiments, where one can quickly switch between simulators, replay memory implementations, and various deep learning backends.

Chimp is also powerful in its flexible handling of inputs to the deep neural network. 
The user can specify the history lengths for observations, actions and even rewards that they want to use as inputs to the model, and Chimp will handle the rest. 

The specification of the input size is in the form of a tuple ```(s_size, a_size, r_size)```. For the DeepMind Atari experiments, this setting would look like (4,0,0), they used four image frames per input and no action or reward history. 

# Components

* Memory: script that implements experience replay
	* Currently, we support in-memory numpy arrays and HDF5 allocated storage

* Learner: the "brain" of the algorithm that does forward and backward passes in a neural net
	* We support DQN with arbitrary observation/action/reward history as input
	* Planning to add LSTM + actor-critic framework

* Simulator: environment for the agent to interact with
	* Single-player Arcade Learning Environment
	* Tiger POMDP problem

* Agent: the general framework that handles all interactions between a learner, a memory and a simulator.

# Dependencies

Chimp relies on existing deep learning back-ends. Currently only [Chainer](http://chainer.org/) is supported. Support
for TensorFlow is on the way.

The following Python packages are required to use PODRL.
* [Chainer](https://github.com/pfnet/chainer)
* NumPy
* SciPy

The following libraries are recommended (some functionality will be absent without them):
* Pygame
* CUDA
* Arcade Learning Environment

# Authors

The original authors of this software are: Yegor Tkachenko, Max Egorov, Hao Yi Ong.

# License

The software is distributed under the Apache License 2.0
