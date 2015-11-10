# ChainerDRL

This is a decomposed implementation of a DQN agent. 

# Components

* Memories: scripts that implements experience replay
	* Currently, we support in-memory numpy arrays and HDF5 allocated storage

* Learners: the "brain" of the algorithm that does forward and backward passes in a neural net
	* Currently, we support DQN with fixed observation/action history
	* Planning to add LSTM

* Simulators: a variety of environments of varied complexity for agents to interact with
	* Single-player Arcade Learning Environment
	* Tiger POMDP problem
	* Intermittent support for agar.io

* Agents: the general framework that handles all interactions between a learner, a memory and a simulator.

See usage examples in run_... .py files

# Dependencies

The following Python packages are strictly required to use DRL.
* [Chainer](https://github.com/pfnet/chainer)
* NumPy
* SciPy

The following libraries are recommended (some functionality will be absent without them):
* Pygame
* CUDA
* Arcade Learning Environment

# Installation

We support non-CUDA installation on OSX and full installation on Ubuntu 14.04. See the corresponding .sh scripts. Note, installation on Ubuntu requires a reboot in the middle of the script. Therefore, one would need to enter commands that follow reboot in .sh script manually.

If you require sudo to run the code on an Ubuntu server, note that sudo resets the path variables. The recommended resolution is to run all scripts in 'sudo su' mode, which given one all the necessary permissions, and preserves the path.