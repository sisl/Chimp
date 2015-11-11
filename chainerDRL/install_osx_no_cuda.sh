#!/bin/sh

''' Install on Mac, no CUDA '''

# sudo chown -R $USER /usr/local/

brew update && brew upgrade 
pip install --upgrade pip

brew install numpy
brew install scipy

pip install chainer

pip install pillow
pip install -U six

brew install sdl2
brew install sdl_gfx
brew install sdl_image

brew install sdl sdl_mixer sdl_ttf smpeg portmidi 
brew install pygame

git clone https://github.com/mgbellemare/Arcade-Learning-Environment ALE
cd ./ALE
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF .
make -j 4
pip install --user .
cd ..
