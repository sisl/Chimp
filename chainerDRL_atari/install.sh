#!/bin/sh

pip install --upgrade pip

brew install numpy
pip install scipy
pip install chainer

pip install pillow


# cd external/SDL2-2.0.3
# ./configure
# make
# sudo make install
# cd ..
# cd ..

brew update && brew upgrade 

brew install sdl2
brew install sdl_gfx
brew install sdl_image

brew install sdl sdl_mixer sdl_ttf smpeg portmidi 
brew install pygame



git clone https://github.com/mgbellemare/Arcade-Learning-Environment ALE
cd ./ALE
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF .
#cp makefile.mac makefile
make -j2
pip install --user .
cd ..



