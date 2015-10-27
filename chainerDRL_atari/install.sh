#!/bin/sh

# sudo chown -R $USER /usr/local/

brew update && brew upgrade 
pip install --upgrade pip


brew install numpy
brew install scipy

pip install chainer

pip install pillow
pip install -U six


# cd external/SDL2-2.0.3
# ./configure
# make
# sudo make install
# cd ..
# cd ..

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


# git clone https://github.com/inducer/pycuda pycuda
# cd pycuda
# python configure.py --cuda-root=/usr/local/cuda/
# may need to edit path from lib64 to lib
# sudo install_name_tool -change @rpath/CUDA.framework/Versions/A/CUDA \
#    /Library/Frameworks/CUDA.framework/CUDA \
#    /usr/local/cuda/lib/libcuda.dylib
# sudo make install
# cd ..

# pip install chainer-cuda-deps
# refer to here http://www.slideshare.net/beam2d/introduction-to-chainer-a-flexible-framework-for-deep-learning

