
# 1.install ubuntu
# When installing the system in Ubuntu, you only need to select 
# (swap, \, EFI)

# 2.Old three sentences
sudo apt-get update
sudo apt-get upgrade
sudo apt-get -f install 

# 3.install nvidia_driver
# Open the settings, find the software update, replace the source and install the driver

# 4.install cuda_cudnn(sometimes python might use it)
# dowmload cuda_xx.xx.xx_xxxxxx.run
sudo sh cuda_10.2.89_xxxxxx.run
gedit ~/.bashrc
# +export CUDA_HOME=/usr/local/cuda
# +export PATH=$PATH:$CUDA_HOME/bin
# +export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc
# then test
cd /usr/local/cuda/samples/1_Utilities/deviceQuery 
sudo make
./deviceQuery
# then cudnn 
unzip the cudnn-xxx.tgz
cd cudnn-xxx/
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include/ 
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/ 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h 
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

# 5.install camke_opencv(sometimes python,qt might use it)
sudo apt-get install libssl-dev
# Download the latest cmake website
# ubzip cmake-xxx.tar.gz to your home dir.
cd cmake-xxx
sudo ./bootstrap
sudo make
sudo make install
# download opencv
sudo apt-get install libgtk2.0-dev pkg-config  
cd opencv-xxx
mkdir build
cd build
sudo cmake .. -D BUILD_TIFF=ON  
sudo make -j64
sudo make install