
# 1.install ubuntu
# When installing the system in Ubuntu, you only need to select 
# (swap, \, EFI)

# 2.Old three sentences
sudo apt-get update
sudo apt-get upgrade
sudo apt-get -f install 
gnome-system-monitor

# 3.install nvidia_driver
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall


# 4.install cuda(sometimes python cuda actions might use it)
# dowmload cuda_xx.xx.xx_xxxxxx.run from https://developer.nvidia.com/cuda-toolkit-archive
sudo sh cuda_10.2.89_xxxxxx.run #not install nvidia-driver
# find .bashrc ,add 
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 5.install cudnn
# download from https://developer.nvidia.com/rdp/cudnn-archive
unzip the cudnn-xxx.tgz
cd cudnn-xxx/
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include/ 
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/ 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h 
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

# 6.sometimes u might use cmake
# download from https://cmake.org/download/
unzip xxx
cd ./bootstrap
make -j 8 
sudo make install
#cmake --version

# 7 some app ushot download
#lantern : https://github.com/getlantern/lantern
#chrome 
#baiduyun
#miniconda
#vscode
#meld
#gnome-system-monitor

#7.5set vscode
# extension: chinese,comparefolders,rainbow brackets
# setting: Font Family= 'monospace', monospace
# tab size =4
#tree indent = 24

#8.change terminal color

#9 make time same step
timedatectl set-local-rtc 1

#10,shot alias,geidt bashrc
alias wn='watch -n 1 nvidia-smi'
alias t1='touch 1.txt'
alias dp='sudo dpkg -i '
alias ap='sudo apt-get '
conda activate voth

end
