
# 1.install miniconda
# download  miniconda from https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/
# install

# 2.install vscode,lantern
# sudo dpkg -i xxx

# 3.change conda，pip source
gedit ~/.condarc
gedit ~/pip/pip.conf 


# 4.conda create a new visual env
conda create -n name python=3.8

# 5.cuda-cudnn-pytorch-torchvision
# first and simple
# copy from https://pytorch.org/get-started/locally/
# if some err happen , find that source,download,and pip install local

# second pip install from local
# download from https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/
# find those Bytesize > 1G ,it combine with cuda,pytorch 



