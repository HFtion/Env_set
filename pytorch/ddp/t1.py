import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

def ddp_setup():
    os.environ["MASTER_ADDR"] = "localhost"
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def main():
    ddp_setup()
    id = int(os.environ['LOCAL_RANK'])

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    data_set = torchvision.datasets.MNIST("./", train=True, transform=trans, target_transform=None, download=True)

    net = torchvision.models.resnet101(num_classes=10)
    net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    net = net.cuda()

    data_loader_train = DataLoader(dataset=data_set, batch_size=32,shuffle=False,sampler=DistributedSampler(data_set))
    net = DDP(net, device_ids=[id]) 

    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(1000):
        data_loader_train.sampler.set_epoch(epoch)
        for i, data in enumerate(data_loader_train):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            opt.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            if i % 10 == 0 and id == 1:
                print("loss: {}".format(loss.item()))   
                snapshot = {"MODEL_STATE": net.module.state_dict(),
                            "EPOCHS_RUN": epoch}
                torch.save(snapshot, './1.pkl')

if __name__ == "__main__":
    main()
    destroy_process_group()

# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=gpu  --master_port=12355 src.py 