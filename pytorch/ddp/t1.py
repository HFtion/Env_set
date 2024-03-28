import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def main():
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    data_set = torchvision.datasets.MNIST("./", train=True, transform=trans, target_transform=None, download=True)

    net = torchvision.models.resnet101(num_classes=10)
    net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    net = net.cuda()

    data_loader_train = DataLoader(dataset=data_set, batch_size=32,shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(1000):
        for i, data in enumerate(data_loader_train):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            opt.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            if i % 10 == 0 :
                print("loss: {}".format(loss.item()))   
                snapshot = {"MODEL_STATE": net.state_dict(),  # 由于多了一层 DDP 包装，通过 .module 获取原始参数 
                            "EPOCHS_RUN": epoch}
                torch.save(snapshot, './1.pkl')

if __name__ == "__main__":
    main()
