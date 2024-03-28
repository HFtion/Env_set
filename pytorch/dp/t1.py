import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

input_size = 5000
output_size = 20000

batch_size = 32
data_size = 128

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

model = Model(input_size, output_size)
model = nn.DataParallel(model, device_ids=[0,1], output_device=[1])

model.cuda()

for data in rand_loader:
    input = data.cuda()
    output = model(input)

    print("Outside: input size", input.size(),
          "output_size", output.size())
