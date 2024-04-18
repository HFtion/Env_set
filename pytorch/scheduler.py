import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from transformers import get_cosine_schedule_with_warmup

model = nn.Linear(10, 1)

optimizer = optim.SGD(model.parameters(), lr=0.1)
 
# 定义一个函数来测试学习率
def test_learning_rate(model, optimizer, learning_rate=0.001):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    iterations=100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.05*iterations, num_training_steps=iterations)
    for epoch in range(iterations):  
        print(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

test_learning_rate(model, optimizer, learning_rate=0.001)