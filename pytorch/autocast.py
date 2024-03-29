
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

for epoch in range(epochs):
    model.train()
    for i, (images, labels) in enumerate(loader_train):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            output = model(images)
            loss = criterion(output, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


# caution!
# https://blog.csdn.net/qq_44089890/article/details/130471991
# 上面说法有误，训练用的混合精度，测试加不加autocast都无所谓。因为混合精度不降低精度。区别是测试scaler不用加，因为scaler是确保反向传播时loss缩放到半精度表征范围内的限制。
# 即 test:
with autocast():
    output = model(images)
# and
output = model(images)
# 都可以

        

# caution！
# https://www.jianshu.com/p/6030e2db07e0
# 上面说法错误，autocast可以在dp和ddp中正常生效。
# 证据：https://blog.csdn.net/weixin_44878336/article/details/136071842
# 所以没有必要管那些修饰器、forward里面特殊的autocast操作
# 即，百无禁忌，随便使用，当成全精度使用都可以