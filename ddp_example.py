################
## main.py文件
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
# 新增：
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

### 1. 基础模块 ### 
# 假设我们的模型是这个，与DDP无关
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        return x

# 假设我们的数据是这个
class MyDataSet(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = torch.rand(1000, 64)
        self.label = torch.LongTensor([1 for _ in range(1000)])

    def __getitem__(self, index: int):
        return self.data[index], self.label[index]

    def __len__(self):
        return 1000

def get_dataset():
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
    #     download=True, transform=transform)
    my_trainset = MyDataSet()
    # DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。
    #      用，就完事儿！sampler的原理，第二篇中有介绍。
    train_sampler = DistributedSampler(my_trainset)
    # DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。
    #      也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    trainloader = torch.utils.data.DataLoader(my_trainset, 
        batch_size=16, num_workers=2, sampler=train_sampler)
    return trainloader
    
### 2. 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# DDP：DDP backend初始化
# torch.cuda.set_device(local_rank)
# dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
dist.init_process_group('gloo')

# 准备数据，要在DDP初始化之后进行
trainloader = get_dataset()

# 构造模型
# model = ToyModel().to(local_rank)
model = ToyModel()
# DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))
# DDP: 构造DDP model
# model = DDP(model, device_ids=[local_rank], output_device=local_rank) ## for GPU
model = DDP(model) ## for CPU

# DDP: 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 假设我们的loss是这个
# loss_func = nn.CrossEntropyLoss().to(local_rank)
loss_func = nn.CrossEntropyLoss()

### 3. 网络训练  ###
model.train()
iterator = tqdm(range(100))
import time
start = time.time()
for epoch in iterator:
    # DDP：设置sampler的epoch，
    # DistributedSampler需要这个来指定shuffle方式，
    # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
    trainloader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    step = 0
    for data, label in trainloader:
        # data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label)
        loss.backward()
        iterator.desc = "loss = %0.3f" % loss
        optimizer.step()
        step += 1
    # DDP:
    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
    # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "%d.ckpt" % epoch)

print ('*' * 10, time.time() - start, '*' * 10)

################
## Bash运行
# DDP: 使用torch.distributed.launch启动DDP模式
# 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 main.py