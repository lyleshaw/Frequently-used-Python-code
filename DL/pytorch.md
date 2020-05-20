# pytorch

> 请注意：该版本为粗略提交版本，未对普适情况做修改，如需使用请仔细分析修改。

**colab挂载google drive**

```python
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
!mkdir -p drive
!google-drive-ocamlfuse drive
import os
import sys
os.chdir('drive/Colab Notebooks')
```

---

**引入依赖包**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F 
import pandas as pd 
import numpy as np 
from torch.utils.data import *
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
```

---

**数据读入**

```python
```图像类
#修改图片为3*224*224
data_transform = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
#从data/train读入训练集
train_img = torchvision.datasets.ImageFolder('data/train',
                                            transform=data_transform
                                            )
test_img = torchvision.datasets.ImageFolder('data/val',
                                            transform=data_transform
                                            )
#从data/val读入测试集
train_data = torch.utils.data.DataLoader(train_img, batch_size=4,shuffle=True)
test_data = torch.utils.data.DataLoader(test_img, batch_size=4,shuffle=True)
```END

```其他数据
tr_data = pd.read_csv("./data/train/outfile5.csv",encoding='gbk')
train = pd.read_csv('./data/train/outfile5.csv',usecols=[1,2],encoding='gbk')
train_label = pd.read_csv('./data/train/outfile5.csv',usecols=[29],encoding='gbk')

te_data = pd.read_csv("./data/val/lval.csv",encoding='gbk')
test = pd.read_csv('./data/val/lval.csv',usecols=[1,2],encoding='gbk')
test_label = pd.read_csv('./data/val/lval.csv',usecols=[29],encoding='gbk')

train_tensor = torch.Tensor(train.values)
train_label_tensor = torch.Tensor(train_label.values)
test_tensor = torch.Tensor(test.values)
test_label_tensor = torch.Tensor(test_label.values)

train_set = TensorDataset(train_tensor,train_label_tensor)
train_data = torch.utils.data.DataLoader(train_set, batch_size=batchsize,shuffle=True)
test_set = TensorDataset(test_tensor,test_label_tensor)
test_data = torch.utils.data.DataLoader(test_set, batch_size=batchsize,shuffle=True)
```END
```

---

**乱序数据**

```python
random_order = list(range(len(input_ids)))
np.random.seed(1984)
np.random.shuffle(random_order)

input_ids_train = np.array([input_ids[i] for i in random_order[:int(len(input_ids)*0.95)]])
input_types_train = np.array([input_types[i] for i in random_order[:int(len(input_ids)*0.95)]])
input_masks_train = np.array([input_masks[i] for i in random_order[:int(len(input_ids)*0.95)]])
y_train = np.array([label[i] for i in random_order[:int(len(input_ids) * 0.95)]])
print(input_ids_train.shape, input_types_train.shape, input_masks_train.shape, y_train.shape)

input_ids_test = np.array([input_ids[i] for i in random_order[int(len(input_ids)*0.95):]])
input_types_test = np.array([input_types[i] for i in random_order[int(len(input_ids)*0.95):]])
input_masks_test = np.array([input_masks[i] for i in random_order[int(len(input_ids)*0.95):]])
y_test = np.array([label[i] for i in random_order[int(len(input_ids) * 0.95):]])
```

---

**data2dataloader**
```python
BATCH_SIZE = 16
train_data = TensorDataset(torch.LongTensor(input_ids_train), 
                           torch.LongTensor(input_types_train), 
                           torch.LongTensor(input_masks_train), 
                           torch.LongTensor(y_train))
train_sampler = RandomSampler(train_data)  
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(torch.LongTensor(input_ids_test), 
                          torch.LongTensor(input_types_test), 
                         torch.LongTensor(input_masks_test),
                          torch.LongTensor(y_test))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
```

---

**模型**
```python
```BERT
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
path = '模型地址'
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]
        _, pooled = self.bert(context, token_type_ids=types, 
                              attention_mask=mask, 
                              output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
```END

```全连接
class Model(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28,100)
        self.fc2 = nn.Linear(100,300)
        self.fc3 = nn.Linear(300,200)
        self.fc4 = nn.Linear(200,200)
        self.fc5 = nn.Linear(200,200)
        self.fc6 = nn.Linear(200,200)
        self.fc7 = nn.Linear(200,200)
        self.fc8 = nn.Linear(200,200)
        self.fc9 = nn.Linear(200,200)
        self.fc0 = nn.Linear(200,2)
    def forward(self,x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.fc7(out)
        out = self.fc8(out)
        out = self.fc9(out)
        out = self.fc0(out)
        return out
```END

```卷积神经网络
class Model(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=5,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64,2)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0),-1)
        return out
```END
```

---

**调用模型**
```python
model = Model()
```根据情况判断是否需要在原有模型基础上训练
# pthpath = 'model.pth'
# a = torch.load(pthpath)
# model.load_state_dict(a)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)
```

---

**优化器**
```python
```BERT
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
NUM_EPOCHS = 8 # 通常在5个epoch后拟合程度最好
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=0.05,
                     t_total=len(train_loader) * NUM_EPOCHS
                    )
```END

```多分类
criterion = nn.MultiLabelSoftMarginLoss() # 使用MultiLabelSoftMarginLoss作为损失函数
optimizer = optim.SGD(net.parameters(),lr=0.0000000001,momentum=0.9) # 使用SGD优化，学习速率为0.0000000001，momentum为0.9
```END

```CV
#代价函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
```END
```

---

**训练与评估**
```python
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (x1,x2,x3, y) in enumerate(train_loader):
        start_time = time.time()
        x1,x2,x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        y_pred = model([x1, x2, x3])
        model.zero_grad()
        loss = F.cross_entropy(y_pred, y.squeeze())
        loss.backward()
        optimizer.step()
        if(batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(x1), 
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader), 
                                                                           loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0 
    acc = 0 
    for batch_idx, (x1,x2,x3, y) in enumerate(test_loader):
        x1,x2,x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            y_ = model([x1,x2,x3])
        test_loss += F.cross_entropy(y_, y.squeeze())
        pred = y_.max(-1, keepdim=True)[1]
        acc += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
          test_loss, acc, len(test_loader.dataset),
          100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)

best_acc = 0.0 
PATH = 'model.pth'
print('TRAINNING START!')
for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch)
    acc = test(model, device, test_loader)
    if best_acc < acc: 
        best_acc = acc 
        torch.save(model.state_dict(), PATH)
    print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))
print("TRAINNING END")
```

**测试集**
```python
```1
net_predict = Net()
net_predict.load_state_dict(torch.load(PATH_checkpoint))

correct = 0
total = 0
with torch.no_grad():
    for data in test_data:
        images, labels = data
        outputs = net_predict(images)
        _, predicted = torch.max(outputs, 1)
        correct += torch.sum(predicted == labels.data).to(torch.float32)    
        total += 1
    print('Accuracy : %2d %%' % (100 * correct / total))
```END

```2
def test(model, device, test_loader):
    res = []
    for batch_idx, (x1,x2,x3) in enumerate(test_loader):
        x1,x2,x3 = x1.to(device), x2.to(device), x3.to(device)
        with torch.no_grad():
            y_ = model([x1,x2,x3])
        pred = y_.max(-1, keepdim=True)[1]
        y_ = y_.cpu().numpy().tolist()[0]
        pr = y_[0]-y_[1]
        res.append(pr)
    return res

raw = test(model,DEVICE,train_loader)

```以下代码为调整模型阈值方法，根据实际情况取舍
# Threshold = -1.25
# result = []
# tag_1 = 0
# tag_0 = 0
# for i in raw:
#   if i>=Threshold:
#     result.append(0)
#     tag_0 += 1
#   if i<Threshold:
#     result.append(1)
#     tag_1 += 1

# tag_1,tag_0,tag_1/tag_0
```END
```