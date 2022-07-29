import random

import torch
import torchvision
import os
#from pytorch_toolbelt import losses as L
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import segmentation_models_pytorch as smp
# import torchvision.models as models
# model=models.segmentation.fcn_resnet50(pretrained=False,num_classes=1)
model = smp.FPN(
    encoder_name="resnet34",  # 选择解码器, 例如 mobilenet_v2 或 efficientnet-b7
    encoder_weights=None,  # 使用预先训练的权重imagenet进行解码器初始化
    in_channels=3,  # 模型输入通道（1个用于灰度图像，3个用于RGB等）
    classes=1,  # 模型输出通道（数据集所分的类别总数）
    activation='sigmoid',
)

model.load_state_dict(torch.load('/ai/store/Second/init_weight/FPN.pkl'))
model.to(device)

model1 = smp.DeepLabV3Plus(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
        activation='sigmoid',  # 二分类需要换成sigmoid
    )
model1.load_state_dict(torch.load('/ai/store/Second/init_weight/deeplabv3.pkl'))
model1.to(device)

train_dir='/ai/store/Second/train/train_image'
train_dir_seg='/ai/store/Second/train/train_mask'
train_un='/ai/store/Second/train/unlabel_train_image'
val_dir='/ai/store/Second/testing/test_image'
val_dir_seg='/ai/store/Second/testing/test_mask'
class MyDataset(Dataset):
    def __init__(self, path,path1,path2,transform1=None,transform2=None,transform3=None,transform4=None,transform5=None):
        """
        tex_path : txt文本路径，该文本包含了图像的路径信息，以及标签信息
        transform：数据处理，对图像进行随机剪裁，以及转换成tensor
        """
        fh=os.listdir(path)  #读取文件/train_img
        un_fh=os.listdir(path2)
        random.shuffle(un_fh)
        #fh.sort()
        imgs = []  #用来存储路径 名字
        for line in range(len(un_fh)): #所有图像名字的列表
            t=os.path.join(path,fh[line//4]) #/val/none
            l = fh[line//4].split('.')[0] + '.png'
            t_seg = os.path.join(path1, l)

            un_t=os.path.join(path2,un_fh[line])
            imgs.append((t, t_seg,un_t))  # 路径和标签添加到列表中
        self.imgs = imgs
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.transform4 = transform4
        self.transform5=transform5

    def __getitem__(self, index):
        fn,fn_seg,un_fn = self.imgs[index]   #通过index索引返回一个图像路径fn 与 标签label
        img = Image.open(fn).convert('RGB')
        un_img = Image.open(un_fn).convert('RGB')
        img_seg=Image.open(fn_seg)
        a=random.randint(1,4)#选择三种数据增强
        if a==1:
            img=self.transform1(img)
            img_seg=self.transform1(img_seg)

        if a==2:
            img=self.transform2(img)
            img_seg=self.transform2(img_seg)

        if a==3:
            img=self.transform3(img)
            img_seg=self.transform3(img_seg)

        a = random.randint(1, 4)  # 选择三种数据增强
        if a == 1:
            un_img = self.transform1(un_img)
        if a == 2:
            un_img = self.transform2(un_img)
        if a == 3:
            un_img = self.transform3(un_img)
        img=self.transform4(img)#转张量、归一化
        un_img=self.transform4(un_img)
        img_seg=self.transform5(img_seg)#转张量
        return img,img_seg ,un_img             #这就返回一个样本
    def __len__(self):
        return len(self.imgs)

class MyDataset_val(Dataset):
    def __init__(self, path,path1,transform4=None,transform5=None):
        fh=os.listdir(path)  #读取文件/train_img
        imgs = []  #用来存储路径 名字
        for line in fh: #所有图像名字的列表
            t=os.path.join(path,line) #/val/none
            l=line.split('.')[0]+'.png'
            t_seg=os.path.join(path1,l)
            imgs.append((t, t_seg))  # 路径和标签添加到列表中
        self.imgs = imgs
        self.transform4=transform4
        self.transform5 = transform5
    def __getitem__(self, index):
        fn,fn_seg = self.imgs[index]   #通过index索引返回一个图像路径fn 与 标签label
        img = Image.open(fn).convert('RGB')
        img_seg=Image.open(fn_seg)

        img=self.transform4(img)#转张量、归一化
        img_seg=self.transform5(img_seg)#转张量
        return img,img_seg              #这就返回一个样本
    def __len__(self):
        return len(self.imgs)
t1=transforms.ToTensor()

train_folder_set = MyDataset(train_dir,train_dir_seg, train_un,transform1=transforms.RandomVerticalFlip(p=1),transform2=transforms.RandomHorizontalFlip(p=1),

    transform3=transforms.RandomApply([transforms.RandomChoice([transforms.ColorJitter(contrast=0.5),
                                                     transforms.ColorJitter(brightness=0.5),
                                                     transforms.ColorJitter(saturation=0.5)
                                                     ])],p=1),
    transform4=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),transform5=t1)

val_folder_set = MyDataset_val(val_dir,val_dir_seg ,transform4=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]),transform5=t1)



from torch import  nn
toPIL = transforms.ToPILImage()
#拼接
train_loader = DataLoader(dataset=train_folder_set, batch_size=8, shuffle=True,num_workers=8 ,drop_last=False,pin_memory=False,prefetch_factor=2)
#un_train_loader = DataLoader(dataset=train_folder_set, batch_size=4, shuffle=True,num_workers=8 ,drop_last=False,pin_memory=False,prefetch_factor=2)
test_loader= DataLoader(dataset=val_folder_set, batch_size=16, shuffle=False,drop_last=False,pin_memory=False,num_workers=16)
optimizer = torch.optim.SGD([
	{'params': model.parameters(), 'lr': 0.001},
	{'params': model1.parameters(), 'lr': 0.001}
	],lr=0.001, weight_decay=0.0001, momentum=0.9)
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
DiceLoss_fn = DiceLoss(mode='binary')  # 多分类改为multiclass
Bceloss_fn = nn.BCELoss()
        # 软交叉熵,即使用了标签平滑的交叉熵,会增加泛化性
        # SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1) #用于多分类
loss_fn  =DiceLoss_fn
import torch.nn.functional as F
from math import exp
def dice_loss(logits, targets):
    bs = targets.size(0)
    smooth = 1

    #probs = F.sigmoid(logits)
    m1 = logits.view(bs, -1)
    m2 = targets.view(bs, -1)
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    score = 1 - score.sum() / bs
    return score
import numpy as np

def binary_cross_entropy(prediction, label):
    loss = -(label * np.log(prediction) + (1 - label) * np.log(1-prediction))

    return loss


# PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)


        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # target[target >= 0.5] = 1
        # target[target < 0.5] = 0


        # intersection = (inputs * targets).sum()
        # dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        # Dice_BCE = BCE + dice_loss

        return BCE
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        # BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        # Dice_BCE = BCE + dice_loss

        return dice_loss
best_dice=0
print("训练")
def t(epoch):
    a=-5*(1-(epoch)/200)**2
    return 0.1*(exp(a))
for epoch in range(200):
    model.train()
    model1.train()
    weight=t(epoch)
    for batch_id,data in tqdm(enumerate(train_loader)):
        #print(batch_id)

        inputs,target=data[0],data[1]
        inputs1 = data[2]
        # print(inputs.size())
        # print(target.size())
        # print(inputs1.size())
        inputs, target = inputs.to(device), target.to(device)
        inputs1=inputs1.to(device)

        optimizer.zero_grad()
        outputs=model(inputs)
        outputs1=model1(inputs)
        target[target >0] = 1
        target[target < 1] = 0

        un_outputs = model(inputs1)
        un_outputs_no = un_outputs.detach()
        un_outputs1 = model1(inputs1)
        un_outputs_no1 = un_outputs1.detach()
        # loss=BCELoss()(outputs, target)+BCELoss()(outputs1, target)
        # loss+=(DiceLoss()(un_outputs,un_outputs1))
        loss=BCELoss()(outputs, target)+BCELoss()(outputs1, target)+\
             DiceLoss()(un_outputs1,un_outputs_no)+DiceLoss()(un_outputs,un_outputs_no1)
        # loss = CrossEntropyLoss(outputs, target)+CrossEntropyLoss(outputs1, target)#+weight*(dice_loss(un_outputs,un_outputs1)+dice_loss(un_outputs1,un_outputs))
        loss.backward()
        optimizer.step()

        # print(outputs.size())  # [16, 1, 480, 640]
        # print(outputs1.size())
        # print(target.size())#[16, 1, 480, 640]


    model.eval()#close dropout and batchnorm
    model1.eval()
    print("测试")
    dice=0
    with torch.no_grad():  #close grad
        for b,(images,labels) in  (enumerate(test_loader)):
            images,labels=images.to(device),labels.to(device)
            predict=model(images)
            predict1=model1(images)
            predict=predict*0.5+predict1*0.5
            #predict[predict >= 0.5] = 1
            # predict[predict < 0.5] = 0

            labels[labels > 0] = 1
            labels[labels < 1] = 0
            dice =1- DiceLoss()(predict, labels)


    print('epoch:%d    Dice系数 in test set:%.4f '%(epoch,(dice)*100))

    if  dice>best_dice:
        best_dice=dice
        p='/ai/store/Second/result/FPN_epoch_'+str(epoch)+'_dice_'+str(int(dice*10000))+'.pkl'
        torch.save(model, p)
        p1 = '/ai/store/Second/result/deeplabv3+_epoch_' + str(epoch) + '_dice_' + str(int(dice * 10000)) + '.pkl'
        torch.save(model1, p1)

