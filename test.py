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

model.load_state_dict(torch.load('/ai/store/Second/result/FPN_epoch_34_dice_7534.pkl').state_dict())
model.to(device)

model1 = smp.DeepLabV3Plus(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
        activation='sigmoid',  # 二分类需要换成sigmoid
    )
model1.load_state_dict(torch.load('/ai/store/Second/result/deeplabv3+_epoch_34_dice_7534.pkl').state_dict())
model1.to(device)


val_dir='/ai/store/Second/official_val'
class MyDataset_val(Dataset):
    def __init__(self, path,transform4=None,transform5=None):
        fh=os.listdir(path)  #读取文件/train_img
        imgs = []  #用来存储路径 名字
        for line in fh: #所有图像名字的列表
            t=os.path.join(path,line) #/val/none
            # l=line.split('.')[0]+'.png'
            # t_seg=os.path.join(path1,l)
            imgs.append((t, line))  # 路径和标签添加到列表中
        self.imgs = imgs
        self.transform4=transform4
        self.transform5=transform5

    def __getitem__(self, index):
        fn,name = self.imgs[index]   #通过index索引返回一个图像路径fn 与 标签label
        img = Image.open(fn).convert('RGB')
        # img_seg=Image.open(fn_seg)
        w=img.size[0]
        h=img.size[1]
        if w<h:

            img=img.rotate(90,expand=1)
        img=self.transform4(img)#转张量、归一化
        # img_seg=self.transform5(img_seg)#转张量
        return img,name,h            #这就返回一个样本
    def __len__(self):
        return len(self.imgs)
t1=transforms.ToTensor()
val_folder_set = MyDataset_val(val_dir,transform4=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))



from torch import  nn
toPIL = transforms.ToPILImage()
#拼接
test_loader= DataLoader(dataset=val_folder_set, batch_size=16, shuffle=False,drop_last=False,pin_memory=False,num_workers=16)
optimizer = torch.optim.SGD([
	{'params': model.parameters(), 'lr': 0.001},
	{'params': model1.parameters(), 'lr': 0.001}
	],lr=0.001, weight_decay=0.0001, momentum=0.9)

import torch.nn.functional as F
from math import exp

import numpy as np






model.eval()#close dropout and batchnorm
model1.eval()
print("测试")
toPIL = transforms.ToPILImage()
p = '/ai/store/Second/result_val_image'
with torch.no_grad():  # close grad
    for b, (images, name,s) in (enumerate(test_loader)):

        images = images.to(device)
        predict = model(images)
        predict1 = model1(images)
        predict = predict * 0.5 + predict1 * 0.5
        predict[predict >= 0.5] = 1
        predict[predict < 0.5] = 0

        # predict = predict.squeeze(1).type(torch.LongTensor)
        #
        batch_size = predict.size()[0]
        for i in range(batch_size):
            image = predict[i]
            n=name[i].split('.')[0]+'.png'
            obj_path = os.path.join(p, n)
            # print(obj_path)
            # print(image.size())
            image = image.cpu().clone()
            # image=np.array(image)

            image= image.transpose(0,2).transpose(0,1)
            image =image.squeeze(2)
            img = toPIL(image)
            # # img = Image.fromarray(image)
            # img=unloader(image)
            if (s[i] ==640):
                img=img.rotate(270,expand=1)
            img.save(obj_path)




