import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import random, numpy.random


# 设置随机种子, numpy, pytorch, python随机种子
def seed_torch(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch()

# rusume是否使用预训练模型继续训练,问号处输入模型的编号
resume = True  # 是继续训练，否重新训练
datasets = 'cifar10'  # 选择cifar10,  mnist, fashion_mnist,STL10，Anime

nc = 3  # 图片的通道数

# 类别数
n_classes = 10

# 控制生成器生成指定标签的图片
# target_label = 4

# 训练批次数
batch_size = 64

# 噪声向量的维度
nz = 100

# 判别器的深度
ndf = 64
# 生成器的深度
ngf = 64

# 真实标签
real_label = 1.0
# 假标签
fake_label = 0.0
start_epoch = 0

# 模型

# 生成器                             #(N,nz, 1,1)
netG = nn.Sequential(nn.ConvTranspose2d(nz + n_classes, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
                     nn.Tanh()  # (N,nc, 128,128)

                     )

# 判别器             #(N,nc, 128,128)
netD = nn.Sequential(nn.Conv2d(nc + n_classes, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # (N,1,1,1)
                     nn.Flatten(),  # (N,1)
                     nn.Sigmoid()
                     )


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


netD.apply(weights_init)
netG.apply(weights_init)

# 加载数据集
apply_transform1 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                             transform=apply_transform1)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义损失函数
criterion = torch.nn.BCELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# setup optimizer
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

image = torch.Tensor(train_dataset.data / 255).permute(0, 3, 1, 2)

lb = LabelBinarizer()
lb.fit(list(range(0, n_classes)))


# 将标签进行one-hot编码
def to_categrical(y: torch.FloatTensor):
    y_one_hot = lb.transform(y.cpu())
    floatTensor = torch.FloatTensor(y_one_hot)
    return floatTensor.to(device)


# 样本和one-hot标签进行连接，以此作为条件生成
def concanate_data_label(data, y):  # data （N,nc, 128,128）
    y_one_hot = to_categrical(y)  # (N,1)->(N,n_classes)

    con = torch.cat((data, y_one_hot), 1)

    return con


# 如果继续训练，就加载预训练模型
# if resume:
#     print('==> Resuming from checkpoint..')
#     checkpoint = torch.load('./checkpoint/GAN_%s_best.pth' % datasets)
#     netG.load_state_dict(checkpoint['net_G'])
#     netD.load_state_dict(checkpoint['net_D'])
#     start_epoch = checkpoint['start_epoch']
# print('netG:', '\n', netG)
# print('netD:', '\n', netD)
#
# print('training on:   ', device, '   start_epoch', start_epoch)

netD, netG = netD.to(device), netG.to(device)
# 固定生成器，训练判别器
images = []
soft_labels = []
for epoch in range(start_epoch, 200):
    for batch, (data, target) in enumerate(train_loader):
        #         if epoch%2==0 and batch==0:
        #             torchvision.utils.save_image(data[:16], filename='./generated_fake/%s/源epoch_%d_grid.png'%(datasets,epoch),nrow=4,normalize=True)
        data = data.to(device)
        target = target.to(device)
        # 拼接真实数据和标签
        target1 = to_categrical(target).unsqueeze(2).unsqueeze(3).float()  # 加到噪声上
        target2 = target1.repeat(1, 1, data.size(2), data.size(3))  # 加到数据上
        data = torch.cat((data, target2),
                         dim=1)  # 将标签与数据拼接 (N,nc,128,128),(N,n_classes, 128,128)->(N,nc+nc_classes,128,128)

        label = torch.full((data.size(0), 1), real_label).to(device)

        # （1）训练判别器
        # training real data
        netD.zero_grad()
        output = netD(data)
        loss_D1 = criterion(output, label)
        loss_D1.backward()

        # training fake data,拼接噪声和标签
        noise_z = torch.randn(data.size(0), nz, 1, 1).to(device)
        noise_z = torch.cat((noise_z, target1), dim=1)  # (N,nz+n_classes,1,1)
        # 拼接假数据和标签
        fake_data = netG(noise_z)
        fake_data = torch.cat((fake_data, target2), dim=1)  # (N,nc+n_classes,128,128)
        label = torch.full((data.size(0), 1), fake_label).to(device)

        output = netD(fake_data.detach())
        loss_D2 = criterion(output, label)
        loss_D2.backward()

        # 更新判别器
        optimizerD.step()

        # （2）训练生成器
        netG.zero_grad()
        label = torch.full((data.size(0), 1), real_label).to(device)
        output = netD(fake_data.to(device))
        lossG = criterion(output, label)
        lossG.backward()

        # 更新生成器
        optimizerG.step()

        if batch % 10 == 0:
            print('epoch: %4d, batch: %4d, discriminator loss: %.4f, generator loss: %.4f'
                  % (epoch, batch, loss_D1.item() + loss_D2.item(), lossG.item()))

        # 每2个epoch保存图片

        if epoch > 185 and epoch % 1 == 0 and batch == 0:
            # 生成指定target_label的图片
            noise_z1 = torch.randn(data.size(0), nz, 1, 1).to(device)

            for target_label in range(10):
                target3 = to_categrical(torch.full((data.size(0), 1), target_label)).unsqueeze(2).unsqueeze(
                    3).float()  # 加到噪声上
                noise_z = torch.cat((noise_z1, target3), dim=1)  # (N,nz+n_classes,1,1)

                fake_data = netG(noise_z.to(device))

                # plt.imsave('./generated_fake/epoch_%d.png' % (datasets, epoch), data[0])
                fake_data = fake_data * 0.5 + 0.5
                t = transforms.Compose([transforms.Resize((32, 32))])
                fake_data = t(fake_data)
                for image in fake_data:
                    image = image.data.cpu().numpy()
                    images.append(image)
                    soft_label = np.zeros(10)
                    soft_label[target_label] += random.uniform(0, 10)  # an unnormalized soft label vector
                    soft_labels.append(soft_label)
images = np.array(images)
soft_labels = np.array(soft_labels)
print(images.shape, images.dtype, soft_labels.shape, soft_labels.dtype)
np.save('data_gan.npy', images)
np.save('label_gan.npy', soft_labels)
