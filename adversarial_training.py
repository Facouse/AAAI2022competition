import numpy as np
from absl import app, flags
import torch
import torchvision
import random
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

FLAGS = flags.FLAGS

models = torchvision.models.resnet50(pretrained=True)
transform_data = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_data)
dataset_load = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

images = []
soft_labels = []
for xs, ys in dataset_load:
    xs = fast_gradient_method(models, xs, 0.3, np.inf).cuda()

    for image in xs:
        image = image.data.cpu().numpy()
        images.append(image)
        soft_label = np.zeros(10)
        soft_label[ys] += random.uniform(0, 10)  # an unnormalized soft label vector
        soft_labels.append(soft_label)

images = np.array(images)
soft_labels = np.array(soft_labels)
print(images.shape, images.dtype, soft_labels.shape, soft_labels.dtype)
np.save('data_at.npy', images)
np.save('label_at.npy', soft_labels)
