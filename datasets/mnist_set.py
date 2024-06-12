import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from torch.utils.data import Subset
import numpy as np

class GrayscaleToRGB:
    def __call__(self, img):
        im = img.convert("RGB")
        return im

class LabeledUnlabeledMNIST(datasets.MNIST):
    def __init__(self, root, labeled_indices, unlabeled_indices, train=True, transform=None, target_transform=None, download=False,np=False):
        super(LabeledUnlabeledMNIST, self).__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
        self.np = np
    def __len__(self):
        return len(self.labeled_indices) + len(self.unlabeled_indices)

    def __getitem__(self, idx):
        image, label = super(LabeledUnlabeledMNIST, self).__getitem__(idx)
        image = image.convert("RGB")
        if self.np:
            image = np.array(image).astype(np.float32)
        return image, label
    
    def adjust_dataset_size(self, n):
        current_len = len(self)
        if n < current_len:
            self.labeled_indices = self.labeled_indices[:n]
            self.unlabeled_indices = self.unlabeled_indices[:n - len(self.labeled_indices)]
        elif n > current_len:
            raise ValueError("Desired dataset size is larger than the current dataset size.")

    
        # if idx < len(self.labeled_indices):
        #     real_idx = self.labeled_indices[idx]
        #     image, label = super(LabeledUnlabeledMNIST, self).__getitem__(real_idx)
        #     image = image.convert("RGB")
        #     return image, label  # Labeled data
        # else:
        #     real_idx = self.unlabeled_indices[idx - len(self.labeled_indices)]
        #     image, _ = super(LabeledUnlabeledMNIST, self).__getitem__(real_idx)
        #     image = image.convert("RGB")
        #     return image, -1  # Unlabeled data (-1 label indicates unlabeled)

    def get_random_labeled_indices(self, num_indices):
        return random.sample(self.labeled_indices, num_indices)

def calc_mean_std(train_loader):
    mean = 0.
    std = 0.
    total_images = 0

    # Iterate through the data loader
    for images, _ in train_loader:
        # Flatten the images into a 2D tensor with shape (batch_size, channels, height, width)
        batch_samples = 1  # batch size (the last batch can have smaller size)
        images = images.view(3, -1)
        mean += images.mean(1)
        std += images.std(1)
        total_images += batch_samples

    # Compute the mean and standard deviation
    mean /= total_images
    std /= total_images


class SequentialTransforms(object):
    def __init__(self, transforms1, transforms2):
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, img):
        img = self.transforms1(img)
        img = self.transforms2(img)
        return img

def get_unlabled_mnist(root, train, download, transform=None,labeled_percentage=0.7):
    # Define a transform to normalize the data
    regtransform = transforms.Compose([
        GrayscaleToRGB(),
        transforms.ToTensor(),
        #  transforms.Normalize((0.5,), (0.5,)),
        ])

    # # Define a transform to convert the data to tensors and to RGB
    # transform = transforms.Compose([
    #     GrayscaleToRGB(), 
    #     transforms.ToTensor()
    # ])

    # Download the MNIST dataset
    class SingleChannelToRGB(object):
        def __call__(self, img):
            return img.expand(3, -1, -1) 
    trainset = datasets.MNIST(root=root, train=True, download=True, transform=regtransform)
    # trainset = [(SingleChannelToRGB()(img), label) for img, label in trainset]
    test_transform= SequentialTransforms(regtransform, transform)
    testset = datasets.MNIST(root=root, train=False, download=True, transform=test_transform)
    # testset = [(SingleChannelToRGB()(img), label) for img, label in testset]
    # calc_mean_std(trainset)

    # subset_indices = range(1000)
    # subset_indicest = range(1000)
    # trainset = Subset(trainset, subset_indices)
    # testset = Subset(testset, subset_indicest)
    #TODO optimize the bottom part
    # Split the training data into labeled and unlabeled subsets
    # num_labeled = 1000  # Number of labeled samples
    all_indices = list(range(len(trainset)))
    num_labeled = int(labeled_percentage * len(trainset))
    labeled_indices = all_indices[:num_labeled]
    unlabeled_indices = all_indices[num_labeled:len(trainset)]

    # Create the combined dataset
    combined_dataset = LabeledUnlabeledMNIST(root,labeled_indices, unlabeled_indices,train=True, transform=None, download=True)

    # num_labeled = 1000  # Number of labeled samples
    all_indices = list(range(len(testset)))
    num_labeled = int(labeled_percentage * len(trainset))
    labeled_indices = all_indices[:num_labeled]
    unlabeled_indices = all_indices[num_labeled:len(testset)]
    combined_test_dataset = LabeledUnlabeledMNIST(root,labeled_indices, unlabeled_indices,train=False, transform=None, download=True,np =True)

    # combined_dataset.adjust_dataset_size(1000)
    # combined_test_dataset.adjust_dataset_size(1000)
    if train:
        return combined_dataset
    else:
        return combined_test_dataset

def main():

    x0 = get_unlabled_mnist('../dummyData', True, True, test_transform=None,labeled_percentage=0.7)
    x1 = get_unlabled_mnist('../dummyData', False, True, test_transform=None,labeled_percentage=0.7)
    print("hi")
    # Define batch size
    # batch_size = 64

    # # Create the data loaders
    # train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Example of iterating through the data loader
    # for images, labels, is_labeled in train_loader:
    #     print(f'Batch images shape: {images.shape}')
    #     print(f'Batch labels: {labels}')
    #     print(f'Is labeled: {is_labeled}')
    #     break


if __name__ == '__main__':
    main()