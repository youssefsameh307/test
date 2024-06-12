from augmentation.randaugment import RandAugment
from arguments import parse_args
from train import get_transform_dict, train
from datasets.datasets import get_datasets
from datasets.loaders import create_loaders
from utils.misc import load_dataset_indices, save_dataset_indices, get_save_path, initialize_logger, seed, save_args
import logging
from utils.train import model_init, set_grads
from models.model_factory import MODEL_GETTERS
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import os

import random

logger = logging.getLogger()
from abc import ABC, abstractmethod

class RetrainFramework(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def getAugmentedData(self,data):
        pass

    @abstractmethod
    def trainModel(self):
        pass

    @abstractmethod
    def evaluateModel(self):
        pass


class FixMatchFrameWork(RetrainFramework):

    def __init__(self,model=None):
        self.args = parse_args()
        self.args.device = "cpu" # TODO remove this 
        self.args.epochs = 1 # TODO remove this
        self.save_path = get_save_path(self.args)
        self.writer = SummaryWriter(self.save_path)
        
        # Step 1: Create random image tensors
        self.num_images = 5
        self.image_size = (480, 480)
        self.directory = '/mnt/data/random_images'
        self.model = model

    def getAugmentedData(self,data=None,augmentations=None):
        strong_augmentation = RandAugment(n=4, randomized_magnitude=True, augmentations= augmentations)
        
        transform_dict = get_transform_dict(self.args, strong_augmentation)
        
        initial_indices = None

        self.train_sets, self.validation_set, self.test_set = get_datasets(
            self.args.data_dir,
            self.args.dataset,
            self.args.num_labeled,
            self.args.num_validation,
            transform_dict["train"],
            transform_dict["train_unlabeled"],
            transform_dict["test"],
            dataset_indices=initial_indices
        )

        save_dataset_indices(self.save_path, self.train_sets, self.validation_set)

        (self.train_loader_labeled, self.train_loader_unlabeled), self.validation_loader, self.test_loader = create_loaders(
            self.args,
            self.train_sets["labeled"],
            self.train_sets["unlabeled"],
            self.validation_set,
            self.test_set,
            self.args.batch_size,
            mu=self.args.mu,
            total_iters=self.args.iters_per_epoch,
            num_workers=self.args.num_workers,
        )

        

        return data
    
    def trainModel(self):
        # Print and log dataset stats
        logger.info("-------- Starting FixMatch Model Training --------")
        logger.info("\t- Labeled train set: {}".format(len(self.train_sets["labeled"])))
        logger.info("\t- Unlabeled train set: {}".format(len(self.train_sets["unlabeled"])))
        logger.info("\t- Validation set: {}".format(len(self.validation_set)))
        logger.info("\t- Test set: {}".format(len(self.test_set)))

        # Load specified image classification network
        logger.info("-------- MODEL --------")
        self.args.num_classes = len(self.test_set.classes)
        model = MODEL_GETTERS[self.args.model](num_classes=self.args.num_classes, pretrained=self.args.pretrained)
        if not self.args.pretrained:
            model.apply(model_init)
        else:
            if len(self.args.trainable_layers) > 0:
                set_grads(model, self.args.trainable_layers)

        num_params = sum([p.numel() for p in model.parameters()])
        logger.info("\t- Number of parameters: {}".format(num_params))
        logger.info("\t- Number of target classes: {}".format(self.args.num_classes))

        # Start FixMatch training
        train(
            self.args,
            model,
            self.train_loader_labeled,
            self.train_loader_unlabeled,
            self.validation_loader,
            self.test_loader,
            self.writer,
            save_path=self.save_path
        )

        save_args(self.args, self.save_path)
        

    def evaluateModel(self, model, data):
        return model.evaluate(data)

num_labels = 5
num_images = 40
image_size = (480, 480)
directory = './dummyData'

def create_dummy_data():

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate and save images
    labels = []
    for i in range(num_images):
        random_image = np.random.randint(0, 256, (image_size[0], image_size[1], 3), dtype=np.uint8)
        image = Image.fromarray(random_image)
        image.save(os.path.join(directory, f'random_image_{i}.png'))
        label= random.randint(0, num_labels)
        labels.append(label)

    labels = np.array(labels)
    np.save(os.path.join(directory, 'labels.npy'), labels)

    print(directory)

def load_dummy_data():

    loaded_images = []
    for i in range(num_images):
        image_path = os.path.join(directory, f'random_image_{i}.png')
        image = Image.open(image_path)
        image_tensor = np.array(image)
        loaded_images.append(image_tensor)

    loaded_images[0].shape

def main():
    fix_match_framework = FixMatchFrameWork()
    fix_match_framework.getAugmentedData()
    fix_match_framework.trainModel()


if __name__ == '__main__':
    main()