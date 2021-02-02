import os
from PIL import Image
from torch.utils.data import Dataset
from torch import nn as nn
import torchvision.models as tm

import xutils.data.text_utils as tu


class ImageTextDataset(Dataset):
    def __init__(self, root_dir, labels, positive_files, negative_files, transform=None):
        self.root_dir = root_dir
        self.labels = labels
        self.files_path = [negative_files, positive_files]
        self.image_list = []

        for label_index in range(len(self.labels)):
            class_files = [[os.path.join(self.root_dir, self.labels[label_index], x), label_index] \
                           for x in tu.read_lines(self.files_path[label_index])]
            self.image_list += class_files

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx][0]

        # Read the image
        image = Image.open(path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = int(self.image_list[idx][1])

        return {
            'img': image,
            'label': label,
            'paths': path
        }


def create_transfer_model(num_outputs=2, model="vgg19_bn"):
    model = getattr(tm, model)(pretrained=True)
    model.classifier[6] = nn.Linear(4096, num_outputs)

    return model
