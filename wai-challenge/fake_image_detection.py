# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T02:00:49.476117Z","iopub.execute_input":"2025-03-02T02:00:49.476498Z","iopub.status.idle":"2025-03-02T02:01:00.080504Z","shell.execute_reply.started":"2025-03-02T02:00:49.476474Z","shell.execute_reply":"2025-03-02T02:01:00.079611Z"}}
!pip install efficientnet_pytorch
!pip install torcheval

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T02:01:02.421883Z","iopub.execute_input":"2025-03-02T02:01:02.422204Z","iopub.status.idle":"2025-03-02T02:01:14.111241Z","shell.execute_reply.started":"2025-03-02T02:01:02.422159Z","shell.execute_reply":"2025-03-02T02:01:14.110530Z"},"id":"7y6Lt3lbC1EZ"}
import argparse
import json
import math
import os
import time
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from efficientnet_pytorch import EfficientNet
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.nn import init
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torcheval.metrics.functional import multiclass_f1_score
from torchsummary import summary
from torchvision import transforms
from torchvision.transforms import v2
from tqdm import tqdm

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T05:27:39.083787Z","iopub.execute_input":"2025-03-02T05:27:39.084067Z","iopub.status.idle":"2025-03-02T05:27:39.176291Z","shell.execute_reply.started":"2025-03-02T05:27:39.084045Z","shell.execute_reply":"2025-03-02T05:27:39.175503Z"}}
xceptionnet_pretrained_weights = "/kaggle/input/wai-challenge-data/xceptionnet_FF_weights.pth"
train_csv = "/kaggle/input/ai-vs-human-generated-dataset/train.csv"
data = "/kaggle/input/ai-vs-human-generated-dataset"
train_folder = "/kaggle/input/ai-vs-human-generated-dataset/train_data"
test_folder = "/kaggle/input/ai-vs-human-generated-dataset/test_data_v2"
test_csv = "/kaggle/input/ai-vs-human-generated-dataset/test.csv"
efficientnetB4_pretrained_weights = "/kaggle/input/wai-challenge-data/EfficientNetB4_FFPP_bestval-93aaad84946829e793d1a67ed7e0309b535e2f2395acb4f8d16b92c0616ba8d7.pth"
test_image_sizes_df = pd.read_csv("/kaggle/input/wai-challenge-data/test_image_sizes")
train_image_sizes_df = pd.read_csv("/kaggle/input/wai-challenge-data/train_image_sizes")
xceptionnet_best_params = "/kaggle/input/wai-challenge-data/xceptionnet_best_weights.pth"
efficinetnetB4_best_params = "/kaggle/input/wai-challenge-data/efficientnetB4_best_weights.pth"

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T02:01:16.030409Z","iopub.execute_input":"2025-03-02T02:01:16.030759Z","iopub.status.idle":"2025-03-02T02:01:16.053283Z","shell.execute_reply.started":"2025-03-02T02:01:16.030730Z","shell.execute_reply":"2025-03-02T02:01:16.052404Z"},"id":"MiJ9u-kGC1EW"}
"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""

pretrained_settings = {
    "xception": {
        "imagenet": {
            "url": xceptionnet_pretrained_weights,
            "input_space": "RGB",
            "input_size": [3, 299, 299],
            "input_range": [0, 1],
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "num_classes": 1000,
            "scale": 0.8975,  # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        reps,
        strides=1,
        start_with_relu=True,
        grow_first=True,
    ):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(
                in_filters, out_filters, 1, stride=strides, bias=False
            )
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(
                    in_filters, out_filters, 3, stride=1, padding=1, bias=False
                )
            )
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False)
            )
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(
                    in_filters, out_filters, 3, stride=1, padding=1, bias=False
                )
            )
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.dropout = nn.Dropout(p=0.2)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def xception(num_classes=1000, pretrained="imagenet"):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings["xception"][pretrained]
        assert (
            num_classes == settings["num_classes"]
        ), "num_classes should be {}, but is {}".format(
            settings["num_classes"], num_classes
        )

        model = Xception(num_classes=num_classes)
        model.load_state_dict(torch.load(settings["url"], weights_only=True))

        model.input_space = settings["input_space"]
        model.input_size = settings["input_size"]
        model.input_range = settings["input_range"]
        model.mean = settings["mean"]
        model.std = settings["std"]

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T02:01:20.677678Z","iopub.execute_input":"2025-03-02T02:01:20.678018Z","iopub.status.idle":"2025-03-02T02:01:20.685456Z","shell.execute_reply.started":"2025-03-02T02:01:20.677992Z","shell.execute_reply":"2025-03-02T02:01:20.684636Z"}}
"""
Video Face Manipulation Detection Through Ensemble of CNNs

Image and Sound Processing Lab - Politecnico di Milano

Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini
"""

"""
Feature Extractor
"""


class FeatureExtractor(nn.Module):
    """
    Abstract class to be extended when supporting features extraction.
    It also provides standard normalized and parameters
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


"""
EfficientNet
"""


class EfficientNetGen(FeatureExtractor):
    def __init__(self, model: str):
        super(EfficientNetGen, self).__init__()

        self.efficientnet = EfficientNet.from_pretrained(model)
        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, 1)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        return x


class EfficientNetB4(EfficientNetGen):
    def __init__(self):
        super(EfficientNetB4, self).__init__(model='efficientnet-b4')

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T05:28:18.624823Z","iopub.execute_input":"2025-03-02T05:28:18.625177Z","iopub.status.idle":"2025-03-02T05:28:18.630994Z","shell.execute_reply.started":"2025-03-02T05:28:18.625144Z","shell.execute_reply":"2025-03-02T05:28:18.629947Z"},"id":"LFkv8_7NC1EZ"}
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
seed = 42
epochs = 5

configs = {
    "xceptionnet": {
        "network": lambda: XceptionNetwork(),
        "best_params": xceptionnet_best_params,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "img_size": 299,
        "logs": {
            "train_losses": [],
            "valid_losses": [],
            "train_f1_scores": [],
            "valid_f1_scores": []
        },
    },
    "efficientnetB4": {
        "network": lambda: EfficientNetB4Network(),
        "best_params": efficinetnetB4_best_params,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "img_size": 224,
        "logs": {
            "train_losses": [],
            "valid_losses": [],
            "train_f1_scores": [],
            "valid_f1_scores": []
        }
    },
}

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T02:01:27.650871Z","iopub.execute_input":"2025-03-02T02:01:27.651162Z","iopub.status.idle":"2025-03-02T02:01:27.656421Z","shell.execute_reply.started":"2025-03-02T02:01:27.651140Z","shell.execute_reply":"2025-03-02T02:01:27.655414Z"},"_kg_hide-input":true,"id":"K6Rx-bDdC1Eb","outputId":"e46c182e-be0f-4343-b1ff-010deaebefbf","jupyter":{"outputs_hidden":false}}
from copy import deepcopy


def save_config(configs, model_name):
    """Saves model configuration and training logs to a JSON file with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{model_name}_config_{timestamp}.json"

    # Remove lambda functions (JSON doesn't support them)
    config_to_save = deepcopy(configs)
    config_to_save.pop("network", None)

    # Save configuration to a JSON file
    with open(filename, "w") as f:
        json.dump(config_to_save, f, indent=4)

    print(f"Configuration saved to {filename}")

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T02:01:31.578212Z","iopub.execute_input":"2025-03-02T02:01:31.578571Z","iopub.status.idle":"2025-03-02T02:01:31.588996Z","shell.execute_reply.started":"2025-03-02T02:01:31.578540Z","shell.execute_reply":"2025-03-02T02:01:31.587990Z"}}
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def _fit(self):
        print(f"INITIALIZING TRAINING ON CUDA GPU")
        start_time = datetime.now()
        print(f"Start Time: {start_time}")

        valid_loss_min = np.inf
        
        for epoch in range(1,  configs[model_name]["epochs"]+1):
            print(f"{'='*50}")
            print(f"EPOCH {epoch} - TRAINING...")
        
            epoch_loss = 0.0
            epoch_f1_score = 0.0
        
            model.train()
            for data, target in tqdm(train_dataloader):
                data = data.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.long)
        
                optimizer.zero_grad()
        
                output = model(data)
                loss = criterion(output, target)
        
                loss.backward()
        
                #f_score = f1_score(target.cpu(), output.cpu().argmax(dim=1))
                f_score = multiclass_f1_score(output, target, num_classes=2)
                epoch_loss += loss.item()
                epoch_f1_score += f_score.item()
        
                optimizer.step()
        
            train_loss, train_f1_score = epoch_loss / len(train_dataloader), epoch_f1_score / len(train_dataloader)
            print(f"\n\t[TRAIN] EPOCH {epoch} - LOSS: {train_loss}, F1-SCORE: {train_f1_score}\n")
            configs[model_name]["logs"]["train_losses"].append(train_loss)
            configs[model_name]["logs"]["train_f1_scores"].append(train_f1_score)
        
            print(f"EPOCH {epoch} - VALIDATING...")
        
            valid_loss = 0.0
            valid_f1_score = 0.0
        
            model.eval()
        
            for data, target in val_dataloader:
                data = data.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.long)
        
                with torch.no_grad():
                    output = model(data)
                    loss = criterion(output, target)
                    #f_score = f1_score(target.cpu(), output.cpu().argmax(dim=1))
                    f_score = multiclass_f1_score(output, target, num_classes=2)
                    valid_loss += loss.item()
                    valid_f1_score += f_score.item()
        
            val_loss, val_f1_socre = valid_loss / len(val_dataloader), valid_f1_score / len(val_dataloader)
            print(f"\t[VALID] LOSS: {val_loss}, F1-SCORE: {val_f1_socre}\n")
            configs[model_name]["logs"]["valid_losses"].append(val_loss)
            configs[model_name]["logs"]["valid_f1_scores"].append(val_f1_socre)
        
            if val_loss <= valid_loss_min and epoch != 1:
                print("Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...".format(valid_loss_min, val_loss))
                torch.save(model.state_dict(),f"{model_name}_best_weights.pth")
                valid_loss_min = val_loss
    
        save_config(configs[model_name], model_name)
        print(f"Execution time: {datetime.now() - start_time}")
        
        torch.save(model.state_dict,f'{model_name}_weights_{datetime.now().strftime("%Y%m%d-%H%M")}.pth')

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T02:01:34.116365Z","iopub.execute_input":"2025-03-02T02:01:34.116650Z","iopub.status.idle":"2025-03-02T02:01:34.123060Z","shell.execute_reply.started":"2025-03-02T02:01:34.116628Z","shell.execute_reply":"2025-03-02T02:01:34.122202Z"}}
# Define EfficientNetB4 Network
class EfficientNetB4Network(Network):
    def __init__(self):
        super(EfficientNetB4Network, self).__init__()
        num_classes=2
        self.model = self._init_efficientnetB4(num_classes)

    def _init_efficientnetB4(self, num_classes):
        """Initializes EfficientNetB4 with a modified classifier."""
        model = EfficientNetB4()
        model.load_state_dict(
            torch.load(
                efficientnetB4_pretrained_weights,
                weights_only=True
            )
        )

        self._freeze_params(model)

        # Modify classifier
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

        # Unfreeze specific layers
        self._unfreeze_params(model.efficientnet._blocks[-1]._project_conv)
        self._unfreeze_params(model.efficientnet._conv_head)
        self._unfreeze_params(model.classifier)

        return model

    def forward(self, x):
        return self.model(x)

    def _freeze_params(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_params(self, module):
        for param in module.parameters():
            param.requires_grad = True

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T02:01:36.751360Z","iopub.execute_input":"2025-03-02T02:01:36.751666Z","iopub.status.idle":"2025-03-02T02:01:36.757979Z","shell.execute_reply.started":"2025-03-02T02:01:36.751641Z","shell.execute_reply":"2025-03-02T02:01:36.757029Z"},"id":"oq8-bANZC1Eb"}
# Define Xception Network
class XceptionNetwork(Network):
    def __init__(self):
        super(XceptionNetwork, self).__init__()
        num_classes=2
        self.model = self._init_xceptionnet(num_classes)

    def _init_xceptionnet(self, num_classes):
        """Initializes Xception model with a modified classifier."""
        model = xception()
        self._freeze_params(model)

        # Modify classifier
        in_features = model.last_linear.in_features
        model.last_linear = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features, num_classes)
        )

        # Unfreeze specific layers
        self._unfreeze_params(model.last_linear)
        self._unfreeze_params(model.conv4)

        return model

    def forward(self, x):
        return self.model(x)

    def _freeze_params(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_params(self, module):
        for param in module.parameters():
            param.requires_grad = True

# %% [code] {"execution":{"iopub.status.busy":"2025-02-28T06:01:07.472890Z","iopub.execute_input":"2025-02-28T06:01:07.473251Z","iopub.status.idle":"2025-02-28T06:09:44.047349Z","shell.execute_reply.started":"2025-02-28T06:01:07.473226Z","shell.execute_reply":"2025-02-28T06:09:44.045743Z"},"jupyter":{"source_hidden":true}}
train_img_sizes = []
test_img_sizes = []

for file_path in os.listdir(train_folder):
    im = Image.open(os.path.join(train_folder,file_path))
    width, height = im.size
    train_img_sizes.append((file_path, width, height))

for file_path in os.listdir(test_folder):
    im = Image.open(os.path.join(test_folder,file_path))
    width, height = im.size
    test_img_sizes.append((file_path, width, height))

train_img_sizes_df = pd.DataFrame(data=train_img_sizes, columns=["filename", "width", "height"])
train_img_sizes_df.to_csv("train_image_sizes", index=False)

test_img_sizes_df = pd.DataFrame(data=test_img_sizes, columns=["filename", "width", "height"])
test_img_sizes_df.to_csv("test_image_sizes", index=False)

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T02:51:40.119908Z","iopub.execute_input":"2025-03-01T02:51:40.120283Z","iopub.status.idle":"2025-03-01T02:51:40.190842Z","shell.execute_reply.started":"2025-03-01T02:51:40.120253Z","shell.execute_reply":"2025-03-01T02:51:40.189716Z"},"jupyter":{"source_hidden":true,"outputs_hidden":true}}
# Compute frequency count of (width, height)
freq_df = train_image_sizes_df.value_counts(subset=['width', 'height']).reset_index()
freq_df.columns = ['width', 'height', 'count']

# Apply a transformation to make smaller counts more visible
freq_df['scaled_size'] = np.sqrt(freq_df['count']) * 3  # Scale up small values

# Create scatter plot with size representing frequency
fig = px.scatter(
    freq_df, 
    x="width", 
    y="height", 
    size="count",  # Adjusted size for visibility
    title="Frequency Distribution of Image Sizes",
    opacity=0.7
)

fig.update_layout(
    xaxis_title="Width (pixels)",
    yaxis_title="Height (pixels)",
    showlegend=False,
    hovermode="closest",
    width=800,
    height=600,
    margin=dict(l=50, r=50, b=50, t=50, pad=4)
)

fig.show()

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T03:07:19.566422Z","iopub.execute_input":"2025-03-01T03:07:19.566768Z","iopub.status.idle":"2025-03-01T03:07:19.590332Z","shell.execute_reply.started":"2025-03-01T03:07:19.566740Z","shell.execute_reply":"2025-03-01T03:07:19.589293Z"},"jupyter":{"source_hidden":true,"outputs_hidden":true}}
train_image_sizes_df.describe()

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T03:08:31.229075Z","iopub.execute_input":"2025-03-01T03:08:31.229466Z","iopub.status.idle":"2025-03-01T03:08:31.293698Z","shell.execute_reply.started":"2025-03-01T03:08:31.229436Z","shell.execute_reply":"2025-03-01T03:08:31.292634Z"},"jupyter":{"source_hidden":true,"outputs_hidden":true}}
# Compute frequency count of (width, height)
freq_df = test_image_sizes_df.value_counts(subset=['width', 'height']).reset_index()
freq_df.columns = ['width', 'height', 'count']

# Apply a transformation to make smaller counts more visible
freq_df['scaled_size'] = np.sqrt(freq_df['count']) * 3  # Scale up small values

# Create scatter plot with size representing frequency
fig = px.scatter(
    freq_df, 
    x="width", 
    y="height", 
    size="count",  # Adjusted size for visibility
    title="Frequency Distribution of Image Sizes",
    opacity=0.7
)

fig.update_layout(
    xaxis_title="Width (pixels)",
    yaxis_title="Height (pixels)",
    showlegend=False,
    hovermode="closest",
    width=800,
    height=600,
    margin=dict(l=50, r=50, b=50, t=50, pad=4)
)

fig.show()

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T03:08:07.914480Z","iopub.execute_input":"2025-03-01T03:08:07.914865Z","iopub.status.idle":"2025-03-01T03:08:07.930790Z","shell.execute_reply.started":"2025-03-01T03:08:07.914832Z","shell.execute_reply":"2025-03-01T03:08:07.929848Z"},"jupyter":{"source_hidden":true,"outputs_hidden":true}}
test_image_sizes_df.describe()

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T02:01:39.386575Z","iopub.execute_input":"2025-03-02T02:01:39.386857Z","iopub.status.idle":"2025-03-02T02:01:39.392122Z","shell.execute_reply.started":"2025-03-02T02:01:39.386836Z","shell.execute_reply":"2025-03-02T02:01:39.391182Z"}}
def get_data_transforms(model_name, is_train=True):
    """Returns the appropriate transformation pipeline based on the model."""
    
    input_size = configs[model_name]["img_size"]
    if is_train:
        transform = v2.Compose(
            [
                v2.ToTensor(),
                v2.Resize(333),
                v2.CenterCrop(input_size),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                v2.GaussianNoise(),
            ]
        )
    else:
        transform = v2.Compose(
            [
                v2.ToTensor(),
                v2.Resize(512),
                v2.CenterCrop(input_size)
            ]
        )  
    return transform

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T02:01:41.299993Z","iopub.execute_input":"2025-03-02T02:01:41.300327Z","iopub.status.idle":"2025-03-02T02:01:41.306153Z","shell.execute_reply.started":"2025-03-02T02:01:41.300302Z","shell.execute_reply":"2025-03-02T02:01:41.305071Z"}}
class ImageDataset(Dataset):
    def __init__(self, csv_file, image_column, label_column=None, transform=None):
        self.data = pd.read_csv(csv_file, delimiter=",")
        self.transform = transform
        self.filenames = self.data[image_column].tolist()
        self.labels = self.data[label_column].tolist() if label_column else None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        img_path = os.path.join(data, self.filenames[index])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[index]
        return image, self.filenames[index]

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T02:01:43.966512Z","iopub.execute_input":"2025-03-02T02:01:43.966825Z","iopub.status.idle":"2025-03-02T02:01:46.293103Z","shell.execute_reply.started":"2025-03-02T02:01:43.966801Z","shell.execute_reply":"2025-03-02T02:01:46.292423Z"},"id":"qZr1W8ThC1Ea","outputId":"60b4109c-6d21-4c94-d6c5-10da710486a7","jupyter":{"outputs_hidden":false}}
model_name = "xceptionnet"

train_transform = get_data_transforms(model_name=model_name, is_train=True)
dataset = ImageDataset(
    csv_file=train_csv,
    image_column="file_name",
    label_column="label",
    transform=train_transform
)

test_transform = get_data_transforms(model_name=model_name, is_train=False)
test_dataset = ImageDataset(
    csv_file=test_csv,
    image_column="id",
    transform=test_transform
)

validation_split = 0.2
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.seed(seed)
train_indices, val_indices = indices[split:], indices[:split]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_dataloader = DataLoader(
    train_dataset, batch_size=configs[model_name]["batch_size"], shuffle=True
)
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=configs[model_name]["batch_size"],
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=configs[model_name]["batch_size"]
)

torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = configs[model_name]["network"]()
model.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=configs[model_name]["learning_rate"],
    weight_decay=configs[model_name]["weight_decay"]
)

criterion = nn.CrossEntropyLoss()

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T02:01:49.976571Z","iopub.execute_input":"2025-03-02T02:01:49.976858Z","iopub.status.idle":"2025-03-02T05:21:28.644708Z","shell.execute_reply.started":"2025-03-02T02:01:49.976838Z","shell.execute_reply":"2025-03-02T05:21:28.643718Z"}}
model._fit()

# %% [code] {"jupyter":{"source_hidden":true}}
model.load_state_dict(torch.load(configs[model_name]['best_params'], map_location=torch.device('cpu')))
model.eval()
predictions = {}

with torch.no_grad():
    for images, filenames in tqdm(test_dataloader):
        images = images.to(device, dtype=torch.float32)

        # Forward pass
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)  # Get probabilities
        preds = torch.argmax(probs, dim=1)  # Get class predictions

        # Store results
        for filename, pred in zip(filenames, preds.cpu().numpy()):
            predictions[filename] = pred

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T05:29:49.078908Z","iopub.execute_input":"2025-03-02T05:29:49.079261Z","iopub.status.idle":"2025-03-02T05:29:49.085023Z","shell.execute_reply.started":"2025-03-02T05:29:49.079231Z","shell.execute_reply":"2025-03-02T05:29:49.084219Z"}}
def get_base_models_predictions(models, dataloader):
    meta_inputs = []
    targets = []
        
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device, dtype=torch.float32)

            # Initialize predictions container for this batch
            batch_meta_inputs = []

            # Get predictions from each base model
            for base_model in base_models:
                base_model.to(device)
                base_model.eval()
                outputs = base_model(images)
                probs = F.softmax(outputs, dim=1)
                preds, _ = torch.max(probs, dim=1)
                batch_meta_inputs.append(preds.cpu().numpy())

            # Concatenate model predictions
            combined_features = np.stack(batch_meta_inputs, axis=1)
            meta_inputs.extend(combined_features)
            targets.extend(labels)

    return np.array(meta_inputs), np.array(targets)

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T05:29:53.208892Z","iopub.execute_input":"2025-03-02T05:29:53.209221Z","iopub.status.idle":"2025-03-02T05:45:53.385265Z","shell.execute_reply.started":"2025-03-02T05:29:53.209160Z","shell.execute_reply":"2025-03-02T05:45:53.384208Z"}}
base_models = []
for model_name in configs.keys():
        base_model = configs[model_name]["network"]()
        print(model_name)
        best_params = configs[model_name]["best_params"]
        base_model.load_state_dict(torch.load(best_params, map_location=torch.device('cpu'), weights_only= True))
        base_models.append(base_model)

meta_train_inputs, meta_targets = get_base_models_predictions(base_models, val_dataloader)
meta_test_inputs, filenames = get_base_models_predictions(base_models, test_dataloader)

np.save("meta_train.npy",meta_train_inputs)
np.save("meta_label.npy", meta_targets)
np.save("meta_test.npy", meta_test_inputs)

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T05:53:50.094985Z","iopub.execute_input":"2025-03-02T05:53:50.095367Z","iopub.status.idle":"2025-03-02T05:53:50.142700Z","shell.execute_reply.started":"2025-03-02T05:53:50.095335Z","shell.execute_reply":"2025-03-02T05:53:50.141919Z"}}
clf = LogisticRegression(random_state=0).fit(meta_train_inputs, meta_targets)

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T05:55:13.494751Z","iopub.execute_input":"2025-03-02T05:55:13.495086Z","iopub.status.idle":"2025-03-02T05:55:13.499886Z","shell.execute_reply.started":"2025-03-02T05:55:13.495061Z","shell.execute_reply":"2025-03-02T05:55:13.498958Z"}}
preds = clf.predict(meta_test_inputs)

# %% [code] {"execution":{"iopub.status.busy":"2025-03-02T06:01:41.780973Z","iopub.execute_input":"2025-03-02T06:01:41.781363Z","iopub.status.idle":"2025-03-02T06:01:41.803202Z","shell.execute_reply.started":"2025-03-02T06:01:41.781331Z","shell.execute_reply":"2025-03-02T06:01:41.802369Z"}}
df = pd.DataFrame({"id": filenames, "label": preds})
df.to_csv("preds.csv")

# %% [code] {"execution":{"execution_failed":"2025-02-27T00:08:40.225Z"},"id":"y52Voa3KC1Ed","outputId":"2b1bac11-b732-4633-ef6b-a863a0c76824","jupyter":{"source_hidden":true}}
train_losses = logs["train_losses"]
valid_losses = logs["valid_losses"]
train_f1_scores = logs["train_f1_scores"]
valid_f1_scores  = logs["valid_f1_scores"]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(
    np.arange(1, len(train_losses) + 1),
    train_losses,
    label="train loss",
    marker="o",
)
plt.plot(
    np.arange(1, len(valid_losses) + 1),
    valid_losses,
    label="validation loss",
    marker="o",
)
plt.title("loss: train vs validation")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(
    np.arange(1, len(train_f1_scores) + 1),
    train_f1_scores,
    label="train F1-score",
    marker="o",
)
plt.plot(
    np.arange(1, len(valid_f1_scores) + 1),
    valid_f1_scores,
    label="validation F1-score",
    marker="o",
)
plt.title("Loss: Train vs Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(
    np.arange(1, epochs + 1), train_f1_scores, label="Train  F1-score", marker="o"
)
plt.plot(
    np.arange(1, epochs + 1),
    valid_f1_scores,
    label="Validation F1-score",
    marker="o",
)
plt.title("F1-score: Train vs Validation")
plt.xlabel("Epochs")
plt.ylabel("F1-score")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("train_val_metrics.png")
plt.show()
