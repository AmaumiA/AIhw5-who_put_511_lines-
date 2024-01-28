使用pytorch环境
需要安装的库如下：

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

hw5文件夹中的三个py文件
2.py 主要代码
img.py 只使用图像进行消融实验的代码
text.py 只使用文本进行消融实验的代码