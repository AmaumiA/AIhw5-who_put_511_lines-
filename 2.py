import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# 定义多模态融合模型
class MultimodalModel(nn.Module):
    def __init__(self, text_input_dim, image_input_dim, output_dim):
        super(MultimodalModel, self).__init__()

        # 文本处理部分
        self.text_embedding = nn.Embedding(text_input_dim, embedding_dim=256)
        self.text_rnn = nn.LSTM(256, 128, batch_first=True)

        # 图像处理部分
        # self.image_fc = nn.Linear(image_input_dim, 128)
        self.image_fc = nn.Linear(224, image_input_dim)
        # 融合部分
        self.fusion_fc = nn.Linear(688896, 64)
        # 输出层
        self.output_fc = nn.Linear(64, output_dim)

    def forward(self, text, image):
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        # tokens = tokenizer(text, return_tensors="pt")
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # 获取模型输出（嵌入）
        with torch.no_grad():
            outputs = model(**tokens)
        # 获取最后一层的输出作为文本的嵌入向量
        text_output = outputs.last_hidden_state[:, 0, :]  # 取CLS token的输出作为文本嵌入
        # text_embedding = self.text_embedding(text_embedding)
        # _, (text_output, _) = self.text_rnn(text_embedding)
        # print(text_output.shape)
        # print(text_output.size())

        image_output = torch.relu(self.image_fc(image))
        image_output = image_output.view(image_output.size(0), -1)
        # print(image_output.size())

        # print(image_output.shape)
        # 融合
        fusion_input = torch.cat((text_output, image_output), dim=1)
        fusion_output = torch.relu(self.fusion_fc(fusion_input))

        output = self.output_fc(fusion_output)
        return output


# 模型超参数
text_input_dim = 10000  # 词汇表大小
image_input_dim = 1024  # 图像特征维度
output_dim = 5

# 创建模型、损失函数和优化器
model = MultimodalModel(text_input_dim, image_input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


class CustomDataset(Dataset):
    def __init__(self, data_folder, txt):
        self.data_folder = data_folder
        self.txt = txt
        self.samples = self._load_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]

        # 读取文本文件
        text_file_path = os.path.join(self.data_folder, sample_path + '.txt')
        with open(text_file_path, 'r', encoding='utf-8', errors='ignore') as text_file:
            text_data = text_file.read()

        # 读取图像文件
        image_file_path = os.path.join(self.data_folder, sample_path + '.jpg')
        image = Image.open(image_file_path).convert('RGB')

        img = image.resize((224, 224), Image.Resampling.LANCZOS)
        img = np.asarray(img, dtype='float32')
        image_data = img.transpose(2, 0, 1)
        image = torch.Tensor(image_data)

        return {'text': text_data, 'image': image, 'label': label}

    def _load_samples(self):
        samples = []
        label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2, 'null': 3}
        with open(self.txt, 'r', encoding='utf-8') as f:
            label_str = f.read().split('\n')[1:]
        for line in label_str:
            # print(line)
            try:
                guid, label = line.split(',')
                label = label_mapping.get(label, -1)
            except:
                continue
            if label != -1:
                samples.append((guid, label))
        return samples


data_folder = './实验五数据/data'
train_txt = './实验五数据/train.txt'
test_txt = './实验五数据/test_without_label.txt'
custom_dataset = CustomDataset(data_folder, txt=train_txt)
test_dataset = CustomDataset(data_folder, txt=test_txt)
# print(len(custom_dataset))
# print(len(test_dataset))

dataset_size = len(custom_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0  # 初始化epoch损失为0
    total_batches = len(train_dataloader)

    for batch_idx, batch in enumerate(train_dataloader):
        text_data_batch = batch['text']
        image_data_batch = batch['image']
        labels_batch = batch['label']

        optimizer.zero_grad()
        output = model(text_data_batch, image_data_batch)
        loss = criterion(output, labels_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # 累积每个batch的损失

        print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{total_batches}, Loss: {loss.item()}')

    average_epoch_loss = epoch_loss / total_batches
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {average_epoch_loss}')

model.eval()
total_val_loss = 0.0
total_val_samples = 0

with torch.no_grad():
    for batch in val_dataloader:
        text_data_batch = batch['text']
        image_data_batch = batch['image']
        labels_batch = batch['label']

        output = model(text_data_batch, image_data_batch)
        val_loss = criterion(output, labels_batch)

        total_val_loss += val_loss.item()
        total_val_samples += len(labels_batch)

average_val_loss = total_val_loss / total_val_samples
print(f'Validation Loss: {average_val_loss}')


test_text_data = []
test_image_data = []

# 创建测试集数据加载器
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
total_test_loss = 0.0
total_test_samples = 0
predictions = []

# print(len(test_dataloader))
# with torch.no_grad():
#     for test_batch in test_dataloader:
#         try:
#             test_text_data_batch = test_batch['text']
#             test_image_data_batch = test_batch['image']
#             labels_batch = test_batch['label']
#
#             output = model(test_text_data_batch, test_image_data_batch)
#             _, predicted_labels = torch.max(output, 1)
#             predictions.extend(predicted_labels.cpu().numpy())
#             test_loss = criterion(output, labels_batch)
#
#             total_test_loss += test_loss.item()
#             total_test_samples += len(labels_batch)
#         except Exception as e:
#             pass

with torch.no_grad():
    for test_batch in test_dataloader:
        test_text_data_batch = test_batch['text']
        test_image_data_batch = test_batch['image']
        labels_batch = test_batch['label']

        output = model(test_text_data_batch, test_image_data_batch)
        _, predicted_labels = torch.max(output, 1)
        predictions.extend(predicted_labels.cpu().numpy())
        test_loss = criterion(output, labels_batch)

        total_test_loss += test_loss.item()
        total_test_samples += len(labels_batch)

# average_test_loss = total_test_loss / total_test_samples
# print(f'Test Loss: {average_test_loss}')

print(predictions)

# label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2, 'null': 3}
# predicted_tags = [key for key, value in label_mapping.items() if value in predictions]
#
# # 读取原始的 "test_without_label.txt" 内容，获取guid列表
# with open('F:/Users/Desktop/rgzn/hw5/实验五数据/test_without_label.txt', 'r') as f:
#     lines = f.readlines()
#     guid_list = [line.split(',')[0] for line in lines[1:]]
#
# # 生成新文件内容
# output_lines = ['guid,tag\n'] + [f'{guid},{tag}\n' for guid, tag in zip(guid_list, predicted_tags)]
#
# # 写入新文件
# with open('test.txt', 'w') as f:
#     f.writelines(output_lines)

predicted_tags = [str(label) for label in predictions]

with open('./实验五数据/test_without_label.txt', 'r') as f:
    lines = f.readlines()
    guid_list = [line.split(',')[0] for line in lines[1:]]

output_lines = ['guid,tag\n'] + [f'{guid},{tag}\n' for guid, tag in zip(guid_list, predicted_tags)]

with open('./实验五数据/test_without_label.txt', 'w') as f:
    f.writelines(output_lines)
with open('./实验五数据/test_without_label.txt', 'r', encoding='utf-8') as file:
    content = file.read()

modified_content = content.replace(',0', ',positive')
modified_content = modified_content.replace(',1', ',neutral')
modified_content = modified_content.replace(',2', ',negative')
with open('./实验五数据/test_without_label.txt', 'w', encoding='utf-8') as file:
    file.write(modified_content)

