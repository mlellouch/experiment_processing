import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from build_dataset import ConversionParams, load_multiple_files
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_size, labels=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 256, 3, padding=1)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(1, 2)
        self.pool3 = nn.MaxPool2d(1, 2)
        self.pool4 = nn.MaxPool2d(1, 2)

        # self.final_conv_size = (((input_size[0]) // 2) // 2) // 2
        vals = {
            25: 3,
            50: 7,
            100: 14
        }
        self.final_conv_size = vals[input_size[0]]
        self.fc1 = nn.Linear(32 * input_size[0] * input_size[1], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, labels)

        self.final = nn.Linear(input_size[0] * input_size[1], 256, bias=False)
        self.final1 = nn.Linear(256, 256, bias=False)
        self.final2 = nn.Linear(256, 256, bias=False)
        self.final3 = nn.Linear(256, 128, bias=False)
        self.final4 = nn.Linear(128, 64, bias=False)
        self.final5 = nn.Linear(64, 32, bias=False)
        self.final6 = nn.Linear(32, labels, bias=False)



    def forward(self, x):
        # x1 = F.relu(self.conv1(x))
        # x2 = F.relu(self.conv2(x1))
        # x3 = F.relu(self.conv3(x2))
        # # x4 = F.relu(self.conv4(x3))
        #
        #
        # x10 = torch.flatten(x3, 1) # flatten all dimensions except batch
        # x11 = F.relu(self.fc1(x10))
        # x12 = F.relu(self.fc2(x11))
        # x13 = self.fc3(x12)
        # return x13

        x1 = torch.flatten(x, 1) # flatten all dimensions except batch
        x2 = F.relu(self.final(x1))
        x3 = F.relu(self.final1(x2))
        x4 = F.relu(self.final2(x3))
        x5 = F.relu(self.final3(x4))
        x6 = F.relu(self.final4(x5))
        x7 = F.relu(self.final5(x6))
        x8 = self.final6(x7)
        return x8

        # x1 = torch.flatten(x, 1) # flatten all dimensions except batch
        # x2 = F.relu(self.final(x1))
        # x3 = F.relu(self.final3(x2))
        # x4 = F.relu(self.final4(x3))
        # x5 = F.relu(self.final5(x4))
        # x6 = self.final6(x5)
        # return x6




def blurs_to_labels(blurs: list):
    sorted_blurs = list(set(blurs.copy()))
    sorted_blurs.sort()
    return [sorted_blurs.index(b) for b in blurs]


class FixationalEyeMovementData(Dataset):
    data: torch.tensor

    def __init__(self, data, blurs):
        self.data = data
        self.blurs = blurs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur_data = self.data[idx]
        cur_data = cur_data.reshape([1, cur_data.shape[0], cur_data.shape[1]])
        cur_data = torch.nan_to_num(cur_data)
        if random.random() > 0.5:
            cur_data[:, :, 0] *= -1
            cur_data[:, :, 2] *= -1

        if random.random() > 0.5:
            cur_data[:, :, 1] *= -1
            cur_data[:, :, 3] *= -1

        return cur_data, self.blurs[idx]


def test_network(net, test_set):
    X_test, y_test = test_set
    correct = 0
    incorrect = 0
    outputs = []
    with torch.no_grad():
        for x, y in zip(X_test, y_test):
            x = x.to(device=device)
            x = torch.nan_to_num(x)
            x = x.reshape([1, 1, x.shape[0], x.shape[1]])
            pred = net(x)
            pred_label = torch.argmax(pred)
            outputs.append(pred_label.item())
            if (pred_label == y).item():
                correct += 1
            else:
                incorrect += 1

    return correct / (correct + incorrect)

def train_epochs(net, dataset, test_set, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device=device), labels.to(dtype=torch.long, device=device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = test_network(net, test_set)
        print(f'learning_loss={running_loss} test accuracy={acc}')


def train_net(data, blurs, epochs=100):
    X = data
    Y = blurs_to_labels(blurs)
    test_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    X_train, X_test = torch.tensor(np.array(X_train), device=device, dtype=torch.float32), torch.tensor(np.array(X_test), device=device, dtype=torch.float32)
    y_train, y_test = torch.tensor(np.array(y_train), device=device, dtype=torch.float32), torch.tensor(np.array(y_test), device=device, dtype=torch.float32)

    print(X_train.shape)
    net = Net(X_train[0].shape)
    net = net.to(device=device)
    net.train()

    dataset = FixationalEyeMovementData(X_train, y_train)
    train_epochs(net, dataset, (X_test, y_test), epochs=epochs)
    return net


def run_deep_learning():
    params = ConversionParams(
        relative_to_initial_position=True,
        relative_to_average=False,
        normalize=True,
        scaleless_normalization=False,
        scaled_normalization=True,
        scaled_normalization_size=25,
        add_metadata=False,
        add_speed=True,
        mark_non_samples=True,
        max_samples=50
    )

    paths = [
        '../../../../outputs/preprocessed_outputs/FGBS/case3_aligned_blur/samples.csv',
        '../../../../outputs/preprocessed_outputs/FGBS/pair/first/samples.csv',
    ]

    blurs, data = load_multiple_files(paths, params=params, ignore_zero_blurs=True)
    train_net(data, blurs, 600)



if __name__ == '__main__':
    run_deep_learning()
