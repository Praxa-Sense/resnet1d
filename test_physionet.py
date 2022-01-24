"""
test on physionet data

Shenda Hong, Nov 2019
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from util import read_data_physionet_2, read_data_physionet_4, preprocess_physionet
from resnet1d import ResNet1D, MyDataset

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torchsummary import summary

def train():

    batch_size = 32


    # make model
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    kernel_size = 16
    stride = 2
    n_block = 48
    downsample_gap = 6
    increasefilter_gap = 12

    # train and test

    for cv in range(3):
        # make data
        writer = SummaryWriter(f'runs/challenge2017/test_physionet_cv{cv}')
        print("Loading data ...", end="")
        X_train, X_test, Y_train, Y_test, pid_test = read_data_physionet_4()
        print(X_train.shape, Y_train.shape)
        dataset = MyDataset(X_train, Y_train)
        dataset_test = MyDataset(X_test, Y_test)

        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False)
        print("Done!")

        # train
        model = ResNet1D(
            in_channels=1,
            base_filters=128,  # 64 for ResNet1D, 352 for ResNeXt1D
            kernel_size=kernel_size,
            stride=stride,
            groups=32,
            n_block=n_block,
            n_classes=4,
            downsample_gap=downsample_gap,
            increasefilter_gap=increasefilter_gap,
            use_do=True, verbose=False)
        model.to(device)
        model.train()
        summary(model, (X_train.shape[1], X_train.shape[2]), device=device_str)

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        loss_func = torch.nn.CrossEntropyLoss()

        n_epoch = 100
        step = 0
        for ep in range(n_epoch):
            dataloader = DataLoader(dataset, batch_size=batch_size)
            prog_iter = tqdm(dataloader, desc=f"Training {ep}", leave=False)
            for batch_idx, batch in enumerate(prog_iter):

                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                loss = loss_func(pred, input_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

                if batch_idx % 100 == 0:
                    final_pred = torch.argmax(pred, 1)
                    train_acc = torch.sum(final_pred == input_y) / len(input_y)
                    writer.add_scalar('Acc/train', train_acc, step)

                writer.add_scalar('Loss/train', loss.item(), step)
                # if batch_idx >= 4:
                #     break

            scheduler.step(ep)

            # test
            model.eval()
            prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
            all_pred_prob = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(prog_iter_test):
                    input_x, input_y = tuple(t.to(device) for t in batch)
                    pred = model(input_x)
                    all_pred_prob.append(pred.cpu().data.numpy())
            all_pred_prob = np.concatenate(all_pred_prob)
            all_pred = np.argmax(all_pred_prob, axis=1)
            ## vote most common
            final_pred = []
            final_gt = []
            for i_pid in np.unique(pid_test):
                tmp_pred = all_pred[pid_test == i_pid]
                tmp_gt = Y_test[pid_test == i_pid]
                final_pred.append(Counter(tmp_pred).most_common(1)[0][0])
                final_gt.append(Counter(tmp_gt).most_common(1)[0][0])

            ## classification report
            tmp_report = classification_report(final_gt, final_pred, output_dict=True)
            print(confusion_matrix(final_gt, final_pred))
            f1_score = (tmp_report['0']['f1-score'] + tmp_report['1']['f1-score'] + tmp_report['2']['f1-score'] +
                        tmp_report['3']['f1-score']) / 4
            writer.add_scalar('F1/f1_score', f1_score, ep)
            writer.add_scalar('F1/label_0', tmp_report['0']['f1-score'], ep)
            writer.add_scalar('F1/label_1', tmp_report['1']['f1-score'], ep)
            writer.add_scalar('F1/label_2', tmp_report['2']['f1-score'], ep)
            writer.add_scalar('F1/label_3', tmp_report['3']['f1-score'], ep)


if __name__ == "__main__":
    train()
