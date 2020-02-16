import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from utils.bar_show import progress_bar

from feedback_alignment import KPLinear, FALinear

parser = argparse.ArgumentParser(description='DL without Weight Transport PyTorch Implementation.')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='', required=True)
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='CIFAR10_pretrain')
parser.add_argument('--method', type=str, default='fa', required=True)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('-b', type=int, default=256, help="Batch Size")
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=180)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=12)

cfg = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.pretrain_dir)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)


def get_dataloader(root, batch_szie, num_workers):
    train_transform_list = [transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
    eval_transform_list = [transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
    train_transform_list = transforms.Compose(train_transform_list)
    eval_transform_list = transforms.Compose(eval_transform_list)
    train_dataset = CIFAR10(root=root, train=True, download=True,
                            transform=train_transform_list)
    train_loader = DataLoader(train_dataset, batch_size=batch_szie, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    eval_dataset = CIFAR10(root=root, train=False, download=True,
                           transform=eval_transform_list)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_szie, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, eval_loader


class Model(nn.Module):
    def __init__(self, method):
        super(Model, self).__init__()
        if method == 'fa':
            linear = FALinear
        elif method == 'kp':
            linear = KPLinear
        else:
            NameError("Linear Type not Implement")
        self.fc1 = linear(3 * 32 * 32, 120)
        self.fc2 = linear(120, 10)

    def forward(self, x):
        print(x.shape)
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    train_loader, eval_loader = get_dataloader(cfg.data_dir, cfg.b, cfg.num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model(cfg.method).to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    if cfg.method == 'fa':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=1e-4, momentum=0.9, weight_decay=0.001, nesterov=True)
    elif cfg.method == 'kp':
        optimizer = KPSGD()
    criterion = torch.nn.CrossEntropyLoss()
    lr_schedu = torch.optim.lr_scheduler.MultiStepLR(optimizer, [90, 150, 200], gamma=0.1)
    summary_writer = SummaryWriter(cfg.log_dir)

    if cfg.pretrain:
        ckpt = torch.load(os.path.join(cfg.ckpt_dir, f'checkpoint.t7'))
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print('===> Load last checkpoint data')
    else:
        start_epoch = 0
        print('===> Start from scratch')

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss, correct, total = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()  # compute the .grad for all weights
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            if batch_idx % cfg.log_interval == 0:  # every log_interval mini_batches...
                summary_writer.add_scalar('Loss/train', train_loss / (batch_idx + 1),
                                          epoch * len(train_loader) + batch_idx)
                summary_writer.add_scalar('Accuracy/train', 100. * correct / total,
                                          epoch * len(train_loader) + batch_idx)
                summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'],
                                          epoch * len(train_loader) + batch_idx)

    def test(epoch):
        global best_acc
        model.eval()

        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(eval_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(eval_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

                if batch_idx % cfg.log_interval == 0:  # every log_interval mini_batches...
                    summary_writer.add_scalar('Loss/test', test_loss / (batch_idx + 1),
                                              epoch * len(train_loader) + batch_idx)
                    summary_writer.add_scalar('Accuracy/test', 100. * correct / total,
                                              epoch * len(train_loader) + batch_idx)
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving Models..')
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(cfg.ckpt_dir, '{}_checkpoint.t7'.format(cfg.log_name)))
            best_acc = acc

    for epoch in range(start_epoch, cfg.max_epochs):
        train(epoch)
        test(epoch)
        lr_schedu.step(epoch)
    summary_writer.close()


if __name__ == '__main__':
    main()
