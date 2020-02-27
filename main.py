import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True

import torchvision
from torch.utils.data import DataLoader
from utils.preprocess import *
from utils.bar_show import progress_bar

# Training settings
parser = argparse.ArgumentParser(description='KP Implementation')
parser.add_argument('--root_dir', type=str, default=".")
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='res18')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='CIFAR100_pretrain')
parser.add_argument('--model', type=str, default='res18')

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--kp_decay', type=float, default=0.9)
parser.add_argument('-b', type=int, default=256, help="Batch Size")
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=180)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=18)

# GLOBAL
cfg = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0

cfg.log_dir = os.path.abspath("~/scratch/{}/logs/{}".format(cfg.root_dir, cfg.log_name))
cfg.ckpt_dir = os.path.abspath("~/scratch/{}/ckpt/{}".format(cfg.root_dir, cfg.pretrain_dir))

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg.model == 'res18':
        from resnet import resnet18
        model = resnet18().to(device)
        print("Runing Baseline ResNet18 (Full Precision)")
    elif cfg.model == 'res50':
        from resnet import resnet50
        model = resnet50().to(device)
        print("Runing Baseline ResNet50 (Full Precision)")
    else:
        assert False, 'Model Unknown !'
    dataset = torchvision.datasets.CIFAR100

    print('===> Preparing data ..')
    train_dataset = dataset(root=cfg.data_dir, train=True, download=True,
                            transform=cifar_transform(cifar=100, is_training=True))
    train_loader = DataLoader(train_dataset, batch_size=cfg.b, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)

    eval_dataset = dataset(root=cfg.data_dir, train=False, download=True,
                           transform=cifar_transform(cifar=100, is_training=False))
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.b, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr / cfg.kp_decay, momentum=0.9, weight_decay=cfg.wd)
    # optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.wd)
    lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [90, 150, 200], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    summary_writer = SummaryWriter(cfg.log_dir)

    if cfg.pretrain:
        ckpt = torch.load(os.path.join(cfg.ckpt_dir, 'checkpoint.t7'))
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

            for param in model.parameters():
                param.data.mul_(cfg.kp_decay)

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
