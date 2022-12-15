import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from data.multi_mnist import MultiMNIST
from net.lenet import MultiLeNetR, MultiLeNetO
from optimizers.pcgrad import PCGrad
from utils import create_logger

import argparse


def main(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ---------------------------------------------------------

    accuracy = lambda logits, gt: ((logits.argmax(dim=-1) == gt).float()).mean()
    to_dev = lambda inp, dev: [x.to(dev) for x in inp]
    logger = create_logger('Main')

    global_transformer = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    train_dst = MultiMNIST(args.data_path,
                           train=True,
                           download=True,
                           transform=global_transformer,
                           multi=True)
    train_loader = torch.utils.data.DataLoader(train_dst,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4)

    val_dst = MultiMNIST(args.data_path,
                         train=False,
                         download=True,
                         transform=global_transformer,
                         multi=True)
    val_loader = torch.utils.data.DataLoader(val_dst,
                                             batch_size=100,
                                             shuffle=True,
                                             num_workers=1)
    nets = {
        'rep': MultiLeNetR().to(DEVICE),
        'L': MultiLeNetO().to(DEVICE),
        'R': MultiLeNetO().to(DEVICE)
    }
    param = [p for v in nets.values() for p in list(v.parameters())]
    optimizer = getattr(torch.optim, args.inner_optimizer)(param, lr=args.lr)
    optimizer = PCGrad(optimizer)

    for ep in range(args.num_epoch):
        for net in nets.values():
            net.train()
        for batch in train_loader:
            mask = None
            optimizer.zero_grad()
            img, label_l, label_r = to_dev(batch, DEVICE)
            rep, mask = nets['rep'](img, mask)
            out_l, mask_l = nets['L'](rep, None)
            out_r, mask_r = nets['R'](rep, None)

            losses = [F.nll_loss(out_l, label_l), F.nll_loss(out_r, label_r)]
            optimizer.multi_loss_backward(losses)
            # sum(losses).backward()
            optimizer.step()

        losses, acc = [], []
        for net in nets.values():
            net.eval()
        for batch in val_loader:
            img, label_l, label_r = to_dev(batch, DEVICE)
            mask = None
            rep, mask = nets['rep'](img, mask)
            out_l, mask_l = nets['L'](rep, None)
            out_r, mask_r = nets['R'](rep, None)

            losses.append([
                F.nll_loss(out_l, label_l).item(),
                F.nll_loss(out_r, label_r).item()
            ])
            acc.append(
                [accuracy(out_l, label_l).item(),
                 accuracy(out_r, label_r).item()])
        losses, acc = np.array(losses), np.array(acc)
        logger.info('epoches {}/{}: loss (left, right) = {:5.4f}, {:5.4f}'.format(
            ep, args.num_epoch, losses[:, 0].mean(), losses[:, 1].mean()))
        logger.info(
            'epoches {}/{}: accuracy (left, right) = {:5.3f}, {:5.3f}'.format(
                ep, args.num_epoch, acc[:, 0].mean(), acc[:, 1].mean()))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Inference script for trained models.')

    parser.add_argument("--data-path", type=str, default="/storage/Pytorch-PCGrad/data")
    parser.add_argument("--output_dir", type=str, default="/storage/Pytorch-PCGrad/finetune/lines")

    parser.add_argument("--inner_optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epoch", type=int, default=100)

    parser.add_argument("--line_num", type=int, default=100000)

    args = parser.parse_args()

    main(args)
