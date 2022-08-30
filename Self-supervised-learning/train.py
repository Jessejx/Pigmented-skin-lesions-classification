import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import *
from model_our_18 import Model
from Loading_Data import loading_data, HAM10000


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def build_lr_schedulers(optimizers):
    scheduler = WarmupMultiStepLR(optimizer, [400, 800, 900],
                                      gamma=0.1,
                                      warmup_factor=0.1,
                                      warmup_iters=10,
                                      warmup_method='linear')
    return scheduler


# train for one epoch to learn unique features
def train(encoder_q, encoder_k, data_loader, train_optimizer):
    global memory_queue
    encoder_q.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for x_q, x_k, rotate_label in train_bar:
    
        x_q, x_k = x_q.cuda(non_blocking=True), x_k.cuda(non_blocking=True)
        rotate_label = rotate_label.cuda(non_blocking=True)
        ### attention x_q is the query ###
        _, query, rotate_predict  = encoder_q(x_q)

        # shuffle BN

        idx = torch.randperm(x_k.size(0), device=x_k.device)
        _, key, _ = encoder_k(x_k[idx])
        key = key[torch.argsort(idx)]

        score_pos = torch.bmm(query.unsqueeze(dim=1), key.unsqueeze(dim=-1)).squeeze(dim=-1)
        score_neg = torch.mm(query, memory_queue.t().contiguous())
        # [B, 1+M]
        out = torch.cat([score_pos, score_neg], dim=-1)
        # lables
        labels = torch.zeros(x_q.size(0), dtype=torch.long, device=x_q.device)
        # compute loss
        loss_MoCo = F.cross_entropy(out / temperature, labels) 
        loss_Rotate = F.cross_entropy(rotate_predict,rotate_label) 
        lada = 0.1
        loss = loss_MoCo + lada * loss_Rotate
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # momentum update
        for parameter_q, parameter_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))
        # update queue
        memory_queue = torch.cat((memory_queue, key), dim=0)[key.size(0):]

        total_num += x_q.size(0)
        loss_MoCo += loss_MoCo.item() * x_q.size(0)
        loss_Rotate += loss_Rotate.item() * x_q.size(0)
        total_loss += loss.item() * x_q.size(0)
        
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Loss_MoCo: {:.4f} Loss_Rotate: {:.4f}'.format(epoch, epochs, total_loss / total_num, loss_MoCo/ total_num , loss_Rotate/ total_num))

    return total_loss / total_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for each image')
    parser.add_argument('--m', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--momentum', default=0.999, type=float, help='Momentum used for the update of memory bank')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim, m, temperature, momentum = args.feature_dim, args.m, args.temperature, args.momentum
    k, batch_size, epochs = args.k, args.batch_size, args.epochs

    # data prepare
    path = "/home/hsilab/Data/Datasets/SkinCancer_Unlabeled/Pretraining_resize_224"
    train_data = Melanoma_rotate_pair(path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)

    
    # model setup and optimizer config
    model_q = Model(feature_dim).cuda()
    model_k = Model(feature_dim).cuda()
    # initialize
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        # not update by gradient
        param_k.requires_grad = False
        
    optimizer = optim.Adam(model_q.parameters(), lr=6e-4, weight_decay=1e-6)
    # optimizer = optim.SGD(model_q.parameters(), lr=6e-3, momentum=0.9,weight_decay=1e-6)
    schedular = build_lr_schedulers(optimizer)

    # c as num of train class
    # c = len(memory_data.classes)
    c = 7
    # init memory queue as unit random vector ---> [M, D]
    memory_queue = F.normalize(torch.randn(m, feature_dim).cuda(), dim=-1)

    # training loop
    results = {'train_loss': []}
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}'.format(feature_dim, m, temperature, momentum, k, batch_size, epochs)
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model_q, model_k, train_loader, optimizer)
        schedular.step()
        print("!" * 10 + " lr " + "!" * 10, optimizer.param_groups[0]['lr'])
        results['train_loss'].append(train_loss)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/last_e4/{}_results.csv'.format(save_name_pre), index_label='epoch')
        if epoch % 100 == 0:
            torch.save(model_q.state_dict(), 'results/last_e4/{}_epoch.pth'.format(epoch))

