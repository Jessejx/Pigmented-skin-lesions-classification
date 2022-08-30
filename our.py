import time
import math
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from load_data import *
from sklearn.metrics import classification_report
from torchvision.utils import save_image
import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


# test using a knn monitor
def test(net, memory_data_loader, test_data_loader, epoch):
    net.eval()
    classes = 7
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    feature_test = []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            _, feature, _= net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
            target = target.tolist()
            feature_test.extend(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_test = torch.tensor(feature_test, device=feature_bank.device)
        # feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)

        # loop test data to predict the label by weighted knn search
        y_label = []
        y_pred = []
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            _, feature, _ = net(data)
            pred_labels = knn_predict(feature, feature_bank, feature_test, classes, 100, 0.1)
            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, epochs, total_top1 / total_num * 100))
            y_label.extend(target.cpu().numpy())
            y_pred.extend(pred_labels[:,0].cpu().numpy())
        plot_labels = ['a','b','c','d','f','g','e']
        y_pred = np.asarray(y_pred)
        y_label = np.asarray(y_label)
        report = classification_report(y_label,y_pred,target_names=plot_labels, digits=5)

    return total_top1 / total_num * 100, report

# train for one epoch to learn unique features
def train(encoder_q, encoder_k, data_loader, train_optimizer):
    global memory_queue
    encoder_q.train()
    total_loss, total_num, train_bar, n = 0.0, 0, tqdm(data_loader), 0

    for x_q, x_k, rotate_label in train_bar:
        """
        if n % 1 == 0:
            save_image(x_q, 'images/x_q/in_{}.jpg'.format(n))
            save_image(x_k, 'images/x_k/out_{}.jpg'.format(n))
        # """
        x_q, x_k = x_q.cuda(non_blocking=True), x_k.cuda(non_blocking=True)
        rotate_label = rotate_label.cuda()
        ### attention x_q is the query ###
        _, query, rotate_predict = encoder_q(x_q)

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
        loss_Rotate = F.cross_entropy(rotate_predict, rotate_label)
        lada = 0.1
        loss = loss_MoCo  + lada * loss_Rotate
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

        train_bar.set_description(
            'Train Epoch: [{}/{}] Loss: {:.4f} Loss_MoCo: {:.4f} Loss_Rotate: {:.4f}'.format(epoch, epochs,
                                                                                             total_loss / total_num,
                                                                                             loss_MoCo ,
                                                                                             loss_Rotate))
        n +=1
    return total_loss / total_num


# MoCo train
def MoCo_train(encoder_q, encoder_k, data_loader, train_optimizer):
    global memory_queue
    encoder_q.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for i, (x_q, x_k, _) in train_bar:
        print(i)
        # save_image(x_q,)
        x_q, x_k = x_q.cuda(non_blocking=True), x_k.cuda(non_blocking=True)
        _, query = encoder_q(x_q)
        _, key = encoder_k(x_k)

        batch_size = query.size(0)
        feature_dim = query.size(1)

        score_pos = torch.bmm(query.view(batch_size, 1, feature_dim), key.view(batch_size, feature_dim, 1))
        score_pos = torch.squeeze(score_pos, dim=1)
        score_neg = torch.mm(query, memory_queue.t().contiguous())
        # [B, 1+M]
        out = torch.cat([score_pos, score_neg], dim=1)
        # compute loss
        loss = F.cross_entropy(out / temperature, torch.zeros(batch_size, dtype=torch.long, device=x_q.device))

        # ---------------- symmetry loss can improve the performance
        _, query2 = encoder_q(x_k)
        _, key2 = encoder_k(x_q)

        score_pos = torch.bmm(query2.view(batch_size, 1, feature_dim), key2.view(batch_size, feature_dim, 1))
        score_pos = torch.squeeze(score_pos, dim=1)
        score_neg = torch.mm(query2, memory_queue.t().contiguous())
        # [B, 1+M]
        out = torch.cat([score_pos, score_neg], dim=1)
        # compute loss
        loss2 = F.cross_entropy(out / temperature, torch.zeros(batch_size, dtype=torch.long, device=x_q.device))
        # ----------------
        loss = (loss + loss2) / 2.0

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # momentum update
        for parameter_q, parameter_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))
        # update queue
        memory_queue = torch.cat((memory_queue, key, key2), dim=0)[2 * batch_size:]

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num

# 如果optimizer是SGD的話，我們根據epoch來調整learning rate，避免learning rate都不變化
def adjust_learning_rate(learning_rate, optimizer, epoch, total_epochs, cosine):
    lr = learning_rate
    lr_decay_rate = 0.2
    if cosine:
        eta_min = lr * (lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / total_epochs)) / 2
    else:
        lr_decay_epochs = np.array([math.floor(total_epochs*0.5), math.floor(total_epochs*0.75), math.floor(total_epochs*0.875)])
        steps = np.sum(epoch > np.asarray(lr_decay_epochs))
        if steps > 0:
            lr = lr * (lr_decay_rate ** steps)

    print("LR:{}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    feature_dim = 128  
    m = 4096  # moco queue大小
    temperature, momentum = 0.1, 0.99  # 超參數設置
    k = 100  
    batch_size = 256
    epochs = 50
    model_name = 'resnet18'  
    Learning_method = 'our'  
    LR = 6e-2

    path = "/media/crb/3CC23B50C23B0DA0/workings_ljx/Pretrain_huge/origin/origin_s"
    train_data = Melanoma_rotate_pair(path)
        # train_data = CustomDataPair(root='~/cutone_data/cifar10_pic_10per/train/', transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True,
                              drop_last=True)

    _, df_train, df_val = loading_data()
    memory_loader = HAM10000(df_train, transform=test_transform)
    memory_loader = DataLoader(memory_loader, batch_size=batch_size, shuffle=False, num_workers=10)
    test_loader = HAM10000(df_val, transform=test_transform)
    test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False, num_workers=10)

    print('running our method')
    model_q = Model(feature_dim).cuda()
    model_k = Model(feature_dim).cuda()

    # init memory queue as unit random vector ---> [M, D]
    memory_queue = F.normalize(torch.randn(m, feature_dim).cuda(), dim=-1)

    optimizer = optim.SGD(model_q.parameters(), lr=LR, momentum=0.9, weight_decay=1e-6)
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False

    c = 7

    # training loop
    results = {'train_loss': [], 'total_time': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_f{}_q{}_t{}_m{}_k{}_B{}_e{}'.format(Learning_method, model_name, feature_dim, m, temperature,
                                                               momentum, k, batch_size, epochs)
    best_acc = 0.0

    begin_time = time.time()
    for epoch in range(1, epochs + 1):
        if type(optimizer) is optim.SGD:
            adjust_learning_rate(LR, optimizer, epoch, epochs, cosine=True)

        # train_loss = train(model_q, model_k, train_loader, optimizer)
        #
        # results['train_loss'].append(train_loss)
        # results['total_time'].append(time.time() - begin_time)

        # KNN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        test_acc_1, report = test(model_q, memory_loader, test_loader, epoch)

        results['test_acc@1'].append(test_acc_1)
        # results['test_acc@5'].append(test_acc_5)
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            print('************************************************')
            print('best record: [epoch %d], [test acc %.5f]' % (epoch, best_acc))
            print('************************************************')
            print(report)


