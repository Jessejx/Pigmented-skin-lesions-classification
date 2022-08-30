import numpy as np
import torch
import os
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from Loading_Data import loading_data, HAM10000
from Model import initialize_model
from utils import AverageMeter, plot_confusion_matrix, build_lr_schedulers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, precision_score, accuracy_score, recall_score, f1_score, precision_recall_fscore_support
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings("ignore")


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)

    for i, data in enumerate(train_loader):
        images, labels = data
        N = images.size(0)
        # torch.Size([32, 3, 224, 224]) :32批次，3通道，及输入尺寸
        # print('image shape:',images.size(0), 'label shape',labels.size(0))
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prediction = outputs.max(1, keepdim=True)[1]
        # torch.Size([32, 7])
        # 1所对应的是最大可能性的标签
        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)
        # eq() 判断预测和标签是否相等，再求和，平均，计算每一个批次32张图的准确率
        train_loss.update(loss.item())
        curr_iter += 1
        if (i + 1) % 20 == 0:
            # 每100批次打印
            print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg))
        total_loss_train.append(train_loss.avg)
        total_acc_train.append(train_acc.avg)

    return train_loss.avg, train_acc.avg

def validate(val_loader, model, criterion, epoch):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    val_precision = AverageMeter()
    val_re = AverageMeter()
    val_f1 = AverageMeter()
    
    y_label = []
    y_predict = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0) #
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]
            
            #### calculate
            val_acc.update(accuracy_score(labels.cpu(), prediction.cpu()))
            val_precision.update(precision_score(labels.cpu(), prediction.cpu(), average='weighted'))      
            val_re.update(recall_score(labels.cpu(), prediction.cpu(), average='macro'))
            val_f1.update(f1_score(labels.cpu(), prediction.cpu(), average='weighted'))

  
            val_loss.update(criterion(outputs, labels).item())

            y_label.extend(labels.cpu().numpy())
            y_predict.extend(np.squeeze(prediction.cpu().numpy().T))



    print('------------------------------------------------------------')
    print('testing result, [val loss %.5f], [val acc %.5f], [val precision %.5f], [val recall %.5f], [valf1 %.5f]' % ( val_loss.avg, val_acc.avg, val_precision.avg, val_re.avg, val_f1.avg))

    print('------------------------------------------------------------')

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(y_label, y_predict)
    # 横坐标是真实值，纵坐标是预测值
    # plot the confusion matrix
    plot_labels = ['akiec', 'bcc', 'bkl', 'Dermatofibroma', 'mv', 'vasc', 'mel']
    plot_confusion_matrix(confusion_mtx, plot_labels)

    # Generate a classification report
    report = classification_report(y_label, y_predict, target_names=plot_labels, digits=6)
    print(report)

    return val_loss.avg, val_acc.avg, val_precision.avg, val_re.avg, val_f1.avg

if __name__ == "__main__":
    """
    1. Dataloader
    """
    _, _, dataset = loading_data()
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor() ,
    transforms.Normalize([0.62485343, 0.62214255, 0.620066], [0.17797484, 0.1801317, 0.18257397])])
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    train_test_dataset = HAM10000(dataset, transform=test_transform)
    """
    2. K-fold Cross Validation model evaluation
    """
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    num_epochs = 10
    # Start print
    print('--------------------------------')
    total_loss_val, total_acc_val, total_precision, total_recall, total_f1 = [], [], [], [], []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_test_dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        """
        3. Initialize Model
        """
        model_name = 'resnet18'
        num_classes = 7
        feature_extract = True
        # Initialize the model for this run
        model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
        # """
        state_dict = torch.load("/home/hsilab/Data/workings_ljx/Rotate_MoCo/results/last_e4/1000_epoch.pth")
        # state_dict = torch.load("/home/hsilab/Data/workings_ljx/Computer_science/Contrastive_contrastive_methods/results/moco/MoCo_resnet18_f128_q4096_t0.1_m0.99_k100_B256_e1000_model.pth")
        # state_dict = {k.replace('network.', ''): v for k, v in state_dict.items()}
        model_ft.load_state_dict(state_dict, False)
        print("loading pre-trained weights is true")
        #"""
        
        # Define the device:
        device = torch.device('cuda:0')
        # Put the model on the device:
        model = model_ft.to(device)
	
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = DataLoader(train_test_dataset, batch_size=128, shuffle=False, num_workers=10, sampler=train_subsampler)
        test_loader = DataLoader(train_test_dataset, batch_size=128, shuffle=False, num_workers=10, sampler=test_subsampler)

        # Initialize optimizer and loss function
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(model.parameters(), lr=6e-2, momentum=0.9,weight_decay=1e-6)

        """
        4. Training Model
        """
        # Run the training loop for defined number of epochs
        total_loss_train, total_acc_train  = [], []
        for epoch in range(0, num_epochs):
            # Print epoch
            print(f'Starting epoch {epoch}')
            loss_train, acc_train = train(train_loader, model, criterion, optimizer, epoch)

        print('Training process has finished. Saving trained model.')
        print('Starting testing')
        loss_val, acc_val, pre, recall, f1 = validate(test_loader, model, criterion, 1)

        total_loss_val.append(loss_val)
        
        total_acc_val.append(acc_val)
        total_precision.append(pre)
        total_recall.append(recall)
        total_f1.append(f1)
        

    """
    5. Analyzation
    """

    mean = np.mean(total_acc_val)
    var = np.var(total_acc_val) * 100
    print(total_acc_val)
    print("this is acc")
    print("mean: %f" % mean)
    print("var: %f" % var)
    print(">>>>>>>>>>>>>>>>>>>>>>") 
    
    print(total_precision)
    mean = np.mean(total_precision)
    var = np.var(total_precision) * 100
    print("this is precision")
    print("mean: %f" % mean)
    print("var: %f" % var)
    print(">>>>>>>>>>>>>>>>>>>>>>")
    
    print(total_recall)
    mean = np.mean(total_recall)
    var = np.var(total_recall) * 100
    print("this is recall")
    print("mean: %f" % mean)
    print("var: %f" % var)
    print(">>>>>>>>>>>>>>>>>>>>>>")
    
    print(total_f1)
    mean = np.mean(total_f1)
    var = np.var(total_f1) * 100
    print("this is f1_score")
    print("mean: %f" % mean)
    print("var: %f" % var)
    
