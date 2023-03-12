import torch
import math
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from statistics import mean

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
import os


def data_renew(pairs, emb_type):
    if emb_type == 'onehot':
        tcr = pairs[:, :, 0:21].type(torch.LongTensor).to(device)
        epi = pairs[:, :, 21:-1].type(torch.LongTensor).to(device)
    elif emb_type == 'BLOSUM62':
        tcr = pairs[:, :, 0:20].to(device)
        epi = pairs[:, :, 20:].to(device)
    else:
        tcr = pairs[:, :, 0:5].to(device)
        epi = pairs[:, :, 5:].to(device)

    return tcr, epi

# 定义训练模型
def train_test(model, train_dataset, test_dataset, optimizer, n_epoch, e_type,criterion):

    # train model

    model.train()
    best_loss = 1
    L = torch.zeros(n_epoch, 500)

    for epoch in range(n_epoch):  # 训练的数据量为5个epoch，每个epoch为一个循环
        # 清除缓存
        # torch.cuda.empty_cache()
        # optimizer.zero_grad()

        for i, data in enumerate(train_dataset, 0):

            #loss = 0  # 定义一个变量方便我们对loss进行输出
            #correct = 0  # 定义准确率
            pairs, label = data

            tcr, epi = data_renew(pairs=pairs, emb_type=e_type)
            label = label.unsqueeze(-1).to(device)
            n = tcr.size()[0]

            # 计算输出
            output = torch.unsqueeze(model(tcr, epi), 1)
            pred = output.argmax(dim=2)
            output = output.view(-1, 2)
            # print('tcr.size(),epi.size(),label.size(),pred.size()')
            # print(tcr.size(),epi.size(),label.size(),pred.size())

            pred = pred.view(-1)
            label = label.long().view(-1)
            # print('pred',pred.size(),'label',label.size())
            # print('output',output.size(),output)
            # 计算每个patch的损失与正确率
            loss = criterion(output, label)
            correct = torch.eq(pred, label.long()).sum().float().item()

            # print('correct',correct)
            # loss = loss / n
            acc = correct / n

            if loss < best_loss:
                # if acc < 1.0:
                best_loss = loss
                param = model.state_dict()
            if loss == best_loss:
                # if acc < 1.0:
                param = model.state_dict()

            optimizer.zero_grad()
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新

            '''
                        if i % 10 == 0:
                ni = math.ceil(i / 10)
                L[epoch, ni] = loss
                # running_loss += loss.item()
                print('Epoch:', epoch, 'ni:', ni, ';', 'loss = ', loss, ';', 'acc = ', acc)

            '''

    print('End training. Begin testing')

    good_model = model
    good_model.load_state_dict(param)
    # test model
    good_model.eval()
    good_model.to(device)
    # optimizer.zero_grad()
    y_score = []
    y_test = []
    test_loss = []

    ACC = []
    AUC = []
    PRE = []
    REC = []
    SPE = []
    F1 = []
    MCC = []
    with torch.no_grad():
        for i, data in enumerate(test_dataset, 0):
            #loss = 0  # 定义一个变量方便我们对loss进行输出
            correct = 0  # 定义准确率
            pairs, label = data
            # print(pairs.size())
            tcr, epi = data_renew(pairs=pairs, emb_type=e_type)
            label = label.unsqueeze(-1).to(device)

            n = tcr.size()[0]

            # 计算输出
            output = torch.unsqueeze(good_model(tcr, epi), 1)
            pred = output.argmax(dim=2)
            output = output.view(-1, 2)
            y_score.append(output[:, 1].type(torch.FloatTensor))
            # print('tcr.size(),epi.size(),label.size(),pred.size()')
            # print(tcr.size(),epi.size(),label.size(),pred.size())

            pred = pred.view(-1)
            # pred = output
            label = label.long().view(-1)

            loss = criterion(output, label.long())

            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            '''
            torch.LongTensor(a.numpy())
            '''

            y_test.append(label)
            # y_test.append(label.type(torch.FloatTensor))
            # print('pred',pred.size(),'label',label.size())
            # print('output',output.size(),output)
            # 计算每个patch的损失与正确率
            #print(label, pred)

            [tn, fp], [fn, tp] = confusion_matrix(label, pred)
            #print(tn,fp,fn,tp)
            acc = accuracy_score(label, pred)
            auc = roc_auc_score(label, pred)
            pre = precision_score(label, pred)
            rec = recall_score(label, pred)
            spe = tn / (tn + fn)
            f1 = f1_score(label, pred)
            mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

            # print('correct',correct)
            loss = loss / n
            #acc = correct / n
            ACC.append(acc)
            AUC.append(auc)
            PRE.append(pre)
            REC.append(rec)
            SPE.append(spe)
            F1.append(f1)
            MCC.append(mcc)
            test_loss.append(loss)

            torch.cuda.empty_cache()
    return test_loss, mean(ACC), mean(AUC), mean(PRE), mean(REC), mean(SPE), mean(F1), mean(MCC)




def train_model(model, dataset, optimizer, n_epoch, save_path, b_loss, e_type):
    # 定义batch个数

    model.train()
    best_loss = b_loss
    L = torch.zeros(n_epoch, 500)

    for epoch in range(n_epoch):  # 训练的数据量为5个epoch，每个epoch为一个循环
        # 清除缓存
        # torch.cuda.empty_cache()
        # optimizer.zero_grad()

        for i, data in enumerate(dataset, 0):

            loss = 0  # 定义一个变量方便我们对loss进行输出
            correct = 0  # 定义准确率
            pairs, label = data

            tcr, epi = data_renew(pairs=pairs, emb_type=e_type)
            label = label.unsqueeze(-1).to(device)
            n = tcr.size()[0]

            # 计算输出
            output = torch.unsqueeze(model(tcr, epi), 1)
            pred = output.argmax(dim=2)
            output = output.view(-1, 2)
            # print('tcr.size(),epi.size(),label.size(),pred.size()')
            # print(tcr.size(),epi.size(),label.size(),pred.size())

            pred = pred.view(-1)
            label = label.long().view(-1)
            # print('pred',pred.size(),'label',label.size())
            # print('output',output.size(),output)
            # 计算每个patch的损失与正确率
            loss = criterion(output, label.long())
            correct = torch.eq(pred, label.long()).sum().float().item()

            # print('correct',correct)
            # loss = loss / n
            acc = correct / n

            if loss < best_loss:
                # if acc < 1.0:
                best_loss = loss
                param = model.state_dict()
            if loss == best_loss:
                # if acc < 1.0:
                param = model.state_dict()

            optimizer.zero_grad()
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新

            if i % 10 == 0:
                ni = math.ceil(i / 10)
                L[epoch, ni] = loss
                # running_loss += loss.item()
                print('Epoch:', epoch, 'ni:', ni, ';', 'loss = ', loss, ';', 'acc = ', acc)

    torch.save(param, save_path)

    # torch.save(model, 'model.pkl')                      # 保存整个神经网络的结构和模型参数
    # torch.save(model.state_dict(), 'model_params.pkl')  # 只保存神经网络的模型参数
    return L, best_loss


# 定义测数模型,原来的测试模型


def test_model(model, dataset, emb_type):
    # 定义batch个数

    model.eval()
    model.to(device)
    # optimizer.zero_grad()
    y_score = []
    y_test = []
    test_loss = []

    ACC = []
    AUC = []
    PRE = []
    REC = []
    SPE = []
    F1 = []
    MCC = []
    with torch.no_grad():
        for i, data in enumerate(dataset, 0):

            loss = 0  # 定义一个变量方便我们对loss进行输出
            correct = 0  # 定义准确率
            pairs, label = data
            # print(pairs.size())
            tcr, epi = data_renew(pairs=pairs, emb_type=emb_type)
            label = label.unsqueeze(-1).to(device)

            n = tcr.size()[0]

            # 计算输出
            output = torch.unsqueeze(model(tcr, epi), 1)
            pred = output.argmax(dim=2)
            output = output.view(-1, 2)
            y_score.append(output[:, 1].type(torch.FloatTensor))
            # print('tcr.size(),epi.size(),label.size(),pred.size()')
            # print(tcr.size(),epi.size(),label.size(),pred.size())

            pred = pred.view(-1)
            # pred = output
            label = label.long().view(-1)

            loss = criterion(output, label.long())

            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            '''
            torch.LongTensor(a.numpy())
            '''

            y_test.append(label)
            # y_test.append(label.type(torch.FloatTensor))
            # print('pred',pred.size(),'label',label.size())
            # print('output',output.size(),output)
            # 计算每个patch的损失与正确率

            [tn, fp], [fn, tp] = confusion_matrix(label, pred)
            acc = accuracy_score(label, pred)
            auc = roc_auc_score(label, pred)
            pre = precision_score(label, pred)
            rec = recall_score(label, pred)
            spe = tn / (tn + fn)
            f1 = f1_score(label, pred)
            mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

            # print('correct',correct)
            # loss = loss / n
            acc = correct / n
            ACC.append(acc)
            AUC.append(auc)
            PRE.append(pre)
            REC.append(rec)
            SPE.append(spe)
            F1.append(f1)
            MCC.append(mcc)
            test_loss.append(loss)

            torch.cuda.empty_cache()
    return test_loss, mean(ACC), mean(AUC), mean(PRE), mean(REC), mean(SPE), mean(F1), mean(MCC)