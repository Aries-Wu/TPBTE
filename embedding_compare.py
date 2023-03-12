
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
from tpbte import Model
from embedding import MyDataset
from train_test import train_model, test_model





# 定义训练的模型
vocab = 21 # onehot使用多少个词编码氨基酸
d_model = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# device = 'cpu'
criterion = nn.CrossEntropyLoss()
path = '220.csv'
save_p = 'embedding.pkl'
b_size = 128
n_epoch = 2


embedding_type = {'BLOSUM62', 'Atchley','onehot'}
for i,t in enumerate(embedding_type):
    print(t)
    data = MyDataset(path,type)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train, test = torch.utils.data.random_split(data, [train_size, test_size])
    print(train_size, test_size)
    train_dataset = DataLoader(dataset=train, batch_size=b_size, shuffle=True, drop_last=True,)
    test_dataset = DataLoader(dataset=test, batch_size=b_size, shuffle=True, drop_last=False)
    model1 = Model(src_vocab=vocab, tgt_vocab=vocab, emb_type=t, N=6, h=8, d_model=d_model, dropout=0.1)
    model1 = model1.to(device)
    good_model = model1
    opt = torch.optim.Adam(model1.parameters(), lr=0.00005, betas=(0.9, 0.98), eps=1e-9)

    # 3、训练模型，并保存参数

    print('')
    start = time.time()
    result = train_model(model1, train_dataset, opt, n_epoch=n_epoch, save_path=save_p, b_loss=10, e_type=t)
    end = time.time()
    print('the time used is : ', end - start)
    torch.cuda.empty_cache()

    # 4、测试模型，保存测试结果
    #    导入训练好的模型参数
    good_model.load_state_dict(torch.load(save_p))
    te_loss, acc, auc, pre, recall, spe, f1, mcc = test_model(good_model, test_dataset, e_type=t)
    print('Accuracy, AUC, precision, recall, specificity, f1, mcc')
    print(acc, auc, pre, recall, spe, f1, mcc)

