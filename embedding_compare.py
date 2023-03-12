import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
from tpbte import Model
from embedding import MyDataset
from trainAndtest import train_test
import pandas as pd


vocab = 21 
d_model = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# device = 'cpu'

criterion = nn.CrossEntropyLoss()
model1 = Model(src_vocab=vocab, tgt_vocab=vocab, emb_type=e_type, N=6, h=8, d_model=d_model,dropout=0.1, device=device)
model1 = model1.to(device)
opt = torch.optim.Adam(model1.parameters(), lr=0.00005, betas=(0.9, 0.98), eps=1e-9)
n_epoch = 200

def single(load_path, param_path, epitope,model,epoch):
    save_p = param_path
    epitope = epitope
    print('Epitope:', epitope)

    path_i = load_path
    data = MyDataset(path_i, emb_type=e_type)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train, test = torch.utils.data.random_split(data, [train_size, test_size])
    print(train_size, test_size)
    b_size = 32

    train_dataset = DataLoader(dataset=train, batch_size=b_size, shuffle=True, drop_last=True)
    test_dataset = DataLoader(dataset=test, batch_size=b_size, shuffle=True, drop_last=False)


    print('')
    start = time.time()
    #result = train_model(model1, train_dataset, opt, n_epoch=n_epoch, save_path=save_p, b_loss=10, e_type=e_type)
    te_loss, acc, auc, pre, recall, spe, f1, mcc = train_test(model=model, train_dataset=train_dataset,test_dataset=test_dataset,n_epoch=n_epoch,
                                                              optimizer=opt,e_type=e_type,criterion=criterion)
    end = time.time()
    print('the time used is : ', end - start)
    torch.cuda.empty_cache()

    #te_loss, acc, auc, pre, recall, spe, f1, mcc = test_model(good_model, test_dataset, emb_type=e_type)
    print('Accuracy, AUC, precision, recall, specificity, f1, mcc')
    print(acc, auc, pre, recall, spe, f1, mcc)

load_path = 'emb_compare.csv'
param_path = 'emb_compare.pkl'
epitope = 'random'

embedding_type = {'BLOSUM62', 'Atchley','onehot'}
for i,e_type in enumerate(embedding_type):
    single(load_path, param_path, epitope, model1, n_epoch)

