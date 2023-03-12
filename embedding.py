import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from Bio.Align import substitution_matrices
import pickle as pk



# one-hot
def onehot(TCR, Epitope, Label):

    alphabet = 'XARNDCQEGHILKMFPSTWYV'

    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))


    expand = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).view(1,20)
    expand = expand.type(torch.LongTensor)
    

    train_TCR = torch.zeros(1,20)
    train_TCR = train_TCR.type(torch.LongTensor)
    train_Epi = train_TCR

#     train_Label = torch.ones(5836,1)
#     train_Label = train_Label.type(torch.LongTensor)


    for i, t in enumerate(TCR):
        t_encoded = [char_to_int[char] for char in t]
        t_encoded = torch.LongTensor(t_encoded).view(1,-1)
        t_all = torch.cat((t_encoded, expand),1)
        t_last = t_all[0,0:20].view(1, 20)
        train_TCR = torch.cat((train_TCR, t_last),0)
    train_TCR = train_TCR[1:,:]


    for i, e in enumerate(Epitope):
        e_encoded = [char_to_int[char] for char in e]
        e_encoded = torch.LongTensor(e_encoded).view(1,-1)
        e_all = torch.cat((e_encoded, expand),1)
        e_last = e_all[0,0:20].view(1,20)
        train_Epi = torch.cat((train_Epi,e_last),0)

    train_Epi = train_Epi[1:,:]
    

    train_Label = torch.LongTensor(Label)
    

    return train_TCR, train_Epi, train_Label


# BLOSUM62 Encoding
def BLOSUM_62(TCR, Epitope, Label, d_model):

    blosum62 = substitution_matrices.load('BLOSUM62')
    Label = torch.LongTensor(Label).view(-1,1)
    n = Label.size()[0]
    ext = list('********************')         
    tcr_embedding = torch.zeros(n,d_model,d_model)
    epi_embedding = torch.zeros(n,d_model,d_model)

    for ti, tcr in enumerate(TCR):
        tcr = list(tcr)
        tcr = tcr + ext
        tcr = tcr[0:d_model]
        # print(ti)
        #print(tcr)
        for i in range(d_model):
            for j in range(d_model):
                tcr_pair = (tcr[i],tcr[j])
                
                if tcr_pair not in blosum62:
                    tcr_embedding[ti,i,j] = blosum62[(tuple(reversed(tcr_pair)))]
                else:
                    tcr_embedding[ti,i,j] = blosum62[tcr_pair]
                tcr_embedding[ti,j,i] = tcr_embedding[ti,i,j]
        

        # print(tcr_embedding)
        
    for ei, epi in enumerate(Epitope):
        if ei == 0:
            epi = list(epi)
            epi = epi + ext
            epi = epi[0:d_model] 
            s = 0
            for i in range(d_model):
                for j in range(d_model):
                    epi_pair = (epi[i],epi[j])
                    #print('i:',i,'j',j)
                    if epi_pair not in blosum62:
                        epi_embedding[ei,i,j] = blosum62[(tuple(reversed(epi_pair)))]

                    else:
                        epi_embedding[ei,i,j] = blosum62[epi_pair]
                    epi_embedding[ei,j,i] = epi_embedding[ei,i,j]
        
        epi_embedding[ei] = epi_embedding[0]

        # print(tcr_embedding)
                #print(epi_embedding[ei,i,j])
    return tcr_embedding, epi_embedding, Label
    


# Atchley
def Atchley(TCR, Epitope, Label, Length):
    # 构建存储tcr与epi编码的tensor数组
    aa_vec = pk.load(open('atchley.pk', 'rb'))
    Label = torch.LongTensor(Label).view(-1, 1)
    n = Label.size()[0]
    ext = list('********************')  # 用于扩增tcr与epi的长度
    tcr_embedding = torch.zeros(n, Length, 6)
    epi_embedding = torch.zeros(n, Length, 6)
    # 计算在这里面计算！
    for ti, tcr in enumerate(TCR):
        tcr = tcr + ' ' * (Length - len(tcr))
        for i in range(Length):
            tcr_embedding[ti, i, :] = torch.from_numpy(aa_vec[tcr[i]])

    for ei, epi in enumerate(Epitope):
        epi = epi + ' ' * (Length - len(epi))
        for i in range(Length):
            epi_embedding[ti, i, :] = torch.from_numpy(aa_vec[epi[i]])

    print("该数据集的总个数:" + str(ei))
    return tcr_embedding[:, :, 0:5], epi_embedding[:, :, 0:5], Label

# Define new dataset
class MyDataset(Dataset):
    def __init__(self, path, emb_type):

        self.data = pd.read_csv(path, names = ['TCR','Epitope','Label'])
        self.TCR = self.data['TCR']
        self.Epitope = self.data['Epitope']
        self.Label = self.data['Label']
        self.emb_type = emb_type
        if self.emb_type == 'onehot':
            TCR, Epi, self.Label = onehot(self.TCR, self.Epitope, self.Label)
        elif self.emb_type == 'BLOSUM62':
            TCR, Epi, self.Label = BLOSUM_62(self.TCR, self.Epitope, self.Label, 20)
        else:
            TCR, Epi, self.Label = Atchley(self.TCR, self.Epitope, self.Label, 20)
        self.pair = torch.cat((TCR, Epi), -1)
        #self.Label = torch.LongTensor(self.data['Label'])
        # self.pair = torch.cat((self.TCR, self.Epitope),1)

    def __getitem__(self, index):
        return self.pair[index], self.Label[index]
    
    def __len__(self):
        
        return torch.LongTensor(self.Label).size()[0]

def AtchleyTriple(TCR1,TCR2, Epitope, Label, Length):
    # 构建存储tcr与epi编码的tensor数组
    aa_vec = pk.load(open('atchley.pk', 'rb'))
    Label = torch.LongTensor(Label).view(-1, 1)
    n = Label.size()[0]
    ext = list('********************')  # 用于扩增tcr与epi的长度
    tcr1_embedding = torch.zeros(n, Length, 6)
    tcr2_embedding = torch.zeros(n, Length, 6)
    epi_embedding = torch.zeros(n, Length, 6)
    # 计算在这里面计算！
    for ti, tcr in enumerate(TCR1):
        tcr = tcr + ' ' * (Length - len(tcr))
        for i in range(Length):
            tcr1_embedding[ti, i, :] = torch.from_numpy(aa_vec[tcr[i]])

    for ti, tcr in enumerate(TCR2):
        tcr = tcr + ' ' * (Length - len(tcr))
        for i in range(Length):
            tcr1_embedding[ti, i, :] = torch.from_numpy(aa_vec[tcr[i]])

    for ei, epi in enumerate(Epitope):
        epi = epi + ' ' * (Length - len(epi))
        for i in range(Length):
            epi_embedding[ti, i, :] = torch.from_numpy(aa_vec[epi[i]])

    print("该数据集的总个数:" + str(ei))
    return tcr1_embedding[:, :, 0:5],tcr2_embedding[:, :, 0:5], epi_embedding[:, :, 0:5], Label

class TripleDataset(Dataset):
    def __init__(self, path, emb_type):

        self.data = pd.read_csv(path, names=['TCR1','TCR2', 'Epitope', 'Label'])
        self.TCR1 = self.data['TCR1']
        self.TCR2 = self.data['TCR2']
        self.Epitope = self.data['Epitope']
        self.Label = self.data['Label']
        self.emb_type = emb_type
        T1, T2, E, self.Label = Atchley(self.TCR1, self.TCR2, self.Epitope, self.Label, 20)
        self.pair = torch.cat((T1, T2, E), -1)
        # self.Label = torch.LongTensor(self.data['Label'])
        # self.pair = torch.cat((self.TCR, self.Epitope),1)

    def __getitem__(self, index):
        return self.pair[index], self.Label[index]

    def __len__(self):

        return torch.LongTensor(self.Label).size()[0]

