#%% Test envrionment for various DNN approaches such as end2you and CGDNN.

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from tqdm import tqdm
import pandas as pd
import numpy as np

#%%
class CGDNN(nn.Module):
    def __init__(self, input_shape, batch_size=1):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,(7,7), padding=(3,3))
        self.maxpool1 = nn.MaxPool2d((6,6), padding=(3,3))
        self.conv2 = nn.Conv2d(16,32,(5,5), padding=(2,2))
        self.maxpool2 = nn.MaxPool2d((2,1), padding=(1,0))
        self.conv3 = nn.Conv2d(32,32,(3,3), padding=(1,1))
        self.maxpool3 = nn.MaxPool2d((2,1), padding=(1,0))
        self.conv4 = nn.Conv2d(32,32,(3,3), padding=(1,1))
        self.maxpool4 = nn.MaxPool2d((2,1), padding=(0,0))
        self.input_shape = input_shape
        self.linear1 = nn.Linear(81 ,128)
        self.flatten = nn.Flatten(1)
        self.gru = nn.GRU(128,256,num_layers=2,batch_first=True)
        self.linear2 = nn.Linear(8192 ,32)
        self.linear3 = nn.Linear(32 ,11)
        self.softmax = nn.Softmax(1)
    def forward(self, x):
        # input signal is 1 x t x k where k = 40

        x = self.conv1(x)

        x = F.relu(x)
        x = self.maxpool1(x)


        x = self.conv2(x)
        x = F.relu(x)

        x = self.maxpool2(x)


        x = self.conv3(x)

        x = F.relu(x)
        x = self.maxpool3(x)


        x = self.conv4(x)

        x = F.relu(x)
        x = self.maxpool4(x)

        x = self.linear1(x)

        x = x.squeeze(2)

        x, h_gru = self.gru(x)


        x = self.flatten(x)

        x = self.linear2(x)

        x = self.linear3(x)

        return self.softmax(x)

class EMO16(nn.Module):
    def __init__(self, input_shape, batch_size=16, num_cats=11):
        super().__init__()


        #self.conv1 = nn.Conv1d(1, 64, 40, stride=1, padding=20)
        #self.conv2 = nn.Conv1d(64, 128, 20, stride=1, padding=10)
        #self.conv3 = nn.Conv1d(128, 256, 10, stride=1, padding=5)
        self.conv1 = nn.Conv1d(1,40,20,stride=1,padding=10)
        self.maxpool1 = nn.MaxPool1d(2,2,padding=1)
        self.conv2 = nn.Conv1d(40,40,40,stride=1,padding=20)
        self.maxpool2 = nn.MaxPool2d((10,1),(10,1),padding=(5,0))



        #self.maxpool1 = nn.MaxPool1d(8,4, padding=4)
        #self.maxpool2 = nn.MaxPool1d(8,4, padding=4)
        #self.maxpool3 = nn.MaxPool1d(20,10, padding=10)

        #self.maxpool2d = nn.MaxPool2d((64,1),(64,1))

        self.lstm = nn.LSTM(240000,128,batch_first=True)

        #self.linear = nn.Linear(1000*4,11)

        #self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten(1)
    def forward(self, x):

        print(x.size())
        x = self.conv1(x)
        print(x.size())
        #x = F.relu(x)
        x = self.maxpool1(x)
        print(x.size())
        x = self.conv2(x)
        print(x.size())
        #x = F.relu(x)
        x = self.maxpool2(x)
        print(x.size())

        print(x.size())    
        x,_ = self.lstm(x)
        print(x.size()) 
        #x = self.conv3(x)
        #x = F.relu(x)
        #x = self.maxpool3(x)
        #x = self.maxpool2d(x)


        #x_t,_= self.lstm(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x
#%%
class DataStore():
    def __init__(self, df, dataset_path, file_col="file_name", label_col="L1", parts=3):
        self.folder = dataset_path + "wav/"
        self.df = df
        self.data = []
        self.labels = []
        for ind, row in tqdm(df.iterrows(), total=df.shape[0]):
            fname = row[file_col]
            label = row[label_col]
            waveform,sample_rate = torchaudio.load(self.folder + fname,offset=16000,num_frames=int(16000*16))
            #if using CGDNN
            mfccs = torchaudio.transforms.MFCC(sample_rate,n_mfcc=80)(waveform)[:,0:40,:]
            self.data.append(mfccs)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# %% 
def setlr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def lr_decay(optimizer, epoch):
    if epoch%1==0:
        new_lr = learning_rate / (10**(epoch//20))
        optimizer = setlr(optimizer, new_lr)
        print(f'Changed learning rate to {new_lr}')
    return optimizer

def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, change_lr=None):
    for epoch in range(1,epochs+1):
        model.train()
        batch_losses=[]
        if change_lr:
            optimizer = change_lr(optimizer, epoch)
        for i, data in tqdm(enumerate(train_loader),total=len(train_loader)):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            y_hat = model(x)
            
            loss = loss_fn(y_hat, y)

            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        train_losses.append(batch_losses)
        print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
        model.eval()
        batch_losses=[]
        trace_y = []
        trace_yhat = []
        for i, data in tqdm(enumerate(valid_loader),total=len(valid_loader)):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())      
            batch_losses.append(loss.item())
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
        print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1]):.3f} Valid-Accuracy : {accuracy:.3f}')
#%%
dataset_path = "------"  
train_and_dev_file =  "lab/ComParE2016_Nativeness.tsv"
print("preparing metadata")
meta = pd.read_csv(dataset_path + train_and_dev_file, sep="\t")

label_encoder = preprocessing.LabelEncoder()
meta.L1 = label_encoder.fit_transform(meta.L1)


print("storing MFCCs to datastores")
meta_train = meta[meta['file_name'].str.contains('train')]
meta_dev = meta[meta['file_name'].str.contains('devel')]

train_ds = DataStore(meta_train,dataset_path)
devel_ds = DataStore(meta_dev,dataset_path)
print("data prepared!")


# %%
print(train_ds.data[1].size())
# %% 

batch_size = 2
loss_fn = nn.NLLLoss()
learning_rate = 2e-5
epochs = 20

if torch.cuda.is_available():
    device=torch.device('cuda:0')
else:
    device=torch.device('cpu')

model = CGDNN(train_ds.data[0].shape,batch_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)



#%%
train_losses = []
valid_losses = []

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(devel_ds, batch_size=batch_size, shuffle=True)
# %%
train(model, loss_fn, valid_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, lr_decay)


# %%
