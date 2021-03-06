from data_processing import train_df
from pad_sequence import pad_features
from torch.utils.data import TensorDataset, DataLoader
from LSTM_NN import LSTM_NN
import numpy as np
import CONFIG 
import torch 
import torch.nn as nn
import pickle


with open("./Intermediate Files/vocab_dictionary.pickle", "rb") as input_file:
    dictionary = pickle.load(input_file)

train_sentence=train_df['text'].tolist()


train_label=train_df['target'].values



def sentence_to_int(li):
    f_list=[]
    for sent in li:
        new_list=[]
        for word in sent:
            new_list.append(dictionary[word])
        t=torch.LongTensor(new_list)
        f_list.append(t)    
    return f_list

train_sentence_int=sentence_to_int(train_sentence)



seq_length=47
train_tensor=torch.from_numpy(pad_features(train_sentence_int,seq_length))


train_data = TensorDataset(train_tensor, torch.from_numpy(train_label))




train_loader = DataLoader(train_data, shuffle=True, batch_size=CONFIG.batch_size)



model=LSTM_NN(CONFIG.vocab_size, CONFIG.output_size,CONFIG.embedding_size ,CONFIG.hidden_size, CONFIG.n_layers, drop_prob=0.5)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam( model.parameters(), lr=CONFIG.learning_rate)


def train_loop(model,batch_size,optimizer,criterion):
    num_epochs = CONFIG.epochs
    clip = 5
    epoch_loss=[]
    model.train()
    for epoch in range(num_epochs):
        h = model.init_hidden(batch_size)
        train_loss=[]
        for i, (sent, label) in enumerate(train_loader):
            h = tuple([e.data for e in h])
            optimizer.zero_grad()
            #print(label.shape)
            output, h = model(sent, h)
            #print(output.shape)
            loss = criterion(output.squeeze(), label.float())
            train_loss.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()    
        epoch_loss.append(np.mean(train_loss))
        print('Epoch: {}  Loss: {}'.format(epoch,np.mean(train_loss)))

train_loop(model,CONFIG.batch_size,optimizer,criterion)


PATH='./Intermediate Files/saved_model.pt'
torch.save({
            
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion_state_dict':criterion.state_dict()
            }, PATH)


