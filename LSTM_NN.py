import torch.nn as nn
import torch 

class LSTM_NN(nn.Module):
    
    def __init__(self,vocab_size, output_size,embedding_size, hidden_size, n_layers,drop_prob=0.5):
        super(LSTM_NN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.l1 = nn.Linear(hidden_size, output_size)
        self.prob = nn.Sigmoid()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_size).zero_())  
        return hidden

    def forward(self,x, hidden):
        x = x.long()
        embeds = self.embed(x)
        lstm_out, (h_n,cell) = self.lstm(embeds, hidden)
        lstm_out=lstm_out[:,-1,:]
        
        out = self.dropout(lstm_out)
        out_l1 = self.l1(out)
        sig_out = self.prob(out_l1)        
        
        return sig_out, hidden