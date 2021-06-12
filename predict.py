import torch
import numpy as np
from build_vocab import vocab_to_int 
from LSTM_NN import LSTM_NN
import pickle 
from pad_sequence import pad_features
import CONFIG 
from string import punctuation

def tokenize_review(test_senti):
    test_words = test_senti.split()
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])

    return test_ints


with open("./Intermediate Files/vocab_dictionary.pickle", "rb") as input_file:
    dictionary = pickle.load(input_file)

model = LSTM_NN(CONFIG.vocab_size, CONFIG.output_size,CONFIG.embedding_size ,CONFIG.hidden_size, CONFIG.n_layers, drop_prob=0.5)

checkpoint = torch.load('./Intermediate Files/saved_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
def predict(model, dictionary,test_review, sequence_length=47):
    
 model.eval()


 review_ints = tokenize_review(test_review)

 review_ints=pad_features(review_ints,sequence_length)

 review_tensor=torch.from_numpy(review_ints)

 batch_size = review_tensor.size(0)

 h = model.init_hidden(batch_size)

 output, h = model(review_tensor, h)

 pred = torch.round(output.squeeze())

 print('Prediction value, pre-rounding: {:.2f}'.format(output.item()))

 if(pred.item()>=0.5):
     print("Positive review detected!")
 else:
     print("Negative review detected.")


predict(model,dictionary,"लोगों ने ऑफर व डिस्काउंट का भरपूर फायदा उठाया।")
