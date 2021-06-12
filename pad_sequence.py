import numpy as np
def pad_features(sentence_ints, seq_length):
  features = np.zeros((len(sentence_ints), seq_length), dtype=int)
  #print(features.shape)
  for i, row in enumerate(sentence_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]                               #
  return features