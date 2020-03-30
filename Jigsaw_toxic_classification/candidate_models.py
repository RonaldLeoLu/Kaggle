import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata

# Spatial Dropout
class SpatialDropout(nn.Dropout2d):
    
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

# baseline model
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, output_dim):
        super(NeuralNet, self).__init__()
        self.output_dim = output_dim
        max_features, embed_size = embedding_matrix.shape
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
    
        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        
        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        if self.output_dim != 1:
            self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, output_dim)
        
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        out = self.linear_out(hidden)
        
        if self.output_dim != 1:
            aux_result = self.linear_aux_out(hidden)
            out = torch.cat([out, aux_result], 1)
        
        return out


#################################
#           TextRCNN
#################################
'''
embedding layer --> ---------------------> Concat -> Pooling -> Conv -> Linear
                    -> BiLSTM -> BiLSTM ->
'''
LSTM_UNITS = 128
CONV_SIZE = 128
KERNEL_SIZE = 3
class TextRCNN(nn.Module):
    def __init__(self, embedding_matrix, output_dim):
        super(TextRCNN, self).__init__()
        self.output_dim = output_dim
        max_features, embed_size = embedding_matrix.shape
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.concat_layers = nn.Sequential(nn.Conv1d(LSTM_UNITS*2+embed_size, CONV_SIZE, KERNEL_SIZE),
                                           nn.BatchNorm1d(CONV_SIZE),
                                           nn.ReLU(),
                                           nn.Conv1d(CONV_SIZE, CONV_SIZE, KERNEL_SIZE),
                                           nn.BatchNorm1d(CONV_SIZE),
                                           nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(CONV_SIZE*2, CONV_SIZE),
                                nn.ReLU(),
                                nn.Dropout(0.3))
        self.out = nn.Linear(CONV_SIZE, 1)
        self.aux = nn.Linear(CONV_SIZE, self.output_dim)

    def forward(self, x):
        embs = self.embedding(x)

        h_lstm1, _ = self.lstm1(embs)
        h_lstm2, _ = self.lstm2(h_lstm1)

        ct = torch.cat([embs, h_lstm2], -1)

        conv_out = self.concat_layers(ct.permute(0,2,1))

        avg_pool = torch.mean(conv_out, -1)
        max_pool, _ = torch.max(conv_out, -1)

        pool_out = torch.cat([avg_pool, max_pool],-1)

        fc_out = self.fc(pool_out)

        out = self.out(fc_out)
        aux = self.aux(fc_out)

        return torch.cat([out, aux],-1)



WINDOW_SIZES = [3, 4, 5]
CONV_SIZE = 64
HIDDEN_SIZE = 32
class TextCNN(nn.Module):
    def __init__(self, emb_mat, output_dim):
        super(TextCNN, self).__init__()
        class_num = output_dim
        vs, es = emb_mat.shape
        
        self.embedding = nn.Embedding(vs, es)
        self.embedding.weight = nn.Parameter(torch.tensor(emb_mat, dtype=torch.float32))
        
        self.emb_dropout = SpatialDropout(0.2)
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(es, CONV_SIZE, kernel_size),
                nn.BatchNorm1d(CONV_SIZE),
                nn.ReLU(),
                nn.Conv1d(CONV_SIZE, CONV_SIZE, kernel_size),
                nn.BatchNorm1d(CONV_SIZE),
                nn.ReLU(),
                #nn.MaxPool1d(kernel_size)
            )
            for kernel_size in WINDOW_SIZES
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(CONV_SIZE*len(WINDOW_SIZES), HIDDEN_SIZE),
            nn.ReLU()
        )
        
        self.out = nn.Linear(HIDDEN_SIZE, class_num)
        
    def custom_max_pool(self, x):
        m, _ = torch.max(x,-1)
        return m
        
    def forward(self, x):
        embs = self.embedding(x)
        embs = self.emb_dropout(embs)
        
        x = embs.permute(0,2,1)
        x = [self.custom_max_pool(conv(x)) for conv in self.convs]
        x = torch.cat(x,1)
        
        x = self.fc(x)
        out = self.out(x)
        
        return out