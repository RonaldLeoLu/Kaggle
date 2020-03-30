from candidate_models import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata


def train_model(model, train_data, batch_size=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    train_loader = tdata.DataLoader(train_data, batch_size=batch_size)
    loss_fn = nn.BCEWithLogitsLoss()

    for x_batch, y_batch in train_loader:
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)

        loss.backward()

        optimizer.zero_grad()
        optimizer.step()

    print('Sucessfully run one epoch!')
    print('Model Test Passed!')

def generate_embedding_matrix(a, b=15):
    return np.random.randn(a, b)

def generate_train(vocab_size):
    x = torch.from_numpy(np.random.randint(0, vocab_size, (1000, 30))).long()
    aux = torch.from_numpy(np.random.randn(1000, 7)).float()

    return tdata.TensorDataset(x,aux)

if __name__ == '__main__':
    v_s = 40
    class_model = TextCNN

    emb_mat = generate_embedding_matrix(v_s)
    train = generate_train(v_s)

    model = class_model(emb_mat, output_dim=7)

    train_model(model, train)


