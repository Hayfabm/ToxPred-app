import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name + ".npy", allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


"""Load preprocessed data."""
dir_input1 = "Data/rdkit/"
dir_input2 = "Data/embeddings/"
compounds = load_tensor(dir_input1 + "compounds", torch.LongTensor)
adjacencies = load_tensor(dir_input1 + "adjacencies", torch.FloatTensor)
embeddings = load_tensor(dir_input2 + "embeddings", torch.FloatTensor)


labels = load_tensor(dir_input1 + "labels", torch.LongTensor)
fingerprint_dict = load_pickle(dir_input1 + "fingerprint_dict.pickle")
n_fingerprint = len(fingerprint_dict)

"""Create a dataset and split it into train/dev/test."""
dataset = list(zip(compounds, adjacencies, embeddings, labels))
dataset = shuffle_dataset(dataset, 1234)
dataset_train, dataset_test = split_dataset(dataset, 0.85)


class AttentionPrediction(nn.Module):
    def __init__(self):
        super(AttentionPrediction, self).__init__()
        self.embed_fingerprint = torch.nn.Embedding(n_fingerprint, dim)
        self.W_gnn = torch.nn.ModuleList(
            [torch.nn.Linear(dim, dim) for _ in range(layer_gnn)]
        )

        self.final_h = nn.Parameter(torch.randn(1, 75 * 4 + dim))
        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(1, 1024),
            stride=(1, 1),
            padding=(0, 0),
        )

        self.rnn = torch.nn.GRU(
            input_size=64 * 1,
            hidden_size=75,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
        )

        self.W_s1 = nn.Linear(75 * 4 + dim, da)
        self.W_s2 = nn.Linear(da, 1)

        self.fc1 = torch.nn.Linear(75 * 4 + dim, units)

        self.fc2 = torch.nn.Linear(units, 2)

    def gnn(self, xs, A, layer):
        gnn_median = []
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
            temp = torch.mean(xs, 0)
            temp = temp.squeeze(0)
            temp = temp.unsqueeze(0)
            gnn_median.append(temp)
        return gnn_median

    def cnn(self, x):
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        x = self.conv1(x)
        x = torch.nn.ReLU()(x)
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(0)
        output, hidden = self.rnn(x)
        return torch.cat([hidden[-1], hidden[-2], hidden[-3], hidden[-4]], dim=1)

    def selfattention(self, cat_vector):

        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(cat_vector)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=1)

        return attn_weight_matrix

    def forward(self, gnn_peptide, gnn_adjacencies, cnn_embeddings):

        """Peptide vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(gnn_peptide)
        gnn_vectors = self.gnn(fingerprint_vectors, gnn_adjacencies, layer_gnn)
        self.feature1 = gnn_vectors

        """Peptide vector with CNN based on embedding."""
        cnn_vectors = self.cnn(cnn_embeddings)
        self.feature2 = cnn_vectors

        """Concatenate the above three vectors and output the prediction."""
        vector = []
        for i in range(layer_gnn):
            vector.append(torch.cat([gnn_vectors[i], cnn_vectors], dim=1))
        all_vector = vector[0]
        for i in range(1, layer_gnn):
            all_vector = torch.cat((all_vector, vector[i]), 0)

        all_vector = all_vector.unsqueeze(0)

        attn_weight_matrix = self.selfattention(all_vector)
        hidden_matrix = torch.bmm(attn_weight_matrix, all_vector)

        x = torch.nn.functional.relu(self.fc1(hidden_matrix.view(1, -1)))
        label = self.fc2(x)

        prediction = torch.nn.functional.softmax(label)
        return prediction


dim = 50
layer_gnn = 4
units = 840
da = 160

model = AttentionPrediction().to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

acc_list = []
train_loss = []
test_loss = []
min_test_acc = 0.5
for epoch in range(100):
    print("epochs:", epoch)
    total_loss = 0
    num = 0
    for i, (train_peptide, train_adjacencies, embeddings_train, y) in enumerate(
        dataset_train, 1
    ):
        train_peptide = train_peptide.to(device)
        train_adjacencies = train_adjacencies.to(device)
        embeddings_train = torch.reshape(embeddings_train, (1, 1024)).to(device)
        y = torch.Tensor([y]).long().to(device)
        output = model(train_peptide, train_adjacencies, embeddings_train)
        loss = loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 500 == 0:
            print(f"[ Epoch {epoch} ", end="")
            print(f"[{i}/{len(dataset_train)}] ", end="")
            print(f"loss={total_loss / i}")

    correct = 0
    total = len(dataset_test)
    print("evaluating trained model ...")
    y_pred = []
    with torch.no_grad():
        for test_peptide, test_adjacencies, embeddings_test, y in dataset_test:
            test_peptide = test_peptide.to(device)
            test_adjacencies = test_adjacencies.to(device)
            embeddings_test = torch.reshape(embeddings_test, (1, 1024)).to(device)
            y = torch.Tensor([y]).long().to(device)
            output = model(test_peptide, test_adjacencies, embeddings_test)
            #             print(output.detach().numpy())
            loss = loss_function(output, y)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(y).item()

        percent = "%.4f" % (100 * correct / total)
        print(f"Test set: Accuray {correct}/{total} {percent}%")
