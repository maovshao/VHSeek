import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class vhseek_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(vhseek_model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 960)
        self.linear2 = nn.Linear(960, 480)
        self.linear3 = nn.Linear(480, 240)
        self.classifier = nn.Linear(240, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(F.relu(self.linear2(x)))
        x = self.dropout(F.relu(self.linear3(x)))
        embeddings = x  # Embeddings before the classifier
        logits = self.classifier(x)
        return logits, embeddings  # Return both logits and embeddings

class load_dataset(Dataset):
    def __init__(self, esm_embedding, label_index, host_dict_path = None):
        # Read key_host_dict
        key_hosts = {}
        self.host_dict_path = host_dict_path
        self.label_index = label_index
        if self.host_dict_path:
            with open(self.host_dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # First, split the line by tab to separate virus_name and hosts
                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue  # Skip lines that do not have enough parts
                    key_name = parts[0]
                    if key_name in esm_embedding:
                        key_hosts[key_name] = parts[1:]
                    else:
                        assert("Difference between label and embedding")
        else:
            for key_name in esm_embedding:
                key_hosts[key_name] = None


        self.num_hosts = len(self.label_index)

        # Build x, y, and virus_names
        self.embeddings = []
        self.key_names = []
        self.hosts = []

        for key_name in key_hosts:
            self.embeddings.append(esm_embedding[key_name])
            self.key_names.append(key_name)
            if self.host_dict_path:
                y_vector = torch.zeros(self.num_hosts)
                for host in key_hosts[key_name]:
                    if host in self.label_index:
                        idx = self.label_index[host]
                        y_vector[idx] = 1
                self.hosts.append(y_vector)
            else:
                self.hosts.append(torch.zeros(self.num_hosts))

        self.embeddings = torch.stack(self.embeddings)
        self.hosts = torch.stack(self.hosts)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.key_names[idx], self.hosts[idx]

    def get_dim(self):
        return int(self.embeddings[0].shape[0])

    def get_class_num(self):
        return self.num_hosts