from networks.rdm_network import RDM_MLP
from training.data import get_data
from training.loops import train_loop, test_loop
from training.loss import fro_loss
from torch.utils.data import DataLoader
import torch

torch.autograd.set_detect_anomaly(True)

config_path = "./training/config.txt"
eeg_data_path = "./training/MRCP_data_av_conditions.mat"
train_data, val_data, test_data = get_data(eeg_data_path, "", config_path)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
val_loader = DataLoader(val_data, batch_size=3, shuffle=True)
test_loader = DataLoader(test_data, batch_size=3, shuffle=True)

network = RDM_MLP(16)

epochs = 10
learning_rate = 1e-5
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

for epoch in range(epochs):
    train_loop(train_loader, network, fro_loss, optimizer, 0.05)
    test_loop(val_loader, network, fro_loss)


print("tmp")
