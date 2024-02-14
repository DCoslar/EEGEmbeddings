import torch
import torch.nn as nn
import numpy as np


class RDM_MLP(nn.Module):
    """Conditions need to be dim 1, number of channels dim 2, data/time points dim 3
    Example array with 3 conditions, 2 channels and 4 points per channel:
    np.array(
        [
            [[1, 2, 3, 4], [1, 2, 3, 4]],
            [[2, 3, 4, 5], [2, 3, 4, 5]],
            [[0, 1, 2, 3], [0, 1, 2, 3]]
        ],
        dtype="float32")
    """
    def __init__(self, in_dim: int):
        super().__init__()

        self.in_dim = in_dim

        self.process = nn.Sequential(
            nn.Linear(self.in_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward_simple(self, data):
        """Compute a single pass through the network of a multivariate time-series.
        The entire modified time-series is returned.

        :param data: A tensor of shape (num_channels, len_per_channel)
        :return: network(data) of the same shape
        """
        outputs = torch.empty(0)

        for data_point in data.T:
            data_point_reshaped = data_point.reshape(1, self.in_dim)
            outputs = torch.cat((outputs, self.process(data_point_reshaped)), 1)

        return outputs

    def forward(self, data):
        """Compute an RDM based on 1 - R^2 of the passed signals.

        :param data: A tensor of shape (classes/conditions, inputs, time)
        :return: 1 - Corr(network(class_1), network(class_2), ..., network(class_N))
        """
        outputs = torch.empty(0)
        for batch in data:
            cur_output = torch.empty(0)
            for signal in batch:
                cur_output = torch.cat((cur_output, self.forward_simple(signal)), 0)

            corr = torch.corrcoef(cur_output)
            ones = torch.ones_like(corr)

            cur_rdm = ones - corr
            cur_rdm = cur_rdm.reshape(1, data.shape[1], data.shape[1])

            outputs = torch.cat((outputs, cur_rdm), 0)

        return outputs



# for test
if __name__ == '__main__':
    net = RDM_MLP(2)
    net.eval()
    input = torch.as_tensor(np.array([[[1, 2, 3, 4], [1, 2, 3, 4]], [[2, 3, 4, 5], [2, 3, 4, 5]], [[0, 1, 2, 3], [0, 1, 2, 3]]], dtype="float32"))
    # input = input.reshape(2, 4)
    print(f"{input.shape=}")
    # print(torch.corrcoef(input))
    print(net(input))
