import numpy as np
import h5py
import configparser
import torch
from torch.utils.data import Dataset
from itertools import product


def load_eeg_data(data_path: str, config_path: str):
    """Loads a matlab matrix and expects the four arrays 'mean_MRCP_XX_precision5', with XX \in {LC, LS, SC, SS}.
    Will split the data into different phases, Regions of Interest, grasps and subjects.

    :return: Two matrices, representing the input and output for an RNN. The data matrix is of shape ()
    """
    DATA_FILE = h5py.File(data_path)
    # Each data below has the shape (1200ms x 64 electrodes x 16 subject)
    LC_DATA = np.array(DATA_FILE.get("mean_MRCP_LC_precision5"), order="F")
    LS_DATA = np.array(DATA_FILE.get("mean_MRCP_LS_precision5"), order="F")
    SC_DATA = np.array(DATA_FILE.get("mean_MRCP_SC_precision5"), order="F")
    SS_DATA = np.array(DATA_FILE.get("mean_MRCP_SS_precision5"), order="F")

    # Resulting array has the shape (1200ms x 64 electrodes x 16 subject x 4 grasps)
    ALL_DATA = np.stack([LC_DATA, LS_DATA, SC_DATA, SS_DATA], axis=3)

    # SPLIT SIGNALS INTO EACH PART AS NECESSARY:
    #   - 4 Phases
    #       - The intervals are: [0:199]; [200:599]; [600:999]; [1000:1200]
    #   - 4 ROIs:              FC1, FCz, FC2;  C3, CZ, C4;    CP3, CP1, CPz, CP2, CP4;   P3, P1, Pz, P2, P4
    #       - Channels (0-63): [8, 45, 9];     [12, 13, 14];  [51, 18, 52, 19, 53];      [23, 56, 24, 57, 25]
    #   - 16 subjects
    #       - Each channel is its own subject
    #   - 4 Grasps:
    #       - Each *_DATA is its own grasp

    PHASE_it = [i for i in range(1200)]
    ROI_it = [8, 45, 9, 12, 13, 14, 51, 18, 52, 19, 53, 23, 56, 24, 57, 25]
    SUBJECT_it = [i for i in range(16)]
    GRASP_it = [i for i in range(4)]

    config = configparser.ConfigParser()
    config["DATA"] = {
        "PHASE_it_len": len(PHASE_it),
        "ROI_it_len": len(ROI_it),
        "SUBJECT_it_len": len(SUBJECT_it),
        "GRASP_it_len": len(GRASP_it),
    }
    config["CHANNELS"] = {"ALL": ROI_it}

    with open(config_path, "w") as configfile:
        config.write(configfile)

    ALL_DATA = ALL_DATA.transpose(2, 3, 1, 0)
    return ALL_DATA[:, :, ROI_it, :]


def split_data(data: np.array, config_path: str, type: str):
    """Split data into training, testing and validation.
    Splits along the time dimension.

    :param data: Numpy array containing all inputs
    :return: Tuple[np.array, np.array, np.array] tuples are training, validation and testing data respectively.
    """

    trainX = data[0:10,]
    valX = data[10:13,]
    testX = data[13:16,]

    config = configparser.ConfigParser()
    config.read(config_path)
    config[type] = {
        "train": trainX.shape[1],
        "val": valX.shape[1],
        "test": testX.shape[1],
    }

    with open(config_path, "w") as configfile:
        config.write(configfile)

    return trainX, valX, testX


class RDMDataset(Dataset):
    def __init__(self, eeg_array, kinematics_array):
        self.labels = self.create_rdms(kinematics_array)
        self.data = torch.as_tensor(eeg_array, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @staticmethod
    def create_rdms(data):
        """Creates Representation Dissimilarity Maps (RDMs) based off of the passed data.
        Data is expected to be of shape (sample_size, condtions, time-points)
        """
        labels = torch.empty(0)

        for datapoint in data:
            cur_rdm = 1 - torch.corrcoef(torch.as_tensor(datapoint))
            cur_rdm = cur_rdm.reshape(1, data.shape[1], data.shape[1])
            labels = torch.cat((labels, cur_rdm), 0)

        return labels


def get_data(eeg_data_path: str, kinematics_data_path: str, config_path: str):
    full_eeg_data = load_eeg_data(eeg_data_path, config_path)
    train_eeg_data, val_eeg_data, test_eeg_data = split_data(full_eeg_data, config_path, "eeg")
    full_kinematics_data = np.tile(np.array([[1, 2, 3, 4],
                                             [1, 2, 3, 4],
                                             [2, 3, 4, 5],
                                             [0, 1, 2, 3]], dtype="float32"),
                                   (16, 1, 1))
    train_kin, val_kin, test_kin = split_data(full_kinematics_data, config_path, "kin")
    return RDMDataset(train_eeg_data, train_kin), RDMDataset(val_eeg_data, val_kin), RDMDataset(test_eeg_data, test_kin)
