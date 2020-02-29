import logging
import torch
import os
from dataset import *

logger = logging.getLogger(__file__)


def toSlicesGroupDataset(train, label, size):
    print('SLICES: ', size)
    if not size:
        return [[train[i].unsqueeze(0), label[i]] for i in range(label.shape[0])]

    paddingSize = int(size / 2)
    paddings = torch.zeros(paddingSize, train.shape[2], train.shape[2])
    train = torch.cat((paddings, train, paddings))

    new_train = []
    for i in range(0, train.shape[0] - size + 1):
        new_train.append(train[i:i + size])

    new_train = torch.stack(new_train)

    train_data = []
    for i in range(label.shape[0]):
        train_data.append([new_train[i], label[i]])

    return train_data


def prepareDataForLoader(data):
    train, label = data.__getitem__()

    train_data_1 = toSlicesGroupDataset(train, label, 0)
    train_data_2 = toSlicesGroupDataset(train, label, SLICES_1)
    train_data_3 = toSlicesGroupDataset(train, label, SLICES_2)

    dataset = ComDataset(train_data_1, train_data_2, train_data_3)
    """
    x1, x2, x3 = dataset.__getitem__(0)
    # print(len(x1), x1[0].shape, x1[1].shape)  # (2, (1, 1250, 1250), (1250, 1250))
    # print(len(x2), x2[0].shape, x2[1].shape)  # (2, (3, 1250, 1250), (1250, 1250))
    # print(len(x3), x3[0].shape, x3[1].shape)  # (2, (5, 1250, 1250), (1250, 1250))
    """

    return dataset


def get_dataset(dataset_path, dataset_cache, SLICES_1, SLICES_2):
    train_path = dataset_path + '/train-volume.tif'
    val_path = dataset_path + '/train-labels.tif'
    dataset_cache = dataset_cache + '_1' + str(SLICES_1) + str(
        SLICES_2)
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load enhanced dataset before DataLoader from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Start Prepare enhanced dataset before DataLoader %s", dataset_path)

        trainData = CREMIDataTrain(train_path, val_path)
        validData = CREMIDataVal(train_path, val_path)

        trainDataset = prepareDataForLoader(trainData)
        validDataset = prepareDataForLoader(validData)

        logger.info("list train and valid as the dataset")
        dataset = [trainDataset, validDataset]
        torch.save(dataset, dataset_cache)

    return dataset


if __name__ == "__main__":
    # A full forward pass
    SLICES_1 = 3
    SLICES_2 = 5
    dataset_cache = 'dataset_cache'
    trainDataset, validDataset = get_dataset('train', dataset_cache, SLICES_1, SLICES_2)

    x1, x2, x3 = validDataset.__getitem__(0)
    print(len(x1), x1[0].shape, x1[1].shape)  # (2, (1, 1250, 1250), (1250, 1250))
    print(len(x2), x2[0].shape, x2[1].shape)  # (2, (3, 1250, 1250), (1250, 1250))
    print(len(x3), x3[0].shape, x3[1].shape)  # (2, (5, 1250, 1250), (1250, 1250))
