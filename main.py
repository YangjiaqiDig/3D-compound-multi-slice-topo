from model import U_Net
from dataset import *
import torch
import numpy as np
import sys
import torch.nn as nn
from modules import *
# from save_history import *

# np.set_printoptions(threshold=sys.maxsize)
def toSlicesGroupDataset(train, label, size):
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

def multiModel( single=1, size_1=0, size_2=0,):
    model_single = U_Net(in_channels=single, out_channels=32)
    if(size_1):
        model_1 = U_Net(in_channels=size_1, out_channels=32)
    if(size_2):
        model_2 = U_Net(in_channels=size_2, out_channels=32)

    return model_single, model_1, model_2
if __name__ == "__main__":
    SLICES = 3
    trainData = CREMIDataTrain('train/train-volume.tif', 'train/train-labels.tif')
    # test = CREMIDataTest('train/test')
    train, label = trainData.__getitem__()
    train_data = toSlicesGroupDataset(train, label, SLICES)

    train_load = torch.utils.data.DataLoader(dataset=train_data, num_workers=3, batch_size=2, shuffle=True)
    # test_load = torch.utils.data.DataLoader(dataset=test, num_workers=3, batch_size=2, shuffle=False)
    model, model_1, _ = multiModel(single=1, size_1=SLICES, size_2=0)


    # model = torch.nn.DataParallel(model, device_ids=list(
    #     range(torch.cuda.device_count()))).cuda()

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    optimizer_1 = torch.optim.RMSprop(model_1.parameters(), lr=0.001)

    header = ['epoch', 'train loss', 'train acc']
    save_file_name = "history/RMS/history_RMS3.csv"
    save_dir = "history/RMS"

    # Saving images and models directories
    model_save_dir = "history/RMS/saved_models3"
    image_save_path = "history/RMS/result_images3"

    epoch = 30 # 2000
    # Train
    print("Initializing Training!")
    for i in range(0, epoch):
        # train the model
        train_model(model, train_load, loss_fun, optimizer)
        # just for print loss
        train_acc, train_loss = get_loss_train(model, train_load, loss_fun)

        # train_loss = train_loss / len(SEM_train)
        print('Epoch', str(i + 1), 'Train loss:', train_loss, "Train acc", train_acc)

        # Validation every 5 epoch
        # if (i + 1) % 5 == 0:
        #     val_acc, val_loss = validate_model(
        #         model, SEM_val_load, criterion, i + 1, True, image_save_path)
        #     print('Val loss:', val_loss, "val acc:", val_acc)
        #     values = [i + 1, train_loss, train_acc, val_loss, val_acc]
        #     export_history(header, values, save_dir, save_file_name)
        #
        #     if (i + 1) % 100 == 0:  # save model every 10 epoch
        #         save_models(model, model_save_dir, i + 1)

"""
# Test
print("generate test prediction")
test_model("../history/RMS/saved_models/model_epoch_440.pwf",
           test_load, 440, "../history/RMS/result_images_test")
"""