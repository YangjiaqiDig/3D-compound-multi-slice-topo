from model import U_Net
from dataset import *
from modules import *
from utils import *
from save_history import *
import torch
import numpy as np
import torch.nn as nn


# np.set_printoptions(threshold=sys.maxsize)

def multiModel(SLICES_COLLECT):
    models = []
    for SLICE in SLICES_COLLECT:
        models.append(U_Net(in_channels=SLICE, out_channels=32))

    return models


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SLICES_COLLECT = [1]
    dataset_path = 'train'
    dataset_cache = 'dataset_cache'
    trainDataset, validDataset = get_dataset(dataset_path, dataset_cache, SLICES_COLLECT)

    train_load = torch.utils.data.DataLoader(dataset=trainDataset, num_workers=6, batch_size=2, shuffle=True)
    val_load = torch.utils.data.DataLoader(dataset=validDataset, num_workers=6, batch_size=1, shuffle=False)

    models = multiModel(SLICES_COLLECT)
    if device.type == "cuda":
        print("GPU: ", torch.cuda.device_count())
        for i in range(len(models)):
            models[i] = torch.nn.DataParallel(models[i], device_ids=list(
                range(torch.cuda.device_count()))).cuda()

    loss_fun = nn.CrossEntropyLoss()
    optimizers = []
    for i in range(len(models)):
        optimizers.append(torch.optim.RMSprop(models[i].parameters(), lr=0.0001))

    header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
    save_file_name = "history/UNET/history_UNET.csv"
    save_dir = "history/UNET"

    # Saving images and models directories
    model_save_dir = "history/UNET/saved_models3"
    image_save_path = "history/UNET/result_images3"

    epoch = 30  # 2000
    # Train
    print("Initializing Training!")
    for i in range(0, epoch):
        train_multi_models(models,
                           train_load,
                           loss_fun,
                           optimizers,
                           device,
                           SLICES_COLLECT)

        # just for print loss
        train_acc, train_loss = get_loss_train(models,
                                               train_load,
                                               loss_fun,
                                               device,
                                               SLICES_COLLECT)

        print('Epoch', str(i + 1), 'Train loss:', train_loss, "Train acc", train_acc)

        # Validation every 5 epoch
        if (i + 1) % 5 == 0:
            val_acc, val_loss = validate_model(
                models,
                val_load,
                loss_fun,
                i + 1,
                True,
                image_save_path,
                device,
                SLICES_COLLECT)
            print('Val loss:', val_loss, "val acc:", val_acc)

            values = [i + 1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name)

            if (i + 1) % 10 == 0:  # save model every 10 epoch
                save_models(models, model_save_dir, i + 1, SLICES_COLLECT)

"""
# Test
print("generate test prediction")
test_model("../history/RMS/saved_models/model_epoch_440.pwf",
           test_load, 440, "../history/RMS/result_images_test")
"""
