from model import U_Net
from dataset import *
from modules import *
from utils import *
from save_history import *
import torch
import numpy as np
import torch.nn as nn

# np.set_printoptions(threshold=sys.maxsize)

def multiModel(single=1, size_1=0, size_2=0):
    model_single = U_Net(in_channels=single, out_channels=32)
    if size_1:
        model_1 = U_Net(in_channels=size_1, out_channels=32)
    if size_2:
        model_2 = U_Net(in_channels=size_2, out_channels=32)

    return model_single, model_1, model_2


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SLICES_1 = 3
    SLICES_2 = 5
    dataset_path = 'train'
    dataset_cache = 'dataset_cache'
    trainDataset, validDataset = get_dataset(dataset_path, dataset_cache, SLICES_1, SLICES_2)

    train_load = torch.utils.data.DataLoader(dataset=trainDataset, num_workers=3, batch_size=2, shuffle=True)
    val_load = torch.utils.data.DataLoader(dataset=validDataset, num_workers=3, batch_size=1, shuffle=False)

    model_1, model_2, model_3 = multiModel(single=1, size_1=SLICES_1, size_2=SLICES_2)
    if device.type == "cuda":
        print("GPU: ", torch.cuda.device_count())
        model_1 = torch.nn.DataParallel(model_1, device_ids=list(
            range(torch.cuda.device_count()))).cuda()
        model_2 = torch.nn.DataParallel(model_2, device_ids=list(
            range(torch.cuda.device_count()))).cuda()
        model_3 = torch.nn.DataParallel(model_3, device_ids=list(
            range(torch.cuda.device_count()))).cuda()

    loss_fun = nn.CrossEntropyLoss()
    optimizer_1 = torch.optim.RMSprop(model_1.parameters(), lr=0.0001)
    optimizer_2 = torch.optim.RMSprop(model_2.parameters(), lr=0.0001)
    optimizer_3 = torch.optim.RMSprop(model_3.parameters(), lr=0.0001)

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
        train_multi_models([model_1, model_2, model_3],
                           train_load,
                           loss_fun,
                           [optimizer_1, optimizer_2, optimizer_3],
                           device)

        # just for print loss
        train_acc, train_loss = get_loss_train([model_1, model_2, model_3],
                                               train_load,
                                               loss_fun, device)

        print('Epoch', str(i + 1), 'Train loss:', train_loss, "Train acc", train_acc)

        # Validation every 5 epoch
        if (i + 1) % 5 == 0:
            val_acc, val_loss = validate_model(
                [model_1, model_2, model_3],
                val_load,
                loss_fun,
                i + 1,
                True,
                image_save_path,
                device)
            print('Val loss:', val_loss, "val acc:", val_acc)

            values = [i + 1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name)

            if (i + 1) % 10 == 0:  # save model every 10 epoch
                save_models([model_1, model_2, model_3], model_save_dir, i + 1)

"""
# Test
print("generate test prediction")
test_model("../history/RMS/saved_models/model_epoch_440.pwf",
           test_load, 440, "../history/RMS/result_images_test")
"""
