from model import U_Net
from dataset import *
import torch
import torch.nn as nn
from modules import *
# from save_history import *



if __name__ == "__main__":
    train = CREMIDataTrain('train/train-volume.tif', 'train/train-labels.tif')
    # test = CREMIDataTest('train/test')

    train_load = torch.utils.data.DataLoader(dataset=train, num_workers=3, batch_size=2, shuffle=True)
    # test_load = torch.utils.data.DataLoader(dataset=test, num_workers=3, batch_size=2, shuffle=False)

    model = U_Net(in_channels=1, out_channels=32)
    # model = torch.nn.DataParallel(model, device_ids=list(
    #     range(torch.cuda.device_count()))).cuda()

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

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