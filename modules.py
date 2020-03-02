import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
import os
from post_processing import *


def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)

    return accuracy / len(np_ims[0].flatten())


def accuracy_check_for_batch(masks, predictions, batch_size):
    total_acc = 0
    for index in range(batch_size):
        total_acc += accuracy_check(masks[index], predictions[index])
    return total_acc / batch_size


def train_multi_models(models, data_train, loss_fun, optimizers, device, SLICES_COLLECT):
    for model in models:
        model.train()
    if len(SLICES_COLLECT) == 3:
        for batch, (data_1, data_2, data_3) in enumerate(data_train):
            images_1, images_2, images_3, masks = data_1[0], data_2[0], data_3[0], data_1[1]
            # print(images_1.shape, images_2.shape, images_3.shape, masks.shape)
            # ((2, 1, 1250, 1250), (2, 3, 1250, 1250), (2, 5, 1250, 1250), (2, 1250, 1250))

            outputs_1, outputs_2, outputs_3 = models[0](images_1.to(device)), models[1](images_2.to(device)), models[2](
                images_3.to(device))
            predict_map = mean_outputs([outputs_1, outputs_2, outputs_3])
            # predict_map = smooth_gaussian(predict_map, device)

            loss = loss_fun(predict_map, masks.to(device))
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
    elif len(SLICES_COLLECT) == 2:
        for batch, (data_1, data_2) in enumerate(data_train):
            images_1, images_2, masks = data_1[0], data_2[0], data_1[1]
            outputs_1, outputs_2 = models[0](images_1.to(device)), models[1](images_2.to(device))
            predict_map = mean_outputs([outputs_1, outputs_2])
            # predict_map = smooth_gaussian(predict_map)

            loss = loss_fun(predict_map, masks.to(device))
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
    elif len(SLICES_COLLECT) == 1:
        for batch, data in enumerate(data_train):
            images, masks = data[0], data[1]
            predict_map = models[0](images.to(device))
            # predict_map = smooth_gaussian(predict_map)

            loss = loss_fun(predict_map, masks.to(device))
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()


def get_loss_train(models, data_train, loss_fun, device, SLICES_COLLECT):
    for model in models:
        model.eval()

    total_acc = 0
    total_loss = 0
    if len(SLICES_COLLECT) == 3:
        for batch, (data_1, data_2, data_3) in enumerate(data_train):
            images_1, images_2, images_3, masks = data_1[0], data_2[0], data_3[0], data_1[1]
            with torch.no_grad():
                outputs_1, outputs_2, outputs_3 = models[0](images_1.to(device)), models[1](images_2.to(device)), \
                                                  models[2](
                                                      images_3.to(device))
                predict_map = mean_outputs([outputs_1, outputs_2, outputs_3])
                # predict_map = smooth_gaussian(predict_map)
                loss = loss_fun(predict_map, masks.to(device))
                pred_class = torch.argmax(predict_map, dim=1).float()
                acc = accuracy_check_for_batch(masks.cpu(), pred_class.cpu(), images_1.size()[0])
                total_acc += acc
                total_loss += loss.cpu().item()
    elif len(SLICES_COLLECT) == 2:
        for batch, (data_1, data_2) in enumerate(data_train):
            images_1, images_2, masks = data_1[0], data_2[0], data_1[1]
            with torch.no_grad():
                outputs_1, outputs_2 = models[0](images_1.to(device)), models[1](images_2.to(device))
                predict_map = mean_outputs([outputs_1, outputs_2])
                # predict_map = smooth_gaussian(predict_map)
                loss = loss_fun(predict_map, masks.to(device))
                pred_class = torch.argmax(predict_map, dim=1).float()
                acc = accuracy_check_for_batch(masks.cpu(), pred_class.cpu(), images_1.size()[0])
                total_acc += acc
                total_loss += loss.cpu().item()
    elif len(SLICES_COLLECT) == 1:
        for batch, data in enumerate(data_train):
            images, masks = data[0], data[1]
            with torch.no_grad():
                predict_map = models[0](images.to(device))
                # predict_map = smooth_gaussian(predict_map)
                loss = loss_fun(predict_map, masks.to(device))
                pred_class = torch.argmax(predict_map, dim=1).float()
                acc = accuracy_check_for_batch(masks.cpu(), pred_class.cpu(), images.size()[0])
                total_acc += acc
                total_loss += loss.cpu().item()

    return total_acc / (batch + 1), total_loss / (batch + 1)


def validate_model(models, data_val, loss_fun, epoch, make_prediction=True, save_folder_name='prediction',
                   device='cpu', SLICES_COLLECT=[]):
    """
        Validation run
    """
    # calculating validation loss
    total_val_loss = 0
    total_val_acc = 0
    if len(SLICES_COLLECT) == 3:
        for batch, (data_1, data_2, data_3) in enumerate(data_val):
            images_1, images_2, images_3, masks = data_1[0], data_2[0], data_3[0], data_1[1]
            with torch.no_grad():
                outputs_1, outputs_2, outputs_3 = models[0](images_1.to(device)), models[1](images_2.to(device)), \
                                                  models[2](
                                                      images_3.to(device))
                predict_map = mean_outputs([outputs_1, outputs_2, outputs_3])
                # predict_map = smooth_gaussian(predict_map)
                total_val_loss = total_val_loss + loss_fun(predict_map, masks.to(device)).cpu().item()
                # print('out', predict_map.shape) # (1, 2, 1250, 1250)
                pred_class = torch.argmax(predict_map, dim=1).float()  # (1, 1250, 1250)
            if make_prediction:
                im_name = batch
                pred_msk = save_prediction_image(pred_class, im_name, epoch, save_folder_name)
                acc_val = accuracy_check(masks.cpu(), pred_class.cpu())
                total_val_acc += acc_val
    elif len(SLICES_COLLECT) == 2:
        for batch, (data_1, data_2) in enumerate(data_val):
            images_1, images_2, masks = data_1[0], data_2[0], data_1[1]
            with torch.no_grad():
                outputs_1, outputs_2 = models[0](images_1.to(device)), models[1](images_2.to(device))
                predict_map = mean_outputs([outputs_1, outputs_2])
                # predict_map = smooth_gaussian(predict_map)
                total_val_loss = total_val_loss + loss_fun(predict_map, masks.to(device)).cpu().item()
                # print('out', predict_map.shape) # (1, 2, 1250, 1250)
                pred_class = torch.argmax(predict_map, dim=1).float()  # (1, 1250, 1250)
            if make_prediction:
                im_name = batch
                pred_msk = save_prediction_image(pred_class, im_name, epoch, save_folder_name)
                acc_val = accuracy_check(masks.cpu(), pred_class.cpu())
                total_val_acc += acc_val
    elif len(SLICES_COLLECT) == 1:
        for batch, data in enumerate(data_val):
            images, masks = data[0], data[1]
            with torch.no_grad():
                predict_map = models[0](images.to(device))
                # predict_map = smooth_gaussian(predict_map)
                total_val_loss = total_val_loss + loss_fun(predict_map, masks.to(device)).cpu().item()
                # print('out', predict_map.shape) # (1, 2, 1250, 1250)
                pred_class = torch.argmax(predict_map, dim=1).float()  # (1, 1250, 1250)
            if make_prediction:
                im_name = batch
                pred_msk = save_prediction_image(pred_class, im_name, epoch, save_folder_name)
                acc_val = accuracy_check(masks.cpu(), pred_class.cpu())
                total_val_acc += acc_val

    return total_val_acc / (batch + 1), total_val_loss / (batch + 1)


def test_model(model_path, data_test, epoch, save_folder_name='prediction'):
    """
        Test run
    """
    model = torch.load(model_path)
    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()
    model.eval()
    for batch, (images_t) in enumerate(data_test):
        stacked_img = torch.Tensor([]).cuda()
        for index in range(images_t.size()[1]):
            with torch.no_grad():
                image_t = images_t[:, index, :, :].unsqueeze(0)  # .cuda()
                # print(image_v.shape, mask_v.shape)
                output_t = model(image_t)
                output_t = torch.argmax(output_t, dim=1).float()
                stacked_img = torch.cat((stacked_img, output_t))
        im_name = batch  # TODO: Change this to real image name so we know
        _ = save_prediction_image(stacked_img, im_name, epoch, save_folder_name)

    print("Finish Prediction!")


def save_prediction_image(pred_class, im_name, epoch, save_folder_name="result_images"):
    """save images to save_path
    Args:
        pred_class (numpy): pred_class images
        save_folder_name (str): saving folder name
    """
    img_as_np = pred_class.cpu().data.numpy()

    img_as_np = polarize(img_as_np) * 255
    img_as_np = img_as_np.astype(np.uint8)
    #    print(img_as_np, img_as_np.shape)
    img = Image.fromarray(img_as_np.squeeze(0))
    # organize images in every epoch
    desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(im_name) + '.png'
    img.save(desired_path + export_name)
    return img_as_np


def polarize(img):
    ''' Polarize the value to zero and one
    Args:
        img (numpy): numpy array of image to be polarized
    return:
        img (numpy): numpy array only with zero and one
    '''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img


if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(1, 2, 1250, 1250)
    validate_model()
