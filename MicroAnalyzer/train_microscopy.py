from collections import namedtuple

import torch

import torch_seg.references.transforms as T
from torch_seg.preprocessing.datasets import MicroscopyDataset
from torch_seg.references import utils
from torch_seg.references.engine import train_one_epoch
from torch_seg.torch_model import get_instance_segmentation_model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train(images_dir, weights_path, test_dir=None, maxdets=100):

    # use our dataset and defined transformations
    dataset = MicroscopyDataset(images_dir, get_transform(train=True))


    # define training data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    if test_dir:
        dataset_test = MicroscopyDataset(test_dir, get_transform(train=False))
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)
    else:
        data_loader_test = None

    # use best device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and cell
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes, maxdets)

    # move model to the right device
    model.to(device)

    # train 30 epochs
    num_epochs = 30

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0.1e-4)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=10,
                                                   gamma=0.1)
    history = {}
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10,
                                        data_loader_test=data_loader_test)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        # if test_dir:
        #     evaluate(model, data_loader_test, device=device)

        for k in metric_logger.meters:
            history.setdefault(k, []).append(metric_logger.meters[k].global_avg)

    # save model at the end
    torch.save(model.state_dict(), weights_path)

    return namedtuple('history', 'history')(history)
