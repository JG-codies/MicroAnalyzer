import albumentations as A


def get_identity_augmentation():
    return A.Lambda(name='Identity')


def get_rigid_augmentation():
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Transpose(),
        A.RandomRotate90()
    ])


def get_non_spatial_augmentation():
    return A.OneOf([
        A.RandomBrightnessContrast(),
        A.RandomGamma(),
    ])


def compose_augmentations(aug_list):
    return A.Compose(aug_list)
