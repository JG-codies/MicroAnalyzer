import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve as __roc_cuve
from .preprocessing import make_stack_binary


def plot_history(history, metric='loss', save_path=None, scaled=False):
    plt.figure()
    plt.title(f'{metric.capitalize()} by Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    if scaled:
        plt.ylim(0, 1)
    plt.plot(history.history[metric], label=f'training {metric}')

    # check for validation data
    validation_metric_key = f'val_{metric}'
    if validation_metric_key in history.history:
        plt.plot(history.history[validation_metric_key], label=f'validation {metric}')
        plt.legend()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def roc_curve(binary_images, prediction_images):
    assert binary_images.shape == prediction_images.shape, 'incompatible prediction and label stacks'

    y_true = binary_images.reshape(-1)
    y_scores = prediction_images.reshape(-1)

    fpr, tpr, thresholds = __roc_cuve(y_true, y_scores)
    plt.figure()
    plt.title('ROC Curve')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.ylim(0, 1)
    plt.plot(fpr, tpr)
    plt.show()

    return thresholds


def compare_images(image, true_mask, pred, t, axes=None):
    true_mask = make_stack_binary(true_mask[None])[0]

    img_cmap = None
    if image.ndim == 3:
        h, w, c = image.shape
        if c == 1:
            image = image.reshape(h, w)
            img_cmap = 'gray'
        else:
            assert c == 3, 'only support RGB and grayscale images'
    else:
        assert image.ndim == 2, 'image parameter is not a single image'
        img_cmap = 'gray'

    if true_mask.ndim == 3:
        h, w, c = true_mask.shape
        assert c == 1, 'only support grayscale binary masks'
        true_mask = true_mask.reshape(h, w)
    else:
        assert true_mask.ndim == 2, 'mask parameter is not a single image'

    if pred.ndim == 3:
        h, w, c = pred.shape
        assert c == 1, 'predictions must be a grayscale probability map'
        pred = pred.reshape(h, w)
    else:
        assert pred.ndim == 2, 'pred parameter is not a single image'

    if axes is None:
        _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(100, 100))
    else:
        ax1, ax2, ax3, ax4, ax5 = axes

    pred_mask = (pred > t).astype(int)

    # remove true objects from pred mask to see extra objects
    extras_in_pred = pred_mask - true_mask
    extras_in_pred[extras_in_pred < 0] = 0

    # remove pred objects from true mask to see missing objects
    missing_in_pred = true_mask - pred_mask
    missing_in_pred[missing_in_pred < 0] = 0

    ax1.set_title('img', fontsize=150)
    ax1.imshow(image, cmap=img_cmap)

    ax2.set_title('true msk', fontsize=150)
    ax2.imshow(true_mask, cmap='gray')

    ax3.set_title('pred msk', fontsize=150)
    ax3.imshow(pred_mask, cmap='gray')

    ax4.set_title('extras in pred', fontsize=150)
    ax4.imshow(extras_in_pred, cmap='gray')

    ax5.set_title('missing in pred', fontsize=150)
    ax5.imshow(missing_in_pred, cmap='gray')

    plt.tight_layout()
    plt.show()
