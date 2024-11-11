# utils.py

import cv2
import torch
import torch.nn as nn
from PIL import Image as pil_image
from dataset.transform import xception_default_data_transforms

def get_boundingbox(face, width, height, scale=0.8, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    :param image: numpy image in opencv form (BGR and of shape)
    :return: pytorch tensor of shape [1, 3, image_size, image_size]
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

def predict_with_model(image, model, post_function=nn.Softmax(dim=1), cuda=True):
    """
    Predicts the label of an input image.
    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda
    :return: prediction (1 = fake, 0 = real), output probabilities
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda)
    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)
    # Get prediction
    _, prediction = torch.max(output, 1)
    prediction = int(prediction.cpu().numpy())
    return prediction, output
