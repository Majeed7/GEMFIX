import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from models.cnn import LeNet
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from sklearn.metrics import confusion_matrix

from skimage.segmentation import slic
from skimage.color import label2rgb, rgb2gray
from skimage import io

import shap
from explainer.gemfix import GEMFIX
from explainer.bishapley_kernel import Bivariate_KernelExplainer

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the MNIST dataset
])

# Download training data
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Download test data
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


training = False

def load_pretrained_model_and_predict():
    # Load the pretrained model
    model = LeNet()
    model.load_state_dict(torch.load('lenet_mnist_model.pth'))
    model.eval()

    # Load one example from the MNIST test set
    test_set = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    data, target = next(iter(test_loader))

    # Predict
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability

    return model 

def evaluate_model(model, device):
    # Transformation for evaluating model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST test data
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    model.eval()  # Set the model to evaluation mode
    model.to(device)

    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():  # Operations inside don't track history
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.view(-1).tolist())
            all_targets.extend(target.view(-1).tolist())

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

class ImageSuperpixelWrapper:
    def __init__(self, image, n_segments=100, compactness=10.0, sigma=1.0, classifier=None, explanation_index=0, base_pixel_color = [128]*3):
        #self.image = io.imread(image_path)
        if type(image) == 'str':
            self.image = io.imread(image)
            if self.image.ndim == 3:
                self.image = rgb2gray(self.image)  # Convert to greyscale if not already
        else:
            self.image = image

        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.segments = slic(self.image, n_segments=n_segments, compactness=self.compactness, sigma=self.sigma, channel_axis=None)
        self.n_segments = np.max(self.segments)
        self.vector = np.ones(self.n_segments)
        self.classifier = classifier
        self.explanation_index = explanation_index
        self.base_pixel_color = base_pixel_color

    def show_segments(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True, sharey=True)
        ax.imshow(label2rgb(self.segments, self.image, kind='avg'), interpolation='nearest')
        ax.set_title('SLIC Segmentation')
        ax.axis('off')
        plt.show()

    def encode_superpixels(self):
        # Binary vector for superpixels' presence
        self.vector = np.ones(self.n_segments)

    def remove_superpixel(self, sp_id):
        # Set to 0 in binary vector
        modified_img = self.image.copy()
        modified_img[self.segments == sp_id] = self.base_pixel_color  # Black out the superpixel
        modified_img.vector[sp_id] = 0
    
        return modified_img

    def binary_encode(self, binary_code):
        modified_img = self.image.copy()
        remove_idx = np.where(np.array(binary_code) == 0)
        index = np.isin(self.segments, remove_idx)
        modified_img[index] = self.base_pixel_color
        
        return modified_img

    def show_explanation(self, impotance, sp_no=10, type='positive'):
        coef = 1 if type == 'positive' else -1
        sort_ind = np.argsort(-coef*impotance)[:sp_no] ## sort descedning the values and get the most important features

        binary_encode_exp = np.zeros_like(self.vector)
        binary_encode_exp[sort_ind] = 1
        img_exp = self.binary_encode(binary_encode_exp)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True, sharey=True)
        ax.imshow(img_exp)

        return img_exp

    def map_original_to_superpixel(self):
        # Maps each pixel to its superpixel ID
        return self.segments

    def map_superpixel_to_original(self, sp_id):
        # Find all pixels belonging to a given superpixel ID
        return np.argwhere(self.segments == sp_id)
    
    def __call__(self, binary_encode):
        if len(binary_encode.shape) == 1:
            modified_image = self.binary_encode(binary_encode)
            probabilities = torch.nn.functional.softmax(self.classifier(torch.from_numpy(modified_image).unsqueeze(0)), dim=1).squeeze(0)

            rtn_val = np.array(probabilities.detach().numpy()[self.explanation_index])
            return rtn_val[...,np.newaxis]
        else:
            pred = []
            for i in range(binary_encode.shape[0]):
                modified_image = self.binary_encode(binary_encode[i,])
                probabilities = torch.nn.functional.softmax(self.classifier(torch.from_numpy(modified_image).unsqueeze(0)), dim=1).squeeze(0)
                pred.append(probabilities.detach().numpy().squeeze()[self.explanation_index])
            
            return np.array(pred)


if __name__ == '__main__':

    # Load a CNN for image classification trained over MNIST
    # The model is stored in folder 'model, where there is a script cnn.py that inlcudes the training part
    model = LeNet()  # Assuming LeNet is already defined
    model.load_state_dict(torch.load('models/lenet_mnist_model.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #evaluate_model(model, device)

    # the base pixel color for generating samples for explanation
    base_pixel_color =  -0.4242 # the background color for the MNIST dataset

    img = train_set[1][0] # explaining the first data in the training set
    probabilities = torch.nn.functional.softmax(model(img), dim=1).squeeze(0)
    predicted_class = probabilities.argmax(dim=0).numpy()
    img = img.squeeze().numpy()

    # create a wrapper for the image where it identifies the superpixel of the image under explanation
    wrapper = ImageSuperpixelWrapper(img, classifier=model, explanation_index=predicted_class, base_pixel_color=base_pixel_color)

    # create a baseline x_train based on the number of pixel identified in wrapper 
    binary_encode = wrapper.vector
    x_train = np.zeros((1, binary_encode.shape[0])) # baseline: x_train --> a zero vector that shows to remove the corresponding superpixel (which is replaced by base_pixel_color in wrapper) 
    wrapper(x_train[0])
    
    # kshap = shap.KernelExplainer(wrapper, x_train)
    # kshap_values = kshap.shap_values(binary_encode)
    # wrapper.show_explanation(kshap_values, sp_no=20)
    
    gemfix = GEMFIX(wrapper, x_train)
    gemfix_values = gemfix.shap_values(binary_encode)
    wrapper.show_explanation(gemfix_values, sp_no=20)

    bishap = Bivariate_KernelExplainer(wrapper, x_train)
    bishap_values = bishap.shap_values(binary_encode)
    wrapper.show_explanation(bishap_values, sp_no=20)


    print("done!")