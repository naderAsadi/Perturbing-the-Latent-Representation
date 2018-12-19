import argparse
import torch
import torchvision
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from models import CAE, LeNet, AlexNet, VGG, load_model

parser = argparse.ArgumentParser(description='Defensive GAN')
parser.add_argument('--model', type=str, default='lenet', help='Target classifier model: "lenet" or "alexnet" or "vgg"')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset : "mnist" , "svhn"')
parser.add_argument('--dataroot', type=str, default='~/AI/Datasets/mnist/data', help='Dataset root. Default is "./datasets"')
opt = parser.parse_args()

###############
# Parameters
###############
epsilons = [0.1, 0.2, 0.3]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 32*32
lr = 0.001
if opt.dataset == 'svhn':
    channels = 3
else:
    channels = 1

###############
# Dataset
###############
print(opt.dataroot)
mnist_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(opt.dataroot, train=False , download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        ])),
    batch_size=1, shuffle=True)

# svhn_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.SVHN(opt.dataroot, split='test', download=True, transform=torchvision.transforms.Compose([
#         torchvision.transforms.Resize(32),
#         torchvision.transforms.ToTensor(),
#         ])),
#     batch_size=1, shuffle=True)

###############
# Models
###############
autoencoder = CAE(in_channels=channels).to(device)
lenet = LeNet(in_channels=channels).to(device)
alexnet = AlexNet(channels=channels, num_classes=10).to(device)
vgg = VGG(in_channels=channels, num_classes=10).to(device)

#####################
# Optimizer & Loss
#####################
optim_lenet = torch.optim.Adam(lenet.parameters(), lr=lr)
optim_alexnet = torch.optim.Adam(alexnet.parameters(), lr=lr)
optim_vgg = torch.optim.Adam(vgg.parameters(), lr=lr)
optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=lr)

###############
# Pretrain
###############
if channels == 1:
    lenet,_,_ = load_model(lenet, optimizer=optim_lenet, root='./saved_models/lenet_mnist32.pth')
    alexnet,_,_ = load_model(alexnet, optimizer=optim_alexnet, root='./saved_models/alexnet_mnist.pth')
    vgg,_,_ = load_model(vgg, optimizer=optim_vgg, root='./saved_models/vgg16_mnist.pth')
    autoencoder,_,_ = load_model(autoencoder, optimizer=optimizer_ae, root='./saved_models/cae_mnist32.pth')
else:
    print('svhn models')
    #lenet, _, _ = load_model(lenet, optimizer=optim_lenet, root='./saved_models/lenet_svhn.pth')
    alexnet, _, _ = load_model(alexnet, optimizer=optim_alexnet, root='./saved_models/alexnet_svhn.pth')
    vgg, _, _ = load_model(vgg, optimizer=optim_vgg, root='./saved_models/vgg16_svhn.pth')
    autoencoder, _, _ = load_model(autoencoder, optimizer=optimizer_ae, root='./saved_models/cae_svhn.pth')

##############
# FSGM
##############
def simple_fsgm_attack(image, epsilon, data_grad):
    sign_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def fgsm(model, data, channels, target, epsilon, i, modelname):
    data.requires_grad = True
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    if init_pred.item() != target.item():
        return
    loss = F.nll_loss(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = simple_fsgm_attack(data, epsilon, data_grad)
    output = model(perturbed_data)
    pred = output.max(1, keepdim=True)[1]

    torchvision.utils.save_image(perturbed_data.reshape(-1, channels, 32, 32).data,
                                filename='./res/{}_{}_{}.png'.format(epsilon,i , pred.item()))

##############
# Test
##############
def test(model , device, test_loader, eps ):
    correct = 0
    adv_examples = []
    i = 1

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        encod = autoencoder.forward_encoder(data)
        encoded = Variable(encod, requires_grad=True)
        image = autoencoder.forward_decoder(encoded)
        output = model(image)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if init_pred.item() != target.item():
            continue
        loss = F.nll_loss(output, target)
        model.zero_grad()
        autoencoder.zero_grad()
        loss.backward()
        data_grad = encoded.grad.data
        perturbed_encoded = simple_fsgm_attack(encoded, eps, data_grad)
        perturbed_data = autoencoder.forward_decoder(perturbed_encoded)
        output = model(perturbed_data)
        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() != target.item():
            fgsm(model, data, channels, target, eps, i ,modelname='model')
            torchvision.utils.save_image(data.data, filename='./res/{}-{}_{}_real.png'.format(eps,i,final_pred.item()))
            torchvision.utils.save_image(perturbed_data.data, filename='./res/{}-{}_{}_recons.png'.format(eps,i,final_pred.item()))
            i += 1
        else:
            correct += 1


    # Calculate final accuracy for this eps
    acc = correct/float(len(test_loader))
    print("Epsilon: {}\ Accuracy = {}".format(eps, acc))
    # Return the accuracy and an adversarial example
    return acc,  adv_examples

#############################
# Run Test for all epsilons
#############################
accs = []
examples = []
autoencoder.eval()
lenet.eval()
alexnet.eval()
vgg.eval()

if opt.model == 'alexnet':
    print('model alex')
    model = alexnet
elif opt.model == 'vgg':
    print('model vgg')
    model = vgg
else:
    print('model lenet')
    if channels == 1:
        model = lenet
    else:
        model = alexnet

# Run test for each epsilon
if channels == 1:
    for eps in epsilons:
        acc, ex = test(model, device, mnist_loader, eps)
        accs.append(acc)
        examples.append(ex)
else:
    print('dataset svhn')
    for eps in epsilons:
        acc, ex = test(model, device, svhn_loader, eps)
        accs.append(acc)
        examples.append(ex)

print(accs)
