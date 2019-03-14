import argparse
import os
import torch
import torchvision
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from models import CAE, LeNet, AlexNet, VGG, load_model

def get_loader(dataset, image_size, dataroot):
    """
    Returns required dataloader
    :param dataset:
    :return dataloader:
    """
    if dataset == 'mnist':
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(dataroot, train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.Resize(image_size),
                                           torchvision.transforms.ToTensor(),
                                       ])),
            batch_size=1, shuffle=True)
    elif dataset == 'svhn':
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(dataroot, train=False , download=True,
                                       transform=torchvision.transforms.Compose([
                                            torchvision.transforms.Resize(image_size),
                                            torchvision.transforms.ToTensor(),
                                        ])),
            batch_size=1, shuffle=True)
    return loader

def get_model(name, device):
    """
    Returns required classifier and autoencoder
    :param name:
    :return: Autoencoder, Classifier
    """
    if name == 'lenet':
        model = LeNet(in_channels=channels).to(device)
    elif name == 'alexnet':
        model = AlexNet(channels=channels, num_classes=10).to(device)
    elif name == 'vgg':
        model = VGG(in_channels=channels, num_classes=10).to(device)

    autoencoder = CAE(in_channels=channels).to(device)
    return model, autoencoder

def get_optimizer(model, lr):
    """
    Returns Adam optimizer for the model
    :param model:
    :param lr:
    :return Adam optimizer:
    """
    return torch.optim.Adam(model.parameters(), lr=lr)

def load(model, optimizer, root):
    """
    Returns pretrained model with optimizer and final training epoch
    :param model:
    :param optimizer:
    :param root:
    :return:
    """
    model, optimizer, epoch = load_model(model, optimizer, root)
    return model, optimizer, epoch

class Attack(object):
    """
    Attack class represents different perturbation methods
    """
    def __init__(self, model, autoencoder, loss_func):
        super(Attack, self).__init__()
        self.model = model
        self.autoencoder = autoencoder
        self.loss_func = loss_func

    def fgsm(self, image, label, eps=0.3):
        image.requires_grad = True
        out = model(image)
        loss = self.loss_func(out, label)
        loss.backward()
        image_grad = image.grad.data

        sign_grad = image_grad.sign()
        perturbed_image = image + eps * sign_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    def latent_fgsm(self, data, label, eps=0.3):
        latent = autoencoder.forward_encoder(data)
        latent = Variable(latent, requires_grad=True)
        data = autoencoder.forward_decoder(latent)

        out = model(data)
        loss = self.loss_func(out, label)
        loss.backward()
        latent_grad = latent.grad.data
        sign_grad = latent_grad.sign()
        perturbed_latent = latent + eps * sign_grad

        perturbed_image = autoencoder.forward_decoder(perturbed_latent)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    def i_fgsm(self, image, label, eps=0.03, alpha=1, iteration=1, x_val_min=-1, x_val_max=1):
        x_adv = Variable(image.data, requires_grad=True)
        for i in range(iteration):
            h_adv = self.model(x_adv)
            loss = -self.loss_func(h_adv, label)

            self.model.zero_grad()
            loss.backward()

            x_adv.grad.sign_()
            x_adv = x_adv - alpha * x_adv.grad
            x_adv = self._where(x_adv > image + eps, image + eps, x_adv)
            x_adv = self._where(x_adv < image - eps, image - eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

        h = self.model(image)
        h_adv = self.model(x_adv)

        return x_adv

    def latent_i_fgsm(self, image, label, eps=0.03, alpha=1, iteration=1, x_val_min=-1, x_val_max=1):
        latent = autoencoder.forward_encoder(image)
        latent = Variable(latent, requires_grad=True)
        for i in range(iteration):
            x_adv = self.autoencoder.forward_decoder(latent)
            h_adv = self.model(x_adv)
            loss = -self.loss_func(h_adv, label)

            self.model.zero_grad()
            loss.backward()

            latent.grad.sign_()
            latent_adv = latent - alpha * latent.grad
            latent_adv = self._where(latent_adv > latent + eps, latent + eps, latent_adv)
            latent_adv = self._where(latent_adv < latent - eps, latent - eps, latent_adv)

            x_adv = self.autoencoder.forward_decoder(latent_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            latent = Variable(latent_adv.data, requires_grad=True)

        h = self.model(image)
        h_adv = self.model(x_adv)

        return x_adv

    def _where(self, cond, x, y):
        """
        code from :
            https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
        """
        cond = cond.float()
        return (cond * x) + ((1 - cond) * y)


def main(model ,autoencoder ,test_loader, device, eps):
    attack = Attack(model=model, autoencoder=autoencoder, loss_func=F.nll_loss)
    fgsm_c = 0
    ifgsm_c = 0
    latent_fgsm_c = 0
    latent_i_fgsm_c = 0

    if not os.path.isdir('./res/fgsm'):
        os.mkdir('./res/fgsm')
    if not os.path.isdir('./res/ifgsm'):
        os.mkdir('./res/ifgsm')
    if not os.path.isdir('./res/ifgsm'):
        os.mkdir('./res/latent_fgsm')
    if not os.path.isdir('./res/latent_i_fgsm'):
        os.mkdir('./res/latent_i_fgsm')

    for i, (data, target) in enumerate(test_loader, 0):
        data, target = data.to(device), target.to(device)

        fgsm = attack.fgsm(data, target, eps)
        out = model(fgsm)
        _, pred = torch.max(out.data, 1)
        fgsm_c += (pred == target).sum().item()
        if pred.item() != target.item():
            torchvision.utils.save_image(data.data, './res/fgsm/{}_real.png'.format(i))
            torchvision.utils.save_image(fgsm.data, './res/fgsm/{}_{}_pert.png'.format(i, pred.item()))
        
        ifgsm = attack.i_fgsm(data, target, eps, iteration=10)
        out = model(ifgsm)
        _, pred = torch.max(out.data, 1)
        ifgsm_c += (pred == target).sum().item()
        
        if pred.item() != target.item():
            torchvision.utils.save_image(ifgsm.data, './res/ifgsm/{}_real.png'.format(i))
            torchvision.utils.save_image(ifgsm.data, './res/ifgsm/{}_{}_{}_pert.png'.format(i, pred.item(), target.item()))
        
        lfgsm = attack.latent_fgsm(data, target, eps)
        out = model(lfgsm)
        _, pred = torch.max(out.data, 1)
        latent_fgsm_c += (pred == target).sum().item()
        if pred.item() != target.item():
            torchvision.utils.save_image(data.data, './res/latent_fgsm/{}_real.png'.format(i))
            torchvision.utils.save_image(lfgsm.data, './res/latent_fgsm/{}_{}_pert.png'.format(i, pred.item()))

        l_i_fgsm = attack.latent_i_fgsm(data, target, eps, iteration=10)
        out = model(l_i_fgsm)
        _, pred = torch.max(out.data, 1)
        latent_i_fgsm_c += (pred == target).sum().item()
        torchvision.utils.save_image(data.data, './res/latent_i_fgsm/0.01/30/{}_real.png'.format(i))
        if pred.item() != target.item():
            torchvision.utils.save_image(data.data, './res/latent_i_fgsm/0.01/30/{}_real.png'.format(i))
            torchvision.utils.save_image(l_i_fgsm.data, './res/latent_i_fgsm/0.01/30/{}_{}_{}_pert.png'.format(i, pred.item(), target.item()))



#############################
# Run Test for all epsilons
#############################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Defensive GAN')
    parser.add_argument('--model', type=str, default='lenet',
                       help='Target classifier model: "lenet" or "alexnet" or "vgg"')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset : "mnist" , "svhn"')
    parser.add_argument('--dataroot', type=str, default='~/AI/Datasets/mnist/data',
                       help='Dataset root. Default is "./datasets"')
    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = opt.model
    dataset_name = opt.dataset
    dataroot = opt.dataroot
    image_size = 32
    lr = 0.001
    if dataset_name == 'svhn':
        channels = 3
    else:
        channels = 1

    model, autoencoder = get_model(model_name, device)
    dataloader = get_loader(dataset_name, image_size, dataroot)

    model_optim = get_optimizer(model, lr)
    ae_optim = get_optimizer(autoencoder, lr)

    model, model_optim, _ = load(model, model_optim, './saved_models/lenet_mnist32.pth')
    autoencoder, ae_optim, _ = load(autoencoder, ae_optim, './saved_models/cae_mnist32.pth')

    main(model=model, autoencoder=autoencoder, test_loader=dataloader, device=device, eps=0.01)