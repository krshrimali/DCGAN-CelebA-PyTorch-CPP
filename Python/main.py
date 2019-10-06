import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

class Arguments:
    def __init__(self, data_root, workers, batch_size, image_size,
    nc, nz, ngf, ndf, num_epochs, lr, beta1, ngpu):
        self.dataroot = data_root
        self.workers = workers
        self.batch_size = batch_size
        self.image_size = image_size
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.num_epochs = num_epochs
        self.lr = lr
        self.beta1 = beta1
        self.ngpu = ngpu
    
    def __str__(self):
        return "Arguments\n----------\ndataroot = {},\nworkers = {},\nbatch size = {},\nimage size = {},\nnc = {},\nnz = {},\nngf = {},\nndf = {},\nnum_epochs = {},\nlr = {},\nbeta1 = {},\nngpu = {}".format(self.dataroot, \
                    self.workers, self.batch_size, self.image_size, self.nc, self.nz, self.ngf, self.ndf, self.num_epochs, self.lr, self.beta1, self.ngpu)
    
if __name__ == "__main__":
    args = Arguments("/home/ubuntu/dcgan/celebA", 2, 64, 64, 3, 100, 64, 64, 5, 0.0002, 0.5, 1)
    print(args)

    dataset = dset.ImageFolder(root = args.dataroot,
    transform=transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    assert dataset