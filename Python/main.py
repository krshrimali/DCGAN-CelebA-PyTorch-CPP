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

class Generator(torch.nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False)
            torch.nn.BatchNorm2d(ngf*8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf*4),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf*2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )
    
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
    
class Discriminator(torch.nn.Module):        
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf*2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf*4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf*8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )
    def forward(self, input):
        if input.is_cuda() and self.ngpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

if __name__ == "__main__":
    args = Arguments("/home/ubuntu/dcgan/celebA/img_align_celeba", 2, 64, 64, 3, 100, 64, 64, 5, 0.0002, 0.5, 1)
    print(args)

    dataset = dset.ImageFolder(root = args.dataroot,
    transform=transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    assert dataset, f'dataset not loaded'

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=int(args.workers))

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    ngpu = int(args.ngpu)
    nz = int(args.nz)
    ngf = int(args.ngf)
    ndf = int(args.ndf)

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    print(netD)

    criterion = torch.nn.BCELoss()

    fixed_noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    for epoch in range(args.num_epochs):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size, ), real_label, device=device)
        
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()

            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    