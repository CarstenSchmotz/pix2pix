import torch
from .base_model import BaseModel
from . import networks

class Pix2PixModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc , opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #+ 1 input_nc remove aan opt.input_nc
        if self.isTrain: #plus 1 in in +out remove "+ opt.output_nc"
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            #print(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, "opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D")

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""

        #masking for discriminator is a bit undefined...
        fake_B = self.fake_B
        mask_B = self.real_B == -1
        #mask_B = torch.where(mask_B == True, torch.rand(mask_B.size()).to(self.device) > 0.001, mask_B)
        fake_B = torch.where(mask_B == True, -1, fake_B)
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        fake_B = self.fake_B
        #normalized images are in range [-1,1] so we can use -1 as a mask
        mask_B = self.real_B == -1
        #optional hack to force 0 intensity for sky
        #mask_B = torch.where(mask_B == True, torch.rand(mask_B.size()).to(self.device) > 0.001, mask_B)
        fake_B = torch.where(mask_B == True, -1, fake_B)

        #masking also not fully defined for generator loss...
        fake_AB = torch.cat((self.real_A, fake_B), 1)
        pred_fake = self.netD(fake_AB)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(fake_B,self.real_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
         #wanted to log L1 before lambda multiplication but not necessary     
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.opt.lambda_L1*10
        


        # combine loss and calculate gradients
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
