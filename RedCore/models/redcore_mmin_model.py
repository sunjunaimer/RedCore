
import torch
import pytorch_warmup as warmup
import numpy as np 
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstmvar import LSTMEncoder
from models.networks.textcnnvar import TextCNN
from models.networks.transformer import Transformer
from models.networks.classifier import FcClassifier
from models.networks.autoencoder import ResidualAE, ResidualXE
from models.networks.xencoder import LinearVXE
from models.utt_fusion_model import UttFusionModel
from .utils.config import OptConfig


class redcoreMMINModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='visual embedding method,last,mean or atten')
        parser.add_argument('--AE_layers', type=str, default='128,64,32', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--pretrained_path', type=str, help='where to load pretrained encoder network')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of mse loss')
        parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight of cycle loss')
        parser.add_argument('--share_weight', action='store_true', help='share weight of forward and backward autoencoders')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['CE', 'mse', 'cycle']
        self.model_names = ['A', 'V', 'L', 'C', 'AE', 'AE_cycle', 'Cls', 'xenc_al2v', 'xenc_av2l', 'xenc_vl2a', 'Cls_a', 'Cls_v', 'Cls_l']
        self.lossA = 0
        self.lossV = 0
        self.lossL = 0
        self.loss_beta = opt.beta   #0.95
        self.beta = np.array([1, 1, 1])
        self.eta = opt.eta  #0.001
        self.iter = 0
        self.interval_i = opt.ii
        self.etaext = opt.etaext
        self.mse_weight = opt.mse_weight


        # acoustic model
        #self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.netA = Transformer(opt.input_dim_a, 3, 8, opt.embd_size_a)
        self.netA.initialize_parameters()
        # lexical model
        #self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
        self.netL = Transformer(opt.input_dim_l, 3, 8, opt.embd_size_l)
        self.netL.initialize_parameters()
        # visual model
        #self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.netV = Transformer(opt.input_dim_v, 3, 8, opt.embd_size_v)
        self.netV.initialize_parameters()
        # AE model
        AE_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))
        AE_input_dim = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l
        self.netAE = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        if opt.share_weight:
            self.netAE_cycle = self.netAE
            self.model_names.pop(-1)
        else:
            self.netAE_cycle = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = AE_layers[-1] * opt.n_blocks
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.feature_dim = 32
        self.netxenc_al2v = ResidualXE(AE_layers, opt.n_blocks, opt.embd_size_a + opt.embd_size_l, opt.embd_size_v, dropout=0, use_bn=False)
        self.netxenc_av2l = ResidualXE(AE_layers, opt.n_blocks, opt.embd_size_a + opt.embd_size_v, opt.embd_size_l, dropout=0, use_bn=False)
        self.netxenc_vl2a = ResidualXE(AE_layers, opt.n_blocks, opt.embd_size_v + opt.embd_size_l, opt.embd_size_a, dropout=0, use_bn=False)
        self.netCls = FcClassifier(opt.embd_size_a + opt.embd_size_l + opt.embd_size_v, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netCls_a = FcClassifier(opt.embd_size_a, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netCls_v = FcClassifier(opt.embd_size_v, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netCls_l = FcClassifier(opt.embd_size_l, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)


        if self.isTrain:
            #self.load_pretrained_encoder(opt)
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_mse = torch.nn.MSELoss(reduction='sum')
            self.criterion_bce = torch.nn.BCELoss(reduction='sum')
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

            
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=opt.total_iters)
            self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)


            self.output_dim = opt.output_dim
            self.ce_weight = opt.ce_weight
            self.mse_weight = opt.mse_weight
            self.cycle_weight = opt.cycle_weight

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def load_pretrained_encoder(self, opt):
        print('Init parameter from {}'.format(opt.pretrained_path))
        pretrained_path = os.path.join(opt.pretrained_path, str(opt.cvNo))
        pretrained_config_path = os.path.join(opt.pretrained_path, 'train_opt.conf')
        pretrained_config = self.load_from_opt_record(pretrained_config_path)
        pretrained_config.isTrain = False                             # teacher model should be in test mode
        pretrained_config.gpu_ids = opt.gpu_ids                       # set gpu to the same
        # self.pretrained_encoder = UttFusionModel(pretrained_config)
        # self.pretrained_encoder.load_networks_cv(pretrained_path)
        # self.pretrained_encoder.cuda()
        # self.pretrained_encoder.eval()
    
    def post_process(self):
        # called after model.setup()
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.'+key, value) for key, value in state_dict.items()])
        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            
            # self.netA.load_state_dict(f(self.pretrained_encoder.netA.state_dict()))
            # self.netV.load_state_dict(f(self.pretrained_encoder.netV.state_dict()))
            # self.netL.load_state_dict(f(self.pretrained_encoder.netL.state_dict()))
        
    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        acoustic = input['A_feat'].float().to(self.device)
        lexical = input['L_feat'].float().to(self.device)
        visual = input['V_feat'].float().to(self.device)
        #if self.isTrain:
        if True:
            self.label = input['label'].to(self.device)
            self.missing_index = input['missing_index'].long().to(self.device)
            # A modality
            self.A_miss_index = self.missing_index[:, 0].unsqueeze(1).unsqueeze(2)
            self.A_miss = acoustic * self.A_miss_index
            self.A_reverse = acoustic * -1 * (self.A_miss_index - 1)
            # L modality
            self.L_miss_index = self.missing_index[:, 2].unsqueeze(1).unsqueeze(2)
            self.L_miss = lexical * self.L_miss_index
            self.L_reverse = lexical * -1 * (self.L_miss_index - 1)
            # V modality
            self.V_miss_index = self.missing_index[:, 1].unsqueeze(1).unsqueeze(2)
            self.V_miss = visual * self.V_miss_index
            self.V_reverse = visual * -1 * (self.V_miss_index - 1)
            
        else:
            self.A_miss = acoustic
            self.V_miss = visual
            self.L_miss = lexical

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # get utt level representattion
        self.feat_A_miss, self.fmu_a, self.flogvar_a = self.netA(self.A_miss)
        self.feat_V_miss, self.fmu_v, self.flogvar_v = self.netV(self.V_miss)
        self.feat_L_miss, self.fmu_l, self.flogvar_l = self.netL(self.L_miss)
        
        # fusion miss
        self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1)
        # calc reconstruction of teacher's output
        self.recon_fusion, self.latent = self.netAE(self.feat_fusion_miss)
        self.recon_cycle, self.latent_cycle = self.netAE_cycle(self.recon_fusion)


        #generate the feature of missing modalities
        self.gen_a, latent_a = self.netxenc_vl2a(torch.cat([self.feat_V_miss, self.feat_L_miss], dim=-1))
        self.gen_l, latent_l = self.netxenc_av2l(torch.cat([self.feat_A_miss, self.feat_V_miss], dim=-1))
        self.gen_v, latent_v = self.netxenc_al2v(torch.cat([self.feat_A_miss, self.feat_L_miss], dim=-1))

        bs = self.feat_A_miss.shape[0]
        self.feat_A_r = self.A_miss_index.reshape(bs, 1) * self.feat_A_miss - (self.A_miss_index.reshape(bs, 1) - 1) * self.gen_a 
        self.feat_L_r = self.L_miss_index.reshape(bs, 1) * self.feat_L_miss - (self.L_miss_index.reshape(bs, 1) - 1) * self.gen_l 
        self.feat_V_r = self.V_miss_index.reshape(bs, 1) * self.feat_V_miss - (self.V_miss_index.reshape(bs, 1) - 1) * self.gen_v 

        self.feat_A_re = self.A_miss_index * (self.feat_A_miss - self.gen_a)
        self.feat_L_re = self.L_miss_index * (self.feat_L_miss - self.gen_l)
        self.feat_V_re = self.V_miss_index * (self.feat_V_miss - self.gen_v)
        self.feat_fusion_r = torch.cat([self.feat_A_r, self.feat_L_r, self.feat_V_r], dim=-1)
        self.logits, _ = self.netCls(self.feat_fusion_r)
        self.pred = F.softmax(self.logits, dim=-1)

        self.logits_a, _ = self.netCls_a(self.feat_A_r)
        self.logits_v, _ = self.netCls_v(self.feat_V_r)
        self.logits_l, _ = self.netCls_l(self.feat_L_r)
        
        # # for training 
        # if self.isTrain:
        #     with torch.no_grad():
        #         # self.T_embd_A = self.pretrained_encoder.netA(self.A_reverse)
        #         # self.T_embd_L = self.pretrained_encoder.netL(self.L_reverse)
        #         # self.T_embd_V = self.pretrained_encoder.netV(self.V_reverse)
        #         self.T_embd_A = self.netA(self.A_reverse)
        #         self.T_embd_L = self.netL(self.L_reverse)
        #         self.T_embd_V = self.netV(self.V_reverse)
        #         self.T_embds = torch.cat([self.T_embd_A, self.T_embd_L, self.T_embd_V], dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        bs = self.A_miss_index.shape[0]
        a_index = self.A_miss_index.reshape(bs, 1)
        v_index = self.V_miss_index.reshape(bs, 1)
        l_index = self.L_miss_index.reshape(bs, 1)
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label)
        self.loss_CE_a = self.ce_weight * self.criterion_ce(self.logits_a, self.label)
        self.loss_CE_v = self.ce_weight * self.criterion_ce(self.logits_v, self.label)
        self.loss_CE_l = self.ce_weight * self.criterion_ce(self.logits_l, self.label)
        lambda1 = 0.0008


        kld_fa = -1 * lambda1 * torch.sum((1 + self.flogvar_a - self.fmu_a.pow(2) - self.flogvar_a.exp()) * a_index) / bs
        kld_fv = -1 * lambda1 * torch.sum((1 + self.flogvar_v - self.fmu_v.pow(2) - self.flogvar_v.exp()) * v_index) / bs
        kld_fl = -1 * lambda1 * torch.sum((1 + self.flogvar_l - self.fmu_l.pow(2) - self.flogvar_l.exp()) * l_index) / bs


        bs_a = sum(self.A_miss_index)
        bs_v = sum(self.V_miss_index)
        bs_l = sum(self.L_miss_index)

        
        self.lossA_r = self.criterion_mse(self.gen_a*a_index, self.feat_A_miss*a_index) / bs_a
        self.lossV_r = self.criterion_mse(self.gen_v*v_index, self.feat_V_miss*v_index) / bs_v
        self.lossL_r = self.criterion_mse(self.gen_l*l_index, self.feat_L_miss*l_index) / bs_l

        if self.lossA_r.item() == 0:
            lossA_update = self.lossA
        else:
            lossA_update = self.lossA_r
        if self.lossV_r == 0:
            lossV_update = self.lossV
        else:
            lossV_update = self.lossV_r
        if self.lossL_r == 0:
            lossL_update = self.lossL
        else:
            lossL_update = self.lossL_r

        self.lossA = (1 - self.loss_beta) * self.lossA + self.loss_beta * lossA_update
        self.lossV = (1 - self.loss_beta) * self.lossV + self.loss_beta * lossV_update
        self.lossL = (1 - self.loss_beta) * self.lossL + self.loss_beta * lossL_update

        lossAVL = np.array([self.lossA.item(), self.lossV.item(), self.lossL.item()])
        loss_avg = sum(lossAVL) / 3
        ra = (loss_avg - lossAVL ) / loss_avg

        if self.iter % 500 ==0 :
            self.eta = self.eta * self.etaext
        if self.iter % self.interval_i == 0:
            self.beta = self.beta - self.eta * ra 
            self.beta[0] = max(0.1, self.beta[0])
            self.beta[1] = max(0.1, self.beta[1])
            self.beta[2] = max(0.1, self.beta[2])
            self.beta = self.beta / (sum(self.beta**2)**(0.5))
        self.iter += 1

        self.loss_mse = self.mse_weight * (self.beta[0] * self.lossA_r + self.beta[1] * self.lossV_r + self.beta[2] * self.lossL_r)

        self.loss_cycle = 0

        loss = self.loss_CE + (kld_fa + kld_fv + kld_fl) + (self.loss_CE_a + self.loss_CE_v + self.loss_CE_l) + self.loss_mse
        #print('loss:', loss)


        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 1.0)
            
    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 
        #with self.warmup_scheduler.dampening():
        #    self.lr_scheduler.step()
