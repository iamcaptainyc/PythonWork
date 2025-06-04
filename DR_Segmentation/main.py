import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import random

import wandb
from torch.utils.data.dataloader import DataLoader
import traceback

import segmentation_models_pytorch as smp

import logger
from pandas.tests.window.conftest import adjust
from model.gan import GANSLTrainer

# from utils import *
# from metrics import *
# from loss import get_criterion, get_scheduler
# from dataset import IDRiDDataset
# from transform import *
# from model.shufflenetv2_sa import *
# from model.sa_resnet import *
# from model.model import *
# from sl import SLTrainer
# from model.deeplabv3_plus import DeepLab

def train(args):
    args=load_args(args)
    seed_torch(args.seed)
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #数据集
    if args.dataset_name=='idrid':
        args.dataset_path='/kaggle/input/idrid-dataset/A. Segmentation'
        trn_set=IDRiDDataset(args.dataset_path, dataset_type='train', resolution=args.resolution, leisions=args.leisions, args=args)
        val_set=IDRiDDataset(args.dataset_path, dataset_type='val', resolution=args.resolution, leisions=args.leisions, args=args)
    elif args.dataset_name=='vessel':
        args.dataset_path='/kaggle/input/retinal-vessel-segmentation-combined/Retina Vessel Segmentation'
        trn_set=VesselDataset(args.dataset_path, dataset_type='train', resolution=args.resolution, args=args)
        val_set=VesselDataset(args.dataset_path, dataset_type='val', resolution=args.resolution, args=args)
    elif args.dataset_name=='ddr':
        if args.preprocess:
            args.dataset_path='/kaggle/input/cropped-ddr-seg'
        else:
            args.dataset_path='/kaggle/input/ddr-dataset/lesion_segmentation'
        trn_set=DDRDataset(args.dataset_path, dataset_type='train', resolution=args.resolution, args=args)
        val_set=DDRDataset(args.dataset_path, dataset_type='test', resolution=args.resolution, args=args)
    
    if args.use_T:
        trn_set.transform=T.Compose([
                    ToNumpy(),
                    OneOf([
                        Resize(args.crop_size),
                        RandomCrop(args.crop_size),
                        ],
                          choose_one_of=args.choose_one_of),
                    HorizontalFlip(),
                    VerticalFlip(),
                    RandomRotate90(),
                    ColorJitter(),
                    Grid(args.grid_size),
                    ToTensor(),
                    Normalize()
                ])

        val_set.transform=T.Compose([
                    ToNumpy(),
                    Grid(args.grid_size),
                    ToTensor(),
                    Normalize()
        ])
    
    
    print(f'transform:{trn_set.transform}')

    trn_loader=DataLoader(trn_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader=DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    #损失函数
    criterion = get_criterion(args, device=device)

    #模型
    kwargs={'args':args}
    kwargs['num_classes']=args.num_classes
    kwargs['backbone']=args.backbone
    kwargs['pretrained']=args.pretrained
    kwargs['downsample_factor']=args.downsample_factor
    kwargs['module_list']=args.module_list
    kwargs['resolution']=args.crop_size[0] if args.crop_size else args.resolution[0]
    kwargs['upsample_mode']=args.upsample_mode
    kwargs['dual']=args.dual
    if args.netframe=='deeplabv3':
        model=DeepLab(**kwargs)
    elif args.netframe=='unet':
        if args.backbone=='ds':
            model=UNetDenseNet(**kwargs)
        elif args.backbone[:2]=='rs':
            model=UNetResNet(**kwargs)
        elif args.backbone=='new':
            model=UNetWithResnet50Encoder(n_classes=args.num_classes)
    elif args.netframe=='unetpp':
        model=UnetPP(**kwargs)
    elif args.netframe=='segformer':
        model=Segformer(num_classes=args.num_classes)
    elif args.netframe=='lseg':
        model=ResNetLseg(**kwargs)
    elif args.netframe[:3]=='smp':
        decoder=getattr(smp, args.netframe[4:])
        model = decoder(
            encoder_name=args.backbone,
            classes=args.num_classes,
            encoder_weights='imagenet' if args.pretrained else None
        )
    elif args.netframe=='dwunet':
        model=DWUnet(dwconv='wtconv',**kwargs)
    elif args.netframe=='hiformer':
        model=HiFormer(img_size=args.crop_size[0] if args.crop_size else args.resolution[0],
                       n_classes=args.num_classes)
    elif args.netframe=='transunet':
        model=VisionTransformer(img_size=kwargs['resolution'], 
                                num_classes=args.num_classes, 
                                upsample_mode=args.upsample_mode, 
                                transformer_block=args.transformer_block, 
                                net_type=args.net_type, 
                                decoder=args.transunet_decoder)
    # model=model(**kwargs)
    model.to(device)
    
#     print({name:m for name,m in model.named_children()})
    
    
    #优化器
    if args.optim == 'Adam':
        optimizer=torch.optim.Adam(model.parameters(),lr=args.lr, betas=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'SGD':
        optimizer=torch.optim.SGD(model.parameters(),lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'AdamW':
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr, betas=args.momentum, weight_decay=args.weight_decay)
    
    
    # for k,v in args.get_dict().items():
    #     print(f'{k} : {v}')
    print(args.get_dict())
    
    #半监督训练
    if args.ssl=='SL':
        trainer = SLTrainer(model, optimizer, criterion, device, args)
    elif args.ssl=='MTSL':
        trainer = MTSLTrainer(model, optimizer, criterion, device, args)
    if args.ssl=='GANSL':
        if args.gan_grading_dataset=='idrid':
            args.dataset_path='/kaggle/input/indian-diabetic-retinopathy-image-datasetidrid/Disease Grading'
            grading_trn_set=IDataset(args.dataset_path, dataset_type='train', resolution=args.resolution, args=args)
            grading_val_set=IDataset(args.dataset_path, dataset_type='val', resolution=args.resolution, args=args)
        
        if args.use_T:
            #使用IMAGENET数据集的均值与标准差进行归一化，有助于网络训练
            channel_stats = dict(mean = [0.425753653049469, 0.29737451672554016, 0.21293757855892181],
                             std = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114])
            trn_set.transform=T.Compose([
    #                     transforms.RandomApply(
    #                         [transforms.RandomCrop(512)],
    #                         p=0.5
    #                     ),
                        T.RandomHorizontalFlip(p=0.5),
                        T.RandomVerticalFlip(p=0.5),
                        T.RandomApply(
                            [T.ColorJitter(
                                brightness=0.2, contrast=0.2,
                                saturation=0, hue=0
                            )],
                            p=0.5
                        ),
                        T.RandomApply(
                            [T.RandomRotation(
                                degrees=[-180,180]
                            )],
                            p=0.5
                        ),
                        T.RandomApply(
                            [T.RandomAffine(
                                degrees=0,
                                translate=[0.2,0.2]
                            )],
                            p=0.5
                        ),
                        T.ToTensor(),
                        T.Normalize(**channel_stats)
                    ])
    
            val_set.transform=T.Compose([
                        T.ToTensor(),
                        T.Normalize(**channel_stats)
            ])
            
            grading_trn_loader=DataLoader(grading_trn_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
            grading_val_loader=DataLoader(grading_val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
            grading_model = resnet50(pretrained=False, module_list=args.module_list, weights=args.backbone_weights)
            grading_criterion=torch.nn.CrossEntropyLoss()
            optimizerG=torch.optim.SGD(grading_model.parameters(),lr=1e-3, momentum=0.9, weight_decay=0.0005)
            
            trainer = GANSLTrainer(model, grading_model, [optimizer, optimizerG], [criterion, grading_criterion], device, args)
        
            
            
    if args.evaluate:
        return trainer.evaluate(val_loader)
    if args.inference:
        trainer.inference(trn_loader, type='trn')
        trainer.inference(val_loader, type='val')
        return
        
    return trainer.loop(args.num_epochs, trn_loader, val_loader)
    
def main():
    args = create_argparser()
    try:
        if args.ablation:
            for selection in args.selection:
                if isinstance(args.ablation, list):
                    assert isinstance(selection, list), 'selection element not list!'
                    args=update_args(args, {a:s for a,s in zip(args.ablation,selection)})
                else:
                    args=update_args(args, {args.ablation:selection})
                train(args)
        else:
            train(args)
    except:
        traceback.print_exc()
    finally:
        if args.use_wandb: wandb.finish()
            
def load_args(args):
    if args.resume_checkpoint:
        if os.path.exists(args.resume_checkpoint):
            _, cpt_name=os.path.split(args.resume_checkpoint)
            if '.pt' in cpt_name:
                print(f'loading args at {args.resume_checkpoint}')
                cpt_dict = torch.load(args.resume_checkpoint, map_location='cpu')
            else:
                return args
        else:
            return args

        if cpt_dict:
            _args = cpt_dict.get('args',-1)
            if _args != -1:
                _args.evaluate=args.evaluate
                _args.inference=args.inference
                _args.inference_root_dir=args.inference_root_dir
                if args.inference:
                    _args.inference_suffix=args.inference_suffix
                    _args.dataset_name=args.dataset_name
                    _args.resolution=args.resolution
                    _args.preprocess=args.preprocess
                    _args.use_T=args.use_T
                _args.num_epochs=args.num_epochs
                _args.skip_epoch=args.skip_epoch
                _args.save_interval=args.save_interval
                _args.validation_interval=args.validation_interval
                _args.good_value=args.good_value
                _args.resume_checkpoint=args.resume_checkpoint
                _args.use_wandb=args.use_wandb
                _args.load_optimizer=args.load_optimizer
                _args.account=args.account
                _args.a_version=args.a_version
                # _args.crop_size=args.crop_size
                _args.choose_one_of=args.choose_one_of
                _args.ablation=args.ablation
                _args.selection=args.selection
                args=adjust_args(_args)
                print('args has been loaded.')
            return args
    return args
            
            
def update_args(args, kwargs):
    args.update(kwargs)
    args=adjust_args(args)
    return args
    
def adjust_args(args):
    if not isinstance(args.resolution, tuple):
        args.resolution=(args.resolution, args.resolution)
    if args.crop_size:
        if not isinstance(args.crop_size, tuple):
            args.crop_size=(args.crop_size, args.crop_size)
    
    if args.ssl == 'MTSL':
        args.num_classes=len(args.leisions)

    a=''
    if args.module_list!=None:
        a=list(args.module_list.values())[-1]+'_'

    # if args.ablation:
    #     val = [getattr(args, args.ablation) for a in args.ablation]
    #     val_names = ''
    #     if isinstance(val, dict):
    #         for i,(k,v) in enumerate(val.items()):
    #             val_names+=f'{k}_{v}_'
    #     else:
    #         val_names+=f'{val}_'
    #     args.id=f'{args.ablation}_{val_names}BASE_{args.base_experiment}_{args.account}_{args.a_version}'
    # else:
    args.id=f'{args.dataset_name}_{args.netframe}_{args.backbone}_{a}{args.ssl}_bs{args.batch_size}_lr{str(args.lr).split(".")[-1]}_{args.loss_method}_{args.optim}_{args.account}_{args.a_version}'
    
    return args

def create_argparser():
    # 运行前改
    defaults = dict( 
        seed=random.randint(1,10000),
        account=ACCOUNT,
    #--------------version--------------
        a_version=11,
        a_note=NOTE,
        a_goal='',
        id='',
    #消融实验
        ablation=None,
        selection=[
            {'attention':'myffs','freq_method':'dct8'},
            {'attention':'myffs','freq_method':'dct8','msfm':'dca','dam':'msfa'},
            {'msfm':'dca','dam':'msfa'},
            {'dam':'msfa'}
        ],
    #推理
        inference=False,
        inference_root_dir='/kaggle/working/inference',
        inference_suffix='VES',
    #测试
        evaluate=False,
        enable_test=0,
        test_single=0,
    #数据位置
        dataset_name='idrid',
        dataset_path='',
        preprocess=True,
    #超参数
        resolution=512,#(W,H)
        crop_size=None,
        grid_size=None,
        batch_size=4,  #有bn层，所以bs应该大于1
        lr=2e-4,
        min_lr=1e-5,
        scheduler='multi',
        milestones=[60,120,180,210],
        lr_gamma=0.5,
        step_size=200,
        upd_by_ep=1,
        accumulation_step=4,
    #模型
        leisions=['MA','HE','EX','SE'], # ['MA','HE','EX','SE']
        num_classes=5,
        backbone='rs50',
        netframe='unet',
        pretrained=True,
        downsample_factor=32,
        module_list={'attention':'myffs','freq_method':'dct8','serial_dam':'dct'},#不用时置None,使用时必须有attention
        backbone_weights=None,
        upsample_mode='interp',
        net_type='linear',
        transunet_decoder='unet',
        transformer_block='vit',
        dual='VES',
        dual_loss_factor=0.5,
        seg_vessel=True,
    #损失函数
        loss_method='bce',
        ce_weight=[0.001,1, 0.1, 0.1, 0.1],
        bce_weight=100,
        focal_gamma=2,
    #优化器设置
        optim='Adam',
        momentum=(0.9,0.999), #adam类优化器，也需要betas=(0.9,0.999),第一个类似动量
        weight_decay=0.001,
    #epoch
        num_epochs=0,
    #validation
        metrics=['auprc'],
        best_metric='auprc',
        good_value=0.6,
        validation_interval=1,
        early_stop=0,
    #-----------------------------------
    #--------------use_wandb------------
    #-----------------------------------
        use_wandb=0,
        experiment_name="DR_Segmentation",
    #log
        log_dir='./log',
        log_interval=10,
        show_log=1,
    #保存
        save_interval=120,
        skip_epoch=120,
        checkpoint_dir='./checkpoint',
        save_optim=1,
    #恢复
        resume_checkpoint='',
        load_optimizer=True,
    #Transfrom
        use_T=1,
        choose_one_of=None,
    #ssl
        ssl='MTSL',
        input_logits=True,
        task_num=4,
        gan_grading_metrics=['acc','kappa'],
        gan_grading_classes=5,
        gan_grading_dataset='idrid'
    )
    args=adjust_args(Aobj(defaults))
    return args
    
    
if __name__ == "__main__":
    main()