# nova-networks/HPA-nova
from utils import parse_args, prepare_for_result
from dataloaders import get_dataloader
from models import get_model
from losses import get_loss, get_class_balanced_weighted
from losses.regular import class_balanced_ce
from optimizers import get_optimizer
from basic_train import basic_train, basic_test, mean_average_precision
from scheduler import get_scheduler
from utils import load_matched_state
from torch.utils.tensorboard import SummaryWriter
from basic_train import tta_validate
import torch
try:
    from apex import amp
except:
    pass
import albumentations as A
from dataloaders.transform_loader import get_tfms
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    print('[ √ ] Landmark!')
    args, cfg = parse_args()
    result_path = prepare_for_result(cfg)
    writer = SummaryWriter(log_dir=result_path)
    if args.mode == 'test':
        
        df = pd.read_csv(
            'results/' + cfg.basic.id + '/train.log', sep='\t')
        if args.epoch > 0:
            best_epoch = args.epoch
        else:
            if 'loss' in args.select:
                asc = True
            else:
                asc = False
            best_epoch = int(df.sort_values(args.select, ascending=asc).iloc[0].Epochs)
            # print(f'{df.sort_values(args.select, ascending=asc)}') # Best mAP
        print('Best check we use is: {}'.format('f{}_epoch-{}.pth'.format(0, best_epoch)))
        if args.tta_tfms == 'none':
            tfms = tta_transform = A.Compose([
                                        A.OneOf([
                                            A.HorizontalFlip(p=0.5),
                                            A.VerticalFlip(p=0.5),
                                        ]),
                                    ])
        elif args.tta_tfms == 'default':
            tfms = None
        else:
            tfms = get_tfms(args.tta_tfms)
        test_dl = get_dataloader(cfg)(cfg).get_dataloader(test_only=True, tta=args.tta, tta_tfms=tfms)
        _, valid_dl, _ = get_dataloader(cfg)(cfg).get_dataloader(tta=args.tta, tta_tfms=tfms)
        # loading model
        model = get_model(cfg)
        if cfg.loss.name == 'weighted_ce_loss':
            # if we use weighted ce loss, we load the loss here.
            weights = torch.Tensor(cfg.loss.param['weight']).cuda()
            loss_func = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')
        else:
            loss_func = get_loss(cfg)
            
        model.load_state_dict(torch.load(
            Path(os.path.dirname(os.path.realpath(__file__))) / 'results' / cfg.basic.id / 'checkpoints' / 'f0_epoch-{}.pth'.format(best_epoch),
            map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu', 'cuda:2': 'cpu', 'cuda:3': 'cpu'}
        ))

        loaded_model = torch.load(
            Path(os.path.dirname(os.path.realpath(__file__))) / 'results' / cfg.basic.id / 'checkpoints' / 'f0_epoch-{}.pth'.format(best_epoch))
        # print("State_dict keys:", loaded_model.keys())
        # weight1 = loaded_model['net.conv1.weight'].cpu()
        # print(Path(os.path.dirname(os.path.realpath(__file__))) / 'results' / cfg.basic.id / 'checkpoints' / 'f0_epoch-{}.pth'.format( best_epoch))
        # print(weight1[50:60])

        # model = loaded_model.cpu()
        model = model.cpu()
        if len(cfg.basic.GPU) == 1:
            print('[ W ] single gpu prediction the gpus is {}'.format(cfg.basic.GPU))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # torch.cuda.set_device(int(cfg.basic.GPU))
            if device == 'cuda':
                torch.cuda.set_device(0)
            model = model.cuda()
        else:
            print('[ W ] dp prediction the gpus is {}'.format(cfg.basic.GPU))
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=[int(x) for x in cfg.basic.GPU])
            
        with torch.no_grad():

            if cfg.loss.name == 'weighted_ce_loss':
                # if we use weighted ce loss, we load the loss here.
                weights = torch.Tensor(cfg.loss.param['weight']).cuda()
                loss_func = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')
            else:
                loss_func = get_loss(cfg)

            
            # print(model.net.layer4['0']['conv1'])
            
            test_mAP, test_loss = basic_test(model, test_dl, loss_func, cfg, best_epoch)
            

            # best_epoch_str = 'f{}_epoch-{}.pth'.format(cfg.experiment.run_fold, best_epoch)
            # print(f'best_epoch_str {best_epoch_str}')
            print('[ √ ] Best Epoch: {}, MAP: {:.4f}, loss: {:.6f}'.format(best_epoch, test_mAP, test_loss))

            with open(result_path / 'test.log', 'a') as fp:
                    fp.write('{}\t{:.8f}\t{:.4f}\n'.format(best_epoch, test_mAP, test_loss))

        # valid_df.to_csv(Path(os.path.dirname(os.path.realpath(__file__))) / '..' / 'results' / cfg.basic.id / 'oof.csv')
        # test_df.to_csv(Path(os.path.dirname(os.path.realpath(__file__))) / '..' / 'results' / cfg.basic.id / 'test.csv')
        # rocs = []
        # for i in range(1000):
        #     s = valid_df.sample(frac=0.8).copy()
        #     rocs.append(roc_auc_score(s.target, s.predict))
        # print('SubSample 0.8, mean: {:.4f}, min: {:.4f}, max: {:.4f}, std: {:.4f}'.format(
        #     np.array(rocs).mean(), np.array(rocs).min(), np.array(rocs).max(), np.array(rocs).std())
        # )

        exit(0)
    # print(cfg)
    
    cfg.dump_json(result_path / 'config.json')

    # modify for training multiple fold
    if cfg.experiment.run_fold == -1:
        for i in range(cfg.experiment.fold):
            torch.cuda.empty_cache()
            print('[ ! ] Full fold coverage training! for fold: {}'.format(i))
            cfg.experiment.run_fold = i
            # print(f'config: {cfg}')
            train_dl, valid_dl, test_dl = get_dataloader(cfg)(cfg).get_dataloader()
            print('[ i ] The length of train_dl is {}, valid dl is {}'.format(len(train_dl), len(valid_dl)))
            # device='cuda'
            model = get_model(cfg).cuda()
            if not cfg.model.from_checkpoint == 'none':
                print('[ ! ] loading model from checkpoint: {}'.format(cfg.model.from_checkpoint))
                load_matched_state(model, torch.load(cfg.model.from_checkpoint))
                # model.load_state_dict(torch.load(cfg.model.from_checkpoint))
            if cfg.loss.name == 'weighted_ce_loss':
                # if we use weighted ce loss, we load the loss here.
                weights = torch.Tensor(cfg.loss.param['weight']).cuda()
                loss_func = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')
            else:
                loss_func = get_loss(cfg)
            optimizer = get_optimizer(model, cfg)
            print('[ i ] Model: {}, loss_func: {}, optimizer: {}'.format(cfg.model.name, cfg.loss.name,
                                                                         cfg.optimizer.name))
            if not cfg.basic.amp == 'None' and not cfg.basic.amp == 'Native':
                print('[ i ] Call apex\'s initialize')
                model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.basic.amp)
            if not cfg.scheduler.name == 'none':
                scheduler = get_scheduler(cfg, optimizer, len(train_dl))
            else:
                scheduler = None
            if len(cfg.basic.GPU) > 1:
                model = torch.nn.DataParallel(model)
            #writer.add_graph(model, input_to_model=next(iter(train_dl))[0]) # tensorboard graph visualization
            #inputs, _ = next(iter(train_dl))
            #writer.add_graph(model, inputs.to('cuda'))
            basic_train(cfg, model, train_dl, valid_dl, loss_func, optimizer, result_path, scheduler, writer)
    else:
        train_dl, valid_dl, test_dl = get_dataloader(cfg)(cfg).get_dataloader()
        print('[ i ] The length of train_dl is {}, valid dl is {}'.format(len(train_dl), len(valid_dl)))
        model = get_model(cfg).cuda()
        if not cfg.model.from_checkpoint == 'none':
            print('[ ! ] loading model from checkpoint: {}'.format(cfg.model.from_checkpoint))
            load_matched_state(model, torch.load(cfg.model.from_checkpoint, map_location='cpu'))
            # model.load_state_dict(torch.load(cfg.model.from_checkpoint))
        if cfg.loss.name == 'weighted_ce_loss':
            # if we use weighted ce loss, we load the loss here.
            weights = torch.Tensor(cfg.loss.param['weight']).cuda()
            loss_func = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')
        else:
            loss_func = get_loss(cfg)
        optimizer = get_optimizer(model, cfg)
        print('[ i ] Model: {}, loss_func: {}, optimizer: {}'.format(cfg.model.name, cfg.loss.name, cfg.optimizer.name))
        if not cfg.basic.amp == 'None' and not cfg.basic.amp == 'Native':
            print('[ i ] Call apex\'s initialize')
            model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.basic.amp)
        if not cfg.scheduler.name == 'none':
            scheduler = get_scheduler(cfg, optimizer, len(train_dl))
        else:
            scheduler = None
        if len(cfg.basic.GPU) > 1:
            model = torch.nn.DataParallel(model)
        # if cfg.train.cutmix:
        #     cutmix_train(cfg, model, train_dl, valid_dl, loss_func, optimizer, result_path, scheduler, writer)
        # elif cfg.train.mixup:
        #     mixup_train(cfg, model, train_dl, valid_dl, loss_func, optimizer, result_path, scheduler, writer)
        # else:
        #writer.add_graph(model, input_to_model=next(iter(train_dl))[0]) # tensorboard graph visualization
        #inputs, _ = next(iter(train_dl))
        #writer.add_graph(model, inputs.to('cuda'))
        basic_train(cfg, model, train_dl, valid_dl, loss_func, optimizer, result_path, scheduler, writer)
