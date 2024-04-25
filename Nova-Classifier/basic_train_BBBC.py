# nova-networks/HPA-nova for BBBC!!!
from utils import *
import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from configs import Config
import torch
from utils import rand_bbox
from utils.mix_methods import snapmix, cutmix, cutout, as_cutmix, mixup
from utils.metric import macro_multilabel_auc
import pickle as pk
from path import Path
from torchinfo import summary
import csv
import os
try:
    from apex import amp
except:
    pass
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import roc_auc_score
from gradcam import *
from scipy import stats


def basic_train(cfg: Config, model, train_dl, valid_dl, loss_func, optimizer, save_path, scheduler, writer, tune=None):
    print(f'[ ! ] pos weight: {1 / cfg.loss.pos_weight}')
    pos_weight = torch.ones(693).cuda()
    # summary(model=model, input_size=(1, 3, 224, 224), col_names=['input_size', 'output_size', 'num_params', 'trainable'])
    print('[ √ ] Basic training')
    if cfg.transform.size == 512:
        img_size = (600, 800)
    else:
        img_size = (cfg.transform.size, cfg.transform.size)
    try:
        optimizer.zero_grad()
        for epoch in range(cfg.train.num_epochs):
            # first we update batch sampler if exist
            if cfg.experiment.batch_sampler:
                train_dl.batch_sampler.update_miu(
                    cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor
                )
                print('[ W ] set miu to {}'.format(cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor))
            if scheduler and cfg.scheduler.name in ['StepLR']:
                scheduler.step(epoch)
            model.train()
            if not tune:
                tq = tqdm.tqdm(train_dl)
            else:
                tq = train_dl
            basic_lr = optimizer.param_groups[0]['lr']
            losses = []
            # native amp
            if cfg.basic.amp == 'Native':
                scaler = torch.cuda.amp.GradScaler()
            
            for i, (ipt, mask, lbl, cnt) in enumerate(tq):
                model.train()
#                 if i == 1:
#                     break
                ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
                mask = mask.view(-1)
                lbl = lbl.view(-1, lbl.shape[-1])
                exp_label = cnt.cuda()
                # warm up lr initial
                if cfg.scheduler.warm_up and epoch == 0:
                    # warm up
                    length = len(train_dl)
                    initial_lr = basic_lr / length
                    optimizer.param_groups[0]['lr'] = initial_lr * (i + 1)
                ipt, lbl = ipt.cuda(), lbl.cuda()
                r = np.random.rand(1)
                if cfg.train.cutmix and cfg.train.beta > 0 and r < cfg.train.cutmix_prob:
                    input, target_a, target_b, lam_a, lam_b = cutmix(ipt, lbl, img_size, cfg.train.beta, model)
                    cell, exp = model(ipt, cfg.experiment.count, gradcam=False)
                    # print(cell.shape, lam_a.shape)
                    cell_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
                    # print(cell.shape, lam_a.shape, cell_loss(cell, target_a).shape)
                    loss_cell = (cell_loss(cell, target_a).mean(1) * torch.tensor(
                        lam_a).cuda().float() +
                            cell_loss(cell, target_b).mean(1) * torch.tensor(
                                lam_b).cuda().float())
                    target_a_exp = target_a.view(-1, cfg.experiment.count, 693).mean(1)
                    target_b_exp = target_b.view(-1, cfg.experiment.count, 693).mean(1)
                    lam_a_exp = lam_a.view(-1, cfg.experiment.count).mean(1)
                    lam_b_exp = lam_b.view(-1, cfg.experiment.count).mean(1)
                    loss_exp = (loss_func(exp, target_a_exp).mean(1) * torch.tensor(
                        lam_a_exp).cuda().float() +
                            loss_func(exp, target_b_exp).mean(1) * torch.tensor(
                                lam_b_exp).cuda().float())
                    loss = (loss_cell * 0.1).mean() + loss_exp.mean()
                    # print(loss)
                    losses.append(loss.item())
                else:
                    if cfg.basic.amp == 'Native':
                        with torch.cuda.amp.autocast():
                            if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                                output = model(ipt, lbl, gradcam=False)
                            elif 'alexnet' in cfg.model.name or 'vit' in cfg.model.name: # for only cell level output
                                # print(f'ipt: {ipt.shape}') # torch.Size([100, 4, 256, 256])
                                output = model(ipt, cfg.experiment.count, gradcam=False)
                                loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')(output, lbl) # OG, try other loss
                                # loss = nn.CrossEntropyLoss(weight=pos_weight, reduction='none')(output, lbl)
                                # softmax followed by logarithm (often used in cross-entropy loss) can become unstable if the input values are too large or too small
                                loss = loss.mean()
                                if loss.item() is None:
                                    print("loss.item() NaN")
                                    # loss.item() = 0
                                # loss.item() += 0.1 # Adding loss value so it doesn't go to 0... March 23
                                losses.append(loss.item())
                            else:
                                cell, exp = model(ipt, cfg.experiment.count, gradcam=False)
                                # cell, exp = model(ipt, cfg.experiment.count, gradcam=False) for our own model efficient2
                                # print(f'cell:{cell.shape}') # cell level: batch_size*num_cells x 19
                                # print(f'exp:{exp.shape}') # image level: batch_size x 19
                                # loss = loss_func(output, lbl)
                                loss_cell = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')(cell, lbl)
                                # print(f'loss_cell:{loss_cell.shape}') # batch_size*num_cells x 19, 40x19
                                # print(f'exp:{exp.shape}')
                                # print(f'exp_label:{exp_label.shape}')
                                loss_exp = loss_func(exp, exp_label)
                                if not len(loss_cell.shape) == 0:
                                    loss_cell = loss_cell.mean()
                                if not len(loss_exp.shape) == 0:
                                    loss_exp = loss_exp.mean()
                                loss = cfg.loss.cellweight * loss_cell + (1-cfg.loss.cellweight)*loss_exp
                                losses.append(loss.item())
                    else:
                        if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                            output = model(ipt, lbl, gradcam=False)
                        else:
                            output = model(ipt, gradcam=False)
                        # loss = loss_func(output, lbl)
                        loss = loss_func(output, lbl)
                        if not len(loss.shape) == 0:
                            loss = loss.mean()
                        losses.append(loss.item())
                # cutmix ended
                # output = model(ipt)
                # loss = loss_func(output, lbl)
                if cfg.basic.amp == 'Native':
                    scaler.scale(loss).backward()
                elif not cfg.basic.amp == 'None':
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # predicted.append(output.detach().sigmoid().cpu().numpy())
                # truth.append(lbl.detach().cpu().numpy())
                if i % cfg.optimizer.step == 0:
                    if cfg.basic.amp == 'Native':
                        if cfg.train.clip:
                            scaler.unscale_(optimizer)
                            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        if cfg.train.clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                        optimizer.step()
                        optimizer.zero_grad()
                if cfg.scheduler.name in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingLR']:
                    if epoch == 0 and cfg.scheduler.warm_up:
                        pass
                    else:
                    # TODO maybe, a bug
                        scheduler.step()
                if not tune:
                    tq.set_postfix(loss=np.array(losses).mean(), lr=optimizer.param_groups[0]['lr'])
            # Validate
            if(epoch % cfg.train.validate_every == 0):
                validate_loss, accuracy, auc, mAP = basic_validate(model, valid_dl, loss_func, cfg, epoch, tune)
                # print(f'type: {type(auc)}, auc:{auc}') #list
                
                print(('[ √ ] epochs: {}, train loss: {:.4f}, valid loss: {:.4f}, ' +
                       'accuracy: {:.4f}, auc: {:.4f}, mAP: {:.4f}').format(
                    epoch, np.array(losses).mean(), validate_loss, accuracy, auc, mAP))
                writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses), epoch)
                writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('valid_f{}/loss_'.format(cfg.experiment.run_fold), validate_loss, epoch)
                writer.add_scalar('valid_f{}/mAP'.format(cfg.experiment.run_fold), mAP, epoch)
                writer.add_scalar('valid_f{}/accuracy'.format(cfg.experiment.run_fold), accuracy, epoch)
                writer.add_scalar('valid_f{}/auc'.format(cfg.experiment.run_fold), auc, epoch)
    
                with open(save_path / 'train.log', 'a') as fp:
                    fp.write('{}\t{:.8f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
                        epoch, optimizer.param_groups[0]['lr'], np.array(losses).mean(), validate_loss, accuracy, auc, mAP))
            # Continue Training
            else: # comment this, move this block (lines 195-201 out so will run after validation. 3/23 3PM
                print(('[ √ ] epochs: {}, train loss: {:.4f}').format(epoch, np.array(losses).mean()))
                writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses), epoch)
                writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'], epoch)
    
                with open(save_path / 'train.log', 'a') as fp:
                    fp.write('{}\t{:.8f}\t{:.4f}\n'.format(
                        epoch, optimizer.param_groups[0]['lr'], np.array(losses).mean()))
                
            torch.save(model.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                cfg.experiment.run_fold, epoch))
            if scheduler and cfg.scheduler.name in ['ReduceLROnPlateau']:
                scheduler.step(validate_loss)
    except KeyboardInterrupt:
        print('[ X ] Ctrl + c, QUIT')
        torch.save(model.state_dict(), save_path / 'checkpoints/quit_f{}.pth'.format(cfg.experiment.run_fold))


def basic_validate(mdl, dl, loss_func, cfg, epoch, tune=None):
    print("Validating...")
    mdl.eval()
    with torch.no_grad():
        results_img, results_cell = [], []
        losses_img, predicted_img, predicted_p_img, truth_img = [], [], [], []
        losses_cell, predicted_cell, predicted_p_cell = [], [], []
        accuracy = 0
        pos_weight = torch.ones(693).cuda()
        
        for i, (ipt, mask, lbl, cnt, n_cell) in enumerate(dl):
            ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
            lbl = lbl.view(-1, lbl.shape[-1])
            exp_label = cnt.cuda().view(-1, 693)
            ipt, lbl = ipt.cuda(), lbl.cuda()
            if cfg.basic.amp == 'Native':
                with torch.cuda.amp.autocast():
                    if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                        output = mdl(ipt, lbl, gradcam=False)
                    elif 'alexnet' in cfg.model.name or 'vit' in cfg.model.name: # for only cell level output
                        output = mdl(ipt, cfg.experiment.count, gradcam=False)
                        loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')(output, lbl)
                        # loss = nn.CrossEntropyLoss(weight=pos_weight, reduction='none')(output, lbl)
                        loss = loss.mean()
                        img_output = torch.mean(output,dim=0).float() # commented out, 3/20, debugging NaN
                        # img_output = output.float()
                        cell_output = output.float()
                    else:
                        cell_output, img_output = mdl(ipt, n_cell,gradcam=False)
                        # cell_output, img_output = mdl(ipt, n_cell, gradcam=False)
                        loss_img_bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')(cell_output, lbl)
                        loss_img_exp = loss_func(img_output, exp_label)
                        loss = (1-cfg.loss.cellweight)*loss_img_exp + cfg.loss.cellweight*loss_img_bce # shape: 10x19
                        if not len(loss.shape) == 0:
                            loss = loss.mean()
                        img_output = img_output.float() # 1x19
                        cell_output = cell_output.float() # 10x19
            else:
                if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                    output = mdl(ipt, lbl, gradcam=False)
                else:
                    output = mdl(ipt, gradcam=False)
                loss = loss_func(output, exp_label)
                if not len(loss.shape) == 0:
                    loss = loss.mean()
            losses_img.append(loss.item())
            predicted_img.append(torch.sigmoid(img_output.cpu()).numpy()) # should be 6x19
            predicted_cell.append(torch.sigmoid(cell_output.cpu()).numpy()) # should be 60x19
            truth_img.append(lbl[0].cpu().numpy())
            results_img.append({
                'step': i,
                'loss': loss.item(),
            })
        # print(f'predicted_img: {np.array(predicted_img).shape},predicted_cell {np.array(predicted_cell).shape}') 
        predicted_img = np.array(predicted_img)
        predicted_cell = np.array(predicted_cell)
        # print(f'predicted_img: {predicted_img.shape}') # 12x1x19
        print(f'predicted_cell: {predicted_cell.shape}') # 12x10x19

        predicted = np.zeros((len(dl), cfg.experiment.count, 693))
        # print(f'truth_img: {len(truth_img)}, {len(truth_img[0])}') # 6,19
        truth = np.array(truth_img)
        print(f'predicted: {predicted.shape}, truth: {truth.shape}')
        roc_values = []
        mAP = []
        for j in range(len(dl)):
            for k in range(cfg.experiment.num_cells):
                predicted[j][k] = cfg.experiment.img_weight*np.array(predicted_img[j]) + (1-cfg.experiment.img_weight)*predicted_cell[j][k]
            
            val_loss_img = np.array(losses_img).mean()
            val_loss_cell = np.array(losses_cell).mean()
            truth_acc = np.tile(truth[j], (cfg.experiment.count, 1)) # 10x19
            accuracy += ((predicted[j] > 0.8) == truth_acc).sum().astype(np.float64) / truth_acc.shape[0] / truth_acc.shape[1]
            
            predicted_auc = np.mean(predicted[j], axis=0).flatten()
            truth_auc = truth[j].flatten() # 10x693
            # print(f'predicted_auc: {predicted_auc.shape}, {predicted_auc}') # 6x19
            roc_values.append(roc_auc_score(truth_auc, predicted_auc)) # Causing NaN errors for ViT
            # print(f'roc_values: {roc_values}')

            # mAP
            mAP_img = mean_average_precision(truth_acc, predicted[j])
            mAP.append(mAP_img)

        auc = sum(roc_values)/len(roc_values)
        accuracy /= len(dl)
        predicted = np.round(predicted, decimals=4)
        truth = np.round(truth, decimals=4)
        mAP = sum(mAP)/len(dl)

        # Image IDs
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        csv_file_path = f'{path}/dataloaders/split/{cfg.experiment.csv_valid}'
        # Open the CSV file and read the first column into a list
        with open(csv_file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            row_ids = [[row[0], row[1]] for row in csv_reader]
        row_ids = row_ids[1:]        
        # print(f'row_ids: {row_ids}')
        predicted = predicted.reshape((len(dl)*n_cell, 693)) # now, num_valid*num_cellsx19 

        # Add a new column with string IDs at index 0
        predicted_with_ids, truth_with_ids = [], []
        for img in range(len(dl)):
            start = img*cfg.experiment.count
            end = start + cfg.experiment.count
            predicted_with_ids.append(np.column_stack((np.broadcast_to(np.array(row_ids[img]), (cfg.experiment.count, 2)).tolist(), predicted[start:end, :])))
            truth_with_ids.append(np.column_stack((np.broadcast_to(row_ids[img], (1, 2)), np.repeat(np.expand_dims(truth[img], 0), 1, axis=0))))
            # truth_with_ids.append(np.column_stack((row_ids[img], np.expand_dims(truth[img],0))))

        predicted_with_ids = np.array(predicted_with_ids).reshape((len(dl)*n_cell, 695)) 
        truth_with_ids = np.array(truth_with_ids).reshape((len(dl), 695)) 
        
        # Create header with 'ID' added at the beginning
        header = ','.join(['Metadata_Plate']+ ['Metadata_Well'] + [f'class{i}' for i in range(truth.shape[1])])
        # + ['Metadata_Well'] + ['Metadata_broad_sample']
        
        # Get the directory and file paths
        base_path = Path(os.path.dirname(os.path.realpath(__file__))) / 'results' / cfg.basic.id
        pred_path = base_path / 'pred.csv'
        truth_path = base_path / 'truth.csv'
        
        # Save with truncated values and the new ID column
        np.savetxt(pred_path, predicted_with_ids, fmt='%s', delimiter=',', header=header, comments='')
        np.savetxt(truth_path, truth_with_ids, fmt='%s', delimiter=',', header=header, comments='')

        return val_loss_img, accuracy, auc, mAP


def basic_test(mdl, dl, loss_func, cfg, epoch, tune=None):
    print("Testing...")
    mdl.eval()
    with torch.no_grad():
        results_img, results_cell = [], []
        losses_img, predicted_img, predicted_p_img, truth_img = [], [], [], []
        losses_cell, predicted_cell, predicted_p_cell = [], [], []
        accuracy = 0
        pos_weight = torch.ones(693).cuda()

        for i, (ipt, mask, lbl, cnt, n_cell) in enumerate(dl):
            ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
            lbl = lbl.view(-1, lbl.shape[-1])
            exp_label = cnt.cuda().view(-1, 693)
            ipt, lbl = ipt.cuda(), lbl.cuda()
            if cfg.basic.amp == 'Native':
                with torch.cuda.amp.autocast():
                    if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                        output = mdl(ipt, lbl, gradcam=False)
                    elif 'alexnet' in cfg.model.name or 'vit' in cfg.model.name: # for only cell level output
                        output = mdl(ipt, cfg.experiment.count, gradcam=False)
                        loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')(output, lbl)
                        loss = loss.mean()
                        img_output = torch.mean(output,dim=0).float()
                        cell_output = output.float()
                    else:
                        cell_output, img_output = mdl(ipt, n_cell, gradcam=False)
                        loss_img_bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')(cell_output, lbl)
                        loss_img_exp = loss_func(img_output, exp_label)
                        loss = (1-cfg.loss.cellweight)*loss_img_exp + cfg.loss.cellweight*loss_img_bce # shape: 10x19
                        if not len(loss.shape) == 0:
                            loss = loss.mean()
                        img_output = img_output.float() # 1x19
                        cell_output = cell_output.float() # 10x19
            else:
                if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                    output = mdl(ipt, lbl, gradcam=False)
                else:
                    output = mdl(ipt, gradcam=False)
                loss = loss_func(output, exp_label)
                if not len(loss.shape) == 0:
                    loss = loss.mean()
            losses_img.append(loss.item())
            predicted_img.append(torch.sigmoid(img_output.cpu()).numpy()) # should be 6x19
            predicted_cell.append(torch.sigmoid(cell_output.cpu()).numpy()) # should be 60x19
            truth_img.append(lbl[0].cpu().numpy())
            results_img.append({
                'step': i,
                'loss': loss.item(),
            })
        predicted_img = np.array(predicted_img)
        predicted_cell = np.array(predicted_cell)

        predicted = np.zeros((len(dl),cfg.experiment.count,693))
        truth = np.array(truth_img)
        roc_values = []
        mAP = []
        p = 0
        
        for j in range(len(dl)):
            for k in range(cfg.experiment.num_cells):
                predicted[j][k] = cfg.experiment.img_weight*np.array(predicted_img[j]) + (1-cfg.experiment.img_weight)*predicted_cell[j][k]
            
            val_loss_img = np.array(losses_img).mean()
            # val_loss_cell = np.array(losses_cell).mean()
            truth_acc = np.tile(truth[j], (cfg.experiment.count, 1)) # 10x19
            accuracy += ((predicted[j] > 0.7) == truth_acc).sum().astype(np.float64) / truth_acc.shape[0] / truth_acc.shape[1]
            
            predicted_auc = np.mean(predicted[j], axis=0).flatten()
            truth_auc = truth[j].flatten() # 10x19
            roc_values.append(roc_auc_score(truth_auc, predicted_auc))
            
            # mAP
            mAP_img = mean_average_precision(truth_acc, predicted[j])
            mAP.append(mAP_img)
        
        auc = sum(roc_values)/len(roc_values)
        accuracy /= len(dl)
        predicted = np.round(predicted, decimals=4)
        truth = np.round(truth, decimals=4)
        mAP = sum(mAP)/len(dl)
        p_acc = p/len(dl)/cfg.experiment.num_cells

        
        # GradCAM
        # print('[!] Computing Grad-CAM in test mode...')
        # # extract first image, first cell from validation set
        # path = Path(os.path.dirname(os.path.realpath(__file__)))
        # csv_file_path = f'{path}/dataloaders/split/{cfg.experiment.csv_test}'
        # # Open the CSV file and read the first column into a list
        # with open(csv_file_path, 'r') as csvfile:
        #     csv_reader = csv.reader(csvfile)
        #     for idx, row in enumerate(csv_reader):
        #         row_id = row[0]
        #         if(idx==2):
        #             break
        # # print(f'row_id: {row_id}')
        # grad_img = f'../{cfg.data.dir}/{row_id}_cell1.png' # this cell is good!
        # print(f'grad_img: {grad_img}')
        # gradcam_instance = novaGradCAM(mdl, grad_img, 256)
        # image, image_prep = gradcam_instance.load()
        # gradcam_instance.make_gradcam(image, image_prep, cfg.basic.id, 'vit' in cfg.model.name)
        # # gradcam_instance.visualize(image, heatmap)
    
        
        return mAP, auc, val_loss_img

def tta_validate(mdl, dl, loss_func, tta):
    mdl.eval()
    with torch.no_grad():
        results = []
        losses, predicted, truth = [], [], []
        tq = tqdm.tqdm(dl)
        for i, (ipt, lbl) in enumerate(tq):
            ipt = [x.cuda() for x in ipt]
            lbl = lbl.cuda().long()
            output = mdl(*ipt)
            loss = loss_func(output, lbl)
            losses.append(loss.item())
            predicted.append(output.cpu().numpy())
            truth.append(lbl.cpu().numpy())
            # loss, gra, vow, con = loss_func(output, GRAPHEME, VOWEL, CONSONANT)
            results.append({
                'step': i,
                'loss': loss.item(),
            })
        predicted = np.concatenate(predicted)
        length = dl.dataset.df.shape[0]
        res = np.zeros_like(predicted[:length, :])
        for i in range(tta):
            res += predicted[i * length: (i + 1) * length]
        res = res / length
        pred = torch.softmax(torch.tensor(res), 1).argmax(1).numpy()
        tru = np.concatenate(truth)[:length]
        val_loss, val_kappa = (np.array(losses).mean(),
                               cohen_kappa_score(tru, pred, weights='quadratic'))
        print('Validation: loss: {:.4f}, kappa: {:.4f}'.format(
            val_loss, val_kappa
        ))
        df = dl.dataset.df.reset_index().drop('index', 1).copy()
        df['prediction'] = pred
        df['truth'] = tru
        return val_loss, val_kappa, df


def average_precision(y_true, y_pred):
    """
    Calculate average precision for a single sample.

    Parameters:
        y_true (1D array): True labels.
        y_pred (1D array): Predicted labels.

    Returns:
        float: Average Precision (AP) for the sample.
    """
    num_true = np.sum(y_true)
    if num_true == 0:
        return 0.0

    sorted_indices = np.argsort(y_pred)[::-1]
    precision = 0.0
    num_correct = 0.0

    for i, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            num_correct += 1
            precision += num_correct / (i + 1)

    return precision / num_true

def mean_average_precision(y_true, y_pred):
    """
    Calculate Mean Average Precision (mAP) for a multilabel setting.

    Parameters:
        y_true (2D array): True labels.
        y_pred (2D array): Predicted labels.

    Returns:
        float: Mean Average Precision (mAP) across all samples.
    """
    num_samples, num_labels = y_true.shape
    total_ap = 0.0

    for i in range(num_samples):
        total_ap += average_precision(y_true[i], y_pred[i])

    return total_ap / num_samples


def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]

        precision = precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls
