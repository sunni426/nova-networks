# nova-networks/HPA-nova
from utils import *
import tqdm
import pandas as pd
from sklearn.metrics import recall_score
from configs import Config
import torch
from utils import rand_bbox
from utils.mix_methods import snapmix, cutmix, cutout, as_cutmix, mixup
from utils.metric import macro_multilabel_auc
from sklearn.metrics import roc_auc_score, roc_curve
import pickle as pk
from path import Path
from PIL import Image
import csv
import os
try:
    from apex import amp
except:
    pass
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import roc_auc_score


def basic_train(cfg: Config, model, train_dl, valid_dl, loss_func, optimizer, save_path, scheduler, writer, tune=None):
    print(f'[ ! ] pos weight: {1 / cfg.loss.pos_weight}')
    pos_weight = torch.ones(19).cuda() / cfg.loss.pos_weight
    pos_weight = torch.ones(19) / cfg.loss.pos_weight
    print('[ √ ] Basic training')
    if cfg.transform.size == 512:
        img_size = (600, 800)
    else:
        img_size = (cfg.transform.size, cfg.transform.size)
    try:
        optimizer.zero_grad()
        for epoch in range(cfg.train.num_epochs):
            print("Training epoch...")
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
            # print(f'\ntrain_dl: {train_dl}')
            basic_lr = optimizer.param_groups[0]['lr']
            losses = []
            # native amp
            if cfg.basic.amp == 'Native':
                scaler = torch.cuda.amp.GradScaler()
            # print(f'tq: {tq} of length {len(tq)}')
            output_list = []
            for i, (ipt, mask, lbl, cnt) in enumerate(tq):
#                 if i == 1:
#                     break
                ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
                mask = mask.view(-1)
                lbl = lbl.view(-1, lbl.shape[-1])
                exp_label = cnt.cuda()
                # print(cnt.shape)
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
                    cell, exp = model(ipt, cfg.experiment.count)
                    # print(cell.shape, lam_a.shape)
                    cell_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
                    # print(cell.shape, lam_a.shape, cell_loss(cell, target_a).shape)
                    loss_cell = (cell_loss(cell, target_a).mean(1) * torch.tensor(
                        lam_a).cuda().float() +
                            cell_loss(cell, target_b).mean(1) * torch.tensor(
                                lam_b).cuda().float())
                    target_a_exp = target_a.view(-1, cfg.experiment.count, 19).mean(1)
                    target_b_exp = target_b.view(-1, cfg.experiment.count, 19).mean(1)
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
                                output = model(ipt, lbl)
                            else:
                                cell, exp = model(ipt, cfg.experiment.count)
                            # loss = loss_func(output, lbl)
                            loss_cell, loss_exp, loss_tot = 0, 0, 0
                            for cell_idx in range(cfg.experiment.num_cells):
                                loss_cell = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.cuda(), reduction='none')(cell[cell_idx].cuda(), lbl[i].cuda())
                                loss_exp = loss_func(exp[cell_idx], exp_label[i])
                                if not len(loss_cell.shape) == 0:
                                    loss_cell = loss_cell.mean()
                                if not len(loss_exp.shape) == 0:
                                    loss_exp = loss_exp.mean()
                                loss = cfg.loss.cellweight * loss_cell + loss_exp
                                loss_tot += loss.item()
                            losses.append(loss_tot)
                            # output = model(ipt)#, lbl)
                            # Added 1/23
                            # output_list.append(output)
                    else:
                        if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                            output = model(ipt, lbl)
                        else:
                            output = model(ipt)
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
                validate_loss, accuracy, auc = basic_validate(model, valid_dl, output_list, loss_func, cfg, epoch, tune)
                # print(f'type: {type(auc)}, auc:{auc}') #list
                
                
                print(('[ √ ] epochs: {}, train loss: {:.4f}, valid img loss: {:.4f}, ' +
                       'valid cell loss: {:.4f}, accuracy: {:.4f}, auc: {:.4f}').format(
                    epoch, np.array(losses).mean(), validate_loss[0], validate_loss[1], accuracy, auc[0]))
                writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses), epoch)
                writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('valid_f{}/loss_img'.format(cfg.experiment.run_fold), validate_loss[0], epoch)
                writer.add_scalar('valid_f{}/loss_cell'.format(cfg.experiment.run_fold), validate_loss[1], epoch)
                writer.add_scalar('valid_f{}/accuracy'.format(cfg.experiment.run_fold), accuracy, epoch)
                writer.add_scalar('valid_f{}/auc'.format(cfg.experiment.run_fold), auc[0], epoch)
    
                with open(save_path / 'train.log', 'a') as fp:
                    fp.write('{}\t{:.8f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
                        epoch, optimizer.param_groups[0]['lr'], np.array(losses).mean(), validate_loss[0], validate_loss[1], accuracy, auc[0]))
            # Continue Training
            else:
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


def basic_validate(mdl, dl, output, loss_func, cfg, epoch, tune=None):
    print("Validating...")
    mdl.eval()
    with torch.no_grad():
        results_img, results_cell = [], []
        losses_img, predicted_img, predicted_p_img, truth_img = [], [], [], []
        losses_cell, predicted_cell, predicted_p_cell = [], [], []
        
        for i, (ipt, mask, lbl, cnt, n_img) in enumerate(dl):
            ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
            lbl = lbl.view(-1, lbl.shape[-1])
            exp_label = cnt.cuda().view(-1, 19)
            ipt, lbl = ipt.cuda(), lbl.cuda()
            # Image_level
            if cfg.basic.amp == 'Native':
                    with torch.cuda.amp.autocast():
                        if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                            output = mdl(ipt, lbl)
                        else:
                            _, output = mdl(ipt, cfg.experiment.num_cells)
                            print(f'output img level: {output.shape}')
                        loss = loss_func(output, exp_label)
                        # print(f'output is {output} with loss {loss}')
                        if not len(loss.shape) == 0:
                            loss = loss.mean()
                        output = output.float()
            else:
                if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                    output = mdl(ipt, lbl)
                else:
                    output = mdl(ipt)
                loss = loss_func(output, exp_label)
                print('Loss:',loss)
                if not len(loss.shape) == 0:
                    loss = loss.mean()

            # print(f'ipt: {ipt.shape}, output: {output.shape}')
            losses_img.append(loss.item())
            predicted_img.append(torch.sigmoid(output.cpu()).numpy())
            truth_img.append(lbl.cpu().numpy())
            results_img.append({
                'step': i,
                'loss': loss.item(),
            })
            
            # Cell_level
            for cell_idx in range(cfg.experiment.num_cells):
                if cfg.basic.amp == 'Native':
                        with torch.cuda.amp.autocast():
                            if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                                output = mdl(ipt, lbl)
                            else:
                                _, output = mdl(ipt[cell_idx].unsqueeze(0), 1)
                            loss = loss_func(output, exp_label)
                            # print(f'output is {output} with loss {loss}')
                            if not len(loss.shape) == 0:
                                loss = loss.mean()
                            output = output.float()
                else:
                    if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                        output = mdl(ipt, lbl)
                    else:
                        output = mdl(ipt)
                    loss = loss_func(output, exp_label)
                    print('Loss:',loss)
                    if not len(loss.shape) == 0:
                        loss = loss.mean()
                    
                # print(f'ipt: {ipt.shape}, output: {output.shape}')
                losses_cell.append(loss.item())
                predicted_cell.append(torch.sigmoid(output.cpu()).numpy())
                results_cell.append({
                    'step': i,
                    'loss': loss.item(),
                })
        # print(f'predicted_img: {predicted_img.shape}') # 1x19
        # print(f'predicted_cell: {predicted_cell.shape}') # 10x19

        predicted = np.zeros((10,19))
        # print(f'predicted: {predicted.shape}')
        # print(f'predicted_img: {predicted_img.shape}, {predicted_img}')
        for j in range(cfg.experiment.num_cells):
            predicted[j] = cfg.experiment.img_weight*np.array(predicted_img) + (1-cfg.experiment.img_weight)*predicted_cell[j]
        # predicted = np.concatenate(predicted)
        truth = np.concatenate(truth_img)
        # print(f'predicted {predicted.shape}, truth: {truth.shape}, loss: {len(losses)}')
        val_loss_img = np.array(losses_img).mean()
        val_loss_cell = np.array(losses_cell).mean()
        
        accuracy = ((predicted > 0.5) == truth).sum().astype(np.float64) / truth.shape[0] / truth.shape[1]
        accuracy /= 10
        
        predicted_auc = np.mean(predicted, axis=0).flatten()
        truth_auc = truth.flatten()
        print(f'predicted_auc: {predicted_auc.shape}')
        roc_values = []
        roc_values.append(roc_auc_score(truth_auc, predicted_auc))
        
        # auc = macro_multilabel_auc(truth, predicted, gpu=0) #OG
        auc_list = []

        predicted = np.round(predicted, decimals=4)
        truth = np.round(truth, decimals=4)
        
        # Image IDs
        csv_file_path = '/projectnb/btec-design3/novanetworks/nova-networks/HPA-nova/dataloaders/split/valid_sunni.csv'
        # Open the CSV file and read the first column into a list
        with open(csv_file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            row_ids = [row[0] for row in csv_reader]
        row_ids = row_ids[1:]        
        print(f'row_ids: {row_ids}')
# row_ids = np.array(['ID1', 'ID2', 'ID3', 'ID4', 'ID5','ID1', 'ID2', 'ID3', 'ID4', 'ID5'])

# broadcasted_array = np.array(scalar_list)  # Convert the list to a NumPy array
# broadcasted_array = np.broadcast_to(broadcasted_array, (10, 1))

        # Add a new column with string IDs at index 0
        predicted_with_ids = np.column_stack((np.broadcast_to(np.array(row_ids), (10, 1)).tolist() , predicted))
        truth_with_ids = np.column_stack((row_ids[0], truth))
        
        # Create header with 'ID' added at the beginning
        header = ','.join(['ID'] + [f'class{i}' for i in range(truth.shape[1])])
        
        # Get the directory and file paths
        base_path = Path(os.path.dirname(os.path.realpath(__file__))) / 'results' / cfg.basic.id
        pred_path = base_path / 'pred.csv'
        truth_path = base_path / 'truth.csv'
        
        # Save with truncated values and the new ID column
        np.savetxt(pred_path, predicted_with_ids, fmt='%s', delimiter=',', header=header, comments='')
        np.savetxt(truth_path, truth_with_ids, fmt='%s', delimiter=',', header=header, comments='')

        # predicted = np.round(predicted, decimals=4)
        # truth = np.round(truth, decimals=4)
        # header = ','.join([f'class{i}' for i in range(truth.shape[1])])
        # pred_path = (Path(os.path.dirname(os.path.realpath(__file__))) / 'results' / cfg.basic.id / f'pred.csv')
        # truth_path = (Path(os.path.dirname(os.path.realpath(__file__))) / 'results' / cfg.basic.id / f'truth.csv')
        
        # # Save with truncated values
        # np.savetxt(pred_path, predicted, fmt='%.4f', delimiter=',', header=header)
        # np.savetxt(truth_path, truth, fmt='%.4f', delimiter=',', header=header)

            # Saving Results (as PNG)
            # p_path = Path(os.path.dirname(os.path.realpath(__file__))) / 'results' / cfg.basic.id / f'predicted{i}.png'
            # # t_path = Path(os.path.dirname(os.path.realpath(__file__))) / 'results' / cfg.basic.id / f'truth{i}.png'
 
            # # Scale the values to the range [0, 255] (assuming it's an image tensor)
            # output = output_list[i].np()
            # output = ((output - output.min()) / (output.max() - output.min())) * 255
            # # truth = ((truth - truth.min()) / (truth.max() - truth.min())) * 255
            
            # # Convert the NumPy array to an unsigned 8-bit integer array
            # predicted_img = output.astype(np.uint8)
            # # truth_img = truth.astype(np.uint8)
            
            # # Create an image from the uint8 array using Pillow
            # predicted_img = Image.fromarray(predicted_img)
            # # truth_img = Image.fromarray(truth_img)
            
            # # Save the image as a PNG file
            # predicted_img.save(p_path)
            # # truth_img.save(t_path)
            
        auc = np.mean(auc_list)
        val_loss = [val_loss_img, val_loss_cell]
        return val_loss, accuracy, roc_values


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
