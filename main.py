import gc
import json
import os
import time
from configparser import ConfigParser

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from callbacks import EarlyStopper
from functions import chart_cbar, compute_metrics, f1_metric, r2_metric
from loaders import (AI4ArcticChallengeDataset, AI4ArcticChallengeTestDataset,
                     get_variable_options)
from models import ResNetASPP
from utils import (CHARTS, FLOE_LOOKUP, SCENE_VARIABLES, SIC_LOOKUP,
                   SOD_LOOKUP, colour_str)


def main(config:ConfigParser):

    dir_in_train = os.path.normpath(config['io']['dir_in_train'])
    dir_in_validation = os.path.normpath(config['io']['dir_in_validation'])
    dir_out = os.path.normpath(config['io']['dir_out'])
    dataset_json_train = os.path.normpath(config['io']['dataset_json_train'])
    dataset_json_validation = os.path.normpath(config['io']['dataset_json_validation'])

    pretrained = config['model']['pretrained'] == 'True'

    gpu_id = int(config['train']['gpu_id'])
    min_epochs = int(config['train']['min_epochs'])
    max_epochs = int(config['train']['max_epochs'])
    patience = int(config['train']['patience'])
    reduce_lr_patience = int(config['train']['reduce_lr_patience'])
    reduce_lr_factor = float(config['train']['reduce_lr_factor'])
    batch_size = int(config['train']['batch_size'])
    lr = float(config['train']['lr'])

    n_samples_per_input = int(config['datamodule']['n_samples_per_input'])
    num_val_scenes = int(config['datamodule']['num_val_scenes'])
    patch_size = int(config['datamodule']['patch_size'])
    seed = int(config['datamodule']['seed'])

    fname_metric_out = os.path.join(dir_out, 'training_metrics.csv')

    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    train_options = {
        # -- Training options -- #
        'path_to_processed_data_train': dir_in_train,  
        'path_to_processed_data_validation': dir_in_validation, 
        # 'path_to_env': os.environ['AI4ARCTIC_ENV'],  # Replace with environmment directory path.
        'lr': lr,  # Optimizer learning rate.
        'epochs': max_epochs,  # Number of epochs before training stop.
        'epoch_len': None,  # Number of batches for each epoch. **This is updated below as it depends on the training file list**
        'patch_size': patch_size,  # Size of patches sampled. Used for both Width and Height.
        'batch_size': batch_size,  # Number of patches for each batch.
        'loader_upsampling': 'nearest',  # How to upscale low resolution variables to high resolution.
        
        # -- Data prepraration lookups and metrics.
        'train_variables': SCENE_VARIABLES[0:5],  # Contains the relevant variables in the scenes.

        'charts': CHARTS,  # Charts to train on.
        'n_classes': {  # number of total classes in the reference charts, including the mask.
            'SIC': SIC_LOOKUP['n_classes'],
            'SOD': SOD_LOOKUP['n_classes'],
            'FLOE': FLOE_LOOKUP['n_classes']
        },
        'pixel_spacing': 80,  # SAR pixel spacing. 80 for the ready-to-train AI4Arctic Challenge dataset.
        'train_fill_value': 0,  # Mask value for SAR training data.
        'class_fill_values': {  # Mask value for class/reference data.
            'SIC': SIC_LOOKUP['mask'],
            'SOD': SOD_LOOKUP['mask'],
            'FLOE': FLOE_LOOKUP['mask'],
        },
        
        # -- Validation options -- #
        'chart_metric': {  # Metric functions for each ice parameter and the associated weight.
            'SIC': {
                'func': r2_metric,
                'weight': 2,
            },
            'SOD': {
                'func': f1_metric,
                'weight': 2,
            },
            'FLOE': {
                'func': f1_metric,
                'weight': 1,
            },
        },

        'num_val_scenes': num_val_scenes,  # Number of scenes randomly sampled from train_list to use in validation.
        
        # -- GPU/cuda options -- #
        'gpu_id': gpu_id,  # Index of GPU. In case of multiple GPUs.
        'num_workers': os.cpu_count()-4,  # Number of parallel processes to fetch data.
        'num_workers_val': 1,  # Number of parallel processes during validation.
       
    }

    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(train_options)

    if torch.cuda.is_available():
        print('GPU available!')
        print(f'Total number of available devices: {torch.cuda.device_count()}')
        device = torch.device(f"cuda:{train_options['gpu_id']}")

    else:
        print(colour_str('GPU not available.', 'red'))
        device = torch.device('cpu')

    # Load training list.
    with open(dataset_json_train) as file:
        print('trying to load the file')
        train_options['train_list'] = json.loads(file.read())
    # Convert the original scene names to the preprocessed names.
    train_options['train_list'] = [file[17:32] + '_' + file[77:80] + '_prep.nc' for file in train_options['train_list']]
    
    # Select a random number of validation scenes with the same seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    #train_options['validate_list'] = np.random.choice(np.array(train_options['train_list']), size=train_options['num_val_scenes'], replace=False)
    # Remove the validation scenes from the train list.
    #train_options['train_list'] = [scene for scene in train_options['train_list'] if scene not in train_options['validate_list']]
    print(f'Training set contains {len(train_options["train_list"])} scenes.')
    # update epoch_len
    train_options['epoch_len'] = int(n_samples_per_input*len(train_options['train_list'])/batch_size)

    # Load validation list.
    with open(dataset_json_validation) as file:
        train_options['validate_list'] = json.loads(file.read())
    # Convert the original scene names to the preprocessed names.
    train_options['validate_list'] = [file[17:32] + '_' + file[77:80] + '_prep.nc' for file in train_options['validate_list']]
    print(f'Validation set contains {len(train_options["validate_list"])} scenes.')

    # Custom dataset and dataloader.
    dataset = AI4ArcticChallengeDataset(files=train_options['train_list'], options=train_options)
    # the __getitem__ is, in reality, a "get batch", so batch size needs to be None 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=train_options['num_workers'], pin_memory=True)
    # - Setup of the validation dataset/dataloader. 
    dataset_val = AI4ArcticChallengeTestDataset(options=train_options, files=train_options['validate_list'])
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)

    print('GPU and data setup complete.')
    
    net = ResNetASPP(pretrained=pretrained, n_classes=train_options['n_classes']).to(device)

    optimizer = torch.optim.Adam(list(net.parameters()), lr=train_options['lr'])
    torch.backends.cudnn.benchmark = True  # Selects the kernel with the best performance for the GPU and given input size.

    # Loss functions to use for each sea ice parameter.
    # The ignore_index argument discounts the masked values, ensuring that the model is not using these pixels to train on.
    # It is equivalent to multiplying the loss of the relevant masked pixel with 0.
    loss_functions = {chart: torch.nn.CrossEntropyLoss(ignore_index=train_options['class_fill_values'][chart]) \
                                                    for chart in train_options['charts']}
    print('Model setup complete')
    print(f'Parameters count: {sum(p.numel() for p in net.parameters())/1e6:.2f} M')
    print(f'Trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6:.2f} M')

    best_combined_score = 0  # Best weighted model score.
    early_stopper = EarlyStopper(patience=patience)
    scheduler = ReduceLROnPlateau(optimizer, 'min', 
                                  factor=reduce_lr_factor, 
                                  patience=reduce_lr_patience, 
                                  verbose=True, 
                                  min_lr=1e-8)

    with open(fname_metric_out, 'a') as fout:
        fout.write('epoch, train_loss, val_loss, val_sic_r2, val_sod_f1, val_floe_f1, val_combined, time \n')

    start_time = time.perf_counter()
    # -- Training Loop -- #
    for epoch in tqdm(iterable=range(train_options['epochs']), position=0):
        gc.collect()  # Collect garbage to free memory.
        loss_sum = torch.tensor([0.])  # To sum the batch losses during the epoch.
        net.train()  # Set network to evaluation mode.

        loss_sum_c = {chart: 0. for chart in train_options['charts']}
        # Loops though batches in queue.
        for i, (batch_x, batch_y) in enumerate(tqdm(iterable=dataloader, total=train_options['epoch_len'], colour='red', position=0)):
            torch.cuda.empty_cache()  # Empties the GPU cache freeing up memory.
            loss_batch = 0  # Reset from previous batch.
            loss_batch_c = {}
            
            # - Transfer to device.
            batch_x = batch_x.to(device, non_blocking=True)

            # - Mixed precision training. (Saving memory)
            with torch.cuda.amp.autocast():
                # - Forward pass. 
                output = net(batch_x)

                # - Calculate loss.
                for chart in train_options['charts']:
                    loss_batch += loss_functions[chart](input=output[chart], target=batch_y[chart].to(device))
                
                for chart in train_options['charts']:
                    loss_batch_c[chart] = loss_functions[chart](input=output[chart], target=batch_y[chart].to(device))

            # - Reset gradients from previous pass.
            optimizer.zero_grad()

            # - Backward pass.
            loss_batch.backward()

            # - Optimizer step
            optimizer.step()

            # - Add batch loss.
            loss_sum += loss_batch.detach().item()

            for chart in loss_batch_c:
                loss_sum_c[chart] += loss_batch_c[chart].detach().item()

            # - Average loss for displaying
            loss_epoch = torch.true_divide(loss_sum, i + 1).detach().item()

            del output, batch_x, batch_y # Free memory.

        tqdm.write('Mean training loss: ' + f'{loss_epoch:.3f}', end='\n')

        for chart in loss_sum_c:
            tqdm.write(f"Mean training loss for {chart}: {loss_sum_c[chart] / (i + 1):.3f}")

        # -- Validation Loop -- #
        loss_batch = loss_batch.detach().item()  # For printing after the validation loop.
        
        # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
        outputs_flat = {chart: np.array([]) for chart in train_options['charts']}
        inf_ys_flat = {chart: np.array([]) for chart in train_options['charts']}

        net.eval()  # Set network to evaluation mode.

        loss_sum_val=0
        # - Loops though scenes in queue.
        for inf_x, inf_y, masks, name in tqdm(iterable=dataloader_val, total=len(train_options['validate_list']), colour='green', position=0):
            torch.cuda.empty_cache()

            # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
            with torch.no_grad(), torch.cuda.amp.autocast():
                inf_x = inf_x.to(device, non_blocking=True)
                output = net(inf_x)

                # - Calculate loss.
                loss_batch_val = 0
                for chart in train_options['charts']:
                    loss_batch_val += loss_functions[chart](input=output[chart], target=inf_y[chart].to(device))
        
            # - Add batch loss.
            loss_sum_val += loss_batch_val.detach().item()

            # - Final output layer, and storing of non masked pixels.
            for chart in train_options['charts']:
                output[chart] = torch.argmax(output[chart], dim=1).squeeze().cpu().numpy()
                outputs_flat[chart] = np.append(outputs_flat[chart], output[chart][~masks[chart]])
                inf_ys_flat[chart] = np.append(inf_ys_flat[chart], np.squeeze(inf_y[chart], 0)[~masks[chart]].numpy())
            
            del inf_x, inf_y, masks, output  # Free memory.

        # - Average loss for displaying
        loss_epoch_val = torch.true_divide(loss_sum_val, len(dataloader_val)).detach().item()

        # - Compute the relevant scores.
        combined_score, scores = compute_metrics(true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'],
                                                metrics=train_options['chart_metric'])

        tqdm.write(f"Final batch loss: {loss_batch:.3f}", end='\n')
        tqdm.write(f"Val loss: {loss_epoch_val:.3f}", end='\n')        
        tqdm.write(f"Epoch {epoch} score:", end='\n')
        for chart in train_options['charts']:
            tqdm.write(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%", end='\n')
        tqdm.write(f"Combined score: {combined_score}%", end='\n')

        # If the scores is better than the previous epoch, then save the model and rename the image to best_validation.
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            torch.save(obj={'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch},
                            f=os.path.join(dir_out, 'best_model.ckpt'))

            tqdm.write(f'---------------saved new best model in epoch {epoch}')
        del inf_ys_flat, outputs_flat  # Free memory.

        with open(fname_metric_out, 'a') as fout:
            #fout.write('train_loss, val_loss, val_sic_r2, val_sod_f1, val_floe_f1, val_combined, time \n')
            fout.write(f'{epoch}, {loss_epoch}, {loss_epoch_val}, {scores["SIC"]}, {scores["SOD"]}, {scores["FLOE"]}, {combined_score}, {time.perf_counter()-start_time}\n')

        # callbacks:
        if early_stopper.early_stop(loss_epoch_val):
            break
        scheduler.step(loss_epoch_val)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', default='config_main.ini')
    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        config = ConfigParser()
        config.read(args.config_file)

        main(config)
    
    else:
        print('Please provide a valid configuration file.')
