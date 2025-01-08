import sys
import os
import numpy as np
import pandas as pd
import random
import argparse
import torch
import torchvision
import timm
import time
import datetime

print('\n[VERSION]')
print('python:', sys.version.replace('\n', ''), '(expected : 3.8.10)')
#pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
print('torch:', torch.__version__, '(expected : 1.12.1)') 
print('torchvision:', torchvision.__version__, '(expected : 0.13.1)')
print('timm:', timm.__version__, '(expected : 0.8.1dev0)')
#requirements for GTX 3070 > cu11.1
print('cuda:', torch.version.cuda) 
print('cuda name:', torch.cuda.get_device_name('cuda:0'), end='\n\n')

from torchvision.transforms import Resize
from skimage.transform import resize
from tqdm import tqdm

from datasets import get_dataset, XAIDataset
from models import get_model
from methods import get_method
from metrics import get_results
from metrics import metric_types

sys.path.append('Quantus')
import quantus

parser = argparse.ArgumentParser(description='Generate xai maps')

parser.add_argument('--dataset_name',   type=str, default='imagenet',                       help='dataset name')
parser.add_argument('--dataset_root',   type=str, default='.',                              help='root folder for all datasets. Complete used path is `dataset_root/dataset_name`')
parser.add_argument('--model',          type=str, default='vit_b16',                        help='model architecture')
parser.add_argument('--method',         type=str, default='gradcam',                        help='xai method')
parser.add_argument('--baseline',       type=str, default='',                               help='Indicates the type of baseline: mean, random, uniform, black or white, "" use default by metric if not specified')
parser.add_argument('--csv_folder',     type=str, default='csv',                            help='csv folder to save metrics')
parser.add_argument('--npz_folder',     type=str, default='npz',                            help='npz folder to save or load xai maps id required')
parser.add_argument('--save_npz',       dest='save_npz', action='store_true',               help='save xai maps in a npz file')
parser.add_argument('--npz_checkpoint', type=str, default='',                               help='use this option to load a checkpoint npz for metric computation, skip map computation if used')
parser.add_argument('--skip_metrics',   dest='skip_metrics', action='store_true',           help='skip metrics computation, useful to just produce the maps without metrics')
parser.add_argument('--gpu',            dest='gpu', action='store_true',                    help='use gpu (default)')
parser.add_argument('--cpu',            dest='gpu', action='store_false',                   help='use cpu instead of gpu')
parser.add_argument('--seed',           type=int, default=123456,                           help='Random seed')
parser.add_argument('--limit_val',      type=int, default=0,                                help='Limit validation size. Default to 0 => no limitation')
parser.add_argument('--batch_size',     type=int, default=1,                                help='Batch size')
parser.add_argument('--start_idx',      type=int, default=0,                                help='Starting index for subset metric computation')
parser.add_argument('--end_idx',        type=int, default=0,                                help='Stop index for subset metric computation')
parser.add_argument('--metrics',        type=str, default='rollout',                   help='metrics used for benchmarking')

parser.set_defaults(save_npz=False)
parser.set_defaults(skip_metrics=False)
parser.set_defaults(gpu=True)

def main():

    print('[XAI]')
    global args
    args = parser.parse_args()
    batch_size = args.batch_size
    global XAI_method

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.cuda.empty_cache()

    # Get dataset
    dataset, n_output = get_dataset(args.dataset_name, args.dataset_root)
    global upsampling_fn 
    upsampling_fn = Resize(dataset[0][0].shape[-2:], antialias=True)

    # Get model
    model = get_model(args.model, n_output, dataset=args.dataset_name)
    model = model.eval()
    print('Dataset loaded in eval mode.')

    # Use GPU
    if args.gpu:
        model = model.cuda()
        print('Model loaded on GPU.')
    
    # Get method if no checkpoint provided
    if args.npz_checkpoint:
        method = None
    else:
        method=get_method(name=args.method, model=model, batch_size=args.batch_size, dataset_name=args.dataset_name)
        print('Method loaded.')

    # Limit validation size if required in arguments (mostly for debugging purpose)
    if args.limit_val != 0:
        subset_indices  = np.random.randint(0, high=(len(dataset)-1), size=min(args.limit_val, len(dataset)))
        subset = torch.utils.data.Subset(dataset, subset_indices)
        print(f'Dataset limited to {args.limit_val} images.')
    else:
        subset = dataset

    # Get dataloader for generating the maps
    val_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle = False)

    scores = []
    saliencies_maps = []

    # Load precomputed maps if a checkpoint is specified, generate them otherwise
    start_saliency = time.time()

    if args.npz_checkpoint:
        saliencies_maps = torch.tensor(np.load(os.path.join(args.npz_folder, args.npz_checkpoint))['arr_0'])
        print('Method checkpoint loaded.')

    else:
        for X, y in tqdm(val_loader, total=len(val_loader), desc=f'Generating saliency maps using {args.method}'):
            if args.gpu:
                X = X.cuda()
                y = y.cuda()

            # One image at a time since some methods process each image multiple times using internal batches
            for i in range(X.shape[0]):
                # generate saliency map depending on the choosen method (sum over channels for gradient methods)
                if args.method in ['scorecam', 'gradcam', 'gradcam++']:
                    saliency_map = method.attribute(X[i].unsqueeze(0), target=y[i])
                    saliency_map = saliency_map.reshape((1, *saliency_map.shape))
                    saliency_map = torch.tensor(saliency_map)
                elif args.method in ['lime', 'inputgrad', 'integratedgrad', 'smoothgrad', 'occlusion']:
                    saliency_map = method.attribute(X[i].unsqueeze(0), target=y[i]).sum(1)
                    saliency_map = torch.tensor(saliency_map)
                else:
                    saliency_map = method.attribute(X[i].unsqueeze(0), target=y[i]) #saliency_map.shape = [14, 14]
                    saliency_map = saliency_map.reshape((1, *saliency_map.shape)) #saliency_map.shape = [1, 14, 14]
                    if saliency_map.shape != X[i].shape:
                        saliency_map = upsampling_fn(saliency_map) #saliency_map.shape = [1, 224, 224]
                saliencies_maps.append(saliency_map)

        # Convert the list of maps into one tensor
        saliencies_maps = torch.stack(saliencies_maps) #saliency_maps.shape = [num_images, 1, 224, 224]
        
        end_saliency = time.time()

        # Save the maps into a npz file if required
        if args.save_npz:
            npz_name = args.method + '_' + args.model + '_' + args.dataset_name
            np.savez(os.path.join(args.npz_folder, npz_name), saliencies_maps.cpu().numpy())
            
            with open(f'npz/{args.method}.txt', 'w') as f:
                f.write(f'{datetime.timedelta(seconds=(end_saliency-start_saliency)/2000)}')
            
            print('Saliency maps saved to npz.')

    print()

    # Create a XAI dataset and loader. Useful to get the image with the corresponding map
    xai_dataset = XAIDataset(subset, saliencies_maps)

    if args.end_idx != 0 and args.end_idx > args.start_idx:
        xai_dataset = torch.utils.data.Subset(xai_dataset, list(range(args.start_idx, args.end_idx)))
        subset_suffix = '_' + str(args.start_idx) + '_' + str(args.end_idx)
    else:
        subset_suffix = ''
    xai_loader = torch.utils.data.DataLoader(xai_dataset, batch_size=batch_size, shuffle = False)

    #Defining XAI_method for robustness and randomisation
    if args.metrics in metric_types['robustness'] or args.metrics in metric_types['randomisation']:
        XAI_method = get_method(args.method, model, dataset_name=args.dataset_name)

    # Compute metrics or skip it if required (in case of only generation)
    if not args.skip_metrics:
        # Perturbation baseline choose, this change the default baseline for metrics using perturb_baseline parameter
        if args.baseline == '':
            perturb_baseline = None
            csv_baseline_suffix = ''
        else:
            perturb_baseline = args.baseline
            csv_baseline_suffix= '_baseline_' + perturb_baseline

        start_metric = time.time()

        for (X, y), A in tqdm(xai_loader, desc=f'Computing metrics using {args.metrics}'):
            device = 'cuda' if args.gpu else 'cpu'
            
            if args.npz_checkpoint in ['btt_vit_b16_imagenet.npz', 'bth_vit_b16_imagenet.npz', 'tam_vit_b16_imagenet.npz']:
                A = torch.tensor(A)
                A = A.reshape((1, *A.shape))
                A = upsampling_fn(A)
    
            try:
                scores_saliency = get_results(model,
                                            name = args.metrics,
                                            x_batch = X,
                                            y_batch = y,
                                            a_batch = A,
                                            perturb_baseline = perturb_baseline,
                                            device = device,
                                            xai_method = XAI_for_Quantus)
                scores.append(scores_saliency)
            except:
                print('ouch')
                pass
            

        # Stack results by batches if the results are dict, else concatenate them by images
        if isinstance(scores[0], dict):
            scores = np.stack(scores)
        else:
            scores = np.concatenate(scores)

        end_metric = time.time()

        # save metrics in csv files
        scores_df = pd.DataFrame(data=scores, index=None, columns=None)
        if args.npz_checkpoint:
            csv_name = args.npz_checkpoint.split('/')[-1].split('.')[0] + '_' + args.metrics + csv_baseline_suffix + subset_suffix + '.csv'
        else:
            csv_name = args.method + '_' + args.model + '_' + args.dataset_name + '_' + args.metrics + csv_baseline_suffix + subset_suffix + '.csv'
        scores_df.to_csv(os.path.join(args.csv_folder, csv_name), header=False, index=False)

        with open(os.path.join('results', csv_name), 'w') as f:
            f.write(f'{datetime.timedelta(seconds=(end_metric-start_metric)/2000)}')

def XAI_for_Quantus(model, inputs, targets, device, batch_size=1, img_shape = [3, 224,224], **kwargs):

    list_maps = []

    if args.dataset_name=='imagenet':
        img_shape = [3, 224, 224]

    X = torch.tensor(inputs.reshape([batch_size] + img_shape), dtype=torch.float).to(device)
    y = torch.tensor(np.array(targets).reshape((batch_size,))).to(device)

    torch.cuda.empty_cache()

    for i in range(batch_size):
        
        if XAI_method in ['smoothgrad', 'gradcam']:
            saliency_map = XAI_method.attribute(X[i].unsqueeze(0), target=y[i]).sum(1)
        else:
            saliency_map = XAI_method.attribute(X[i].unsqueeze(0), target=y[i])
            saliency_map = saliency_map.reshape((1, *saliency_map.shape))
            saliency_map = torch.tensor(saliency_map)
            if saliency_map.shape != X[i].shape:
                saliency_map = upsampling_fn(saliency_map)
        list_maps.append(saliency_map)

    list_maps = torch.stack(list_maps)

    #Upsample images if saliency's shape != image's shape
    if list_maps.shape[-2:] != img_shape[-2:]:
        list_maps = torch.nn.functional.interpolate(list_maps, img_shape[-2:], mode='bilinear')

    return list_maps.cpu().numpy()

if __name__ == '__main__':
    main()
