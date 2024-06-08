import torch
import coreset_selection
from data_reweighting.meta_weight_net import meta_weight_net, get_args
from data import get_classes_count, DATASETS_DICT, get_transforms, get_dataset, CustomSubset
from utils import set_cudnn, get_model, broadcast_weights
from args import parse_args
from train import train

def main():
    args = parse_args()
    args.broadcast_weights = str(args.broadcast_weights).lower() == 'true'

    print(args)
    
    if args.weighting_method == 'uniform' and args.broadcast_weights:
        args.coreset_method = '-'
        args.coreset_ratio = 1
        args.feature_extractor = '-'
    
    set_cudnn(device=args.device)
    
    if args.dataset not in DATASETS_DICT:
        raise ValueError(f"Dataset {args.dataset} not supported. Supported datasets: {list(DATASETS_DICT.keys())}")
    
    _, imagenet_transform = get_transforms('imagenet')
    train_dataset_imagenet = get_dataset(root=args.data_dir, dataset_name=args.dataset, train=True, transform=imagenet_transform)
    
    train_transform, test_transform = get_transforms(args.dataset)
    trainset = get_dataset(root=args.data_dir, dataset_name=args.dataset, train=True, transform=train_transform)
    testset = get_dataset(root=args.data_dir, dataset_name=args.dataset, train=False, transform=test_transform)
    
    count = int(args.coreset_ratio * len(train_dataset_imagenet))
    coreset_indices = torch.arange(len(trainset))
    
    if args.coreset_method == 'random':
        coreset_indices = coreset_selection.random_selection.RandomCoresetSelection(
            args.feature_extractor, train_dataset_imagenet, args.dataset, count, device=args.device
        ).get_coreset()
    elif args.coreset_method == 'moderate_selection':
        coreset_indices = coreset_selection.moderate_selection.ModerateCoresetSelection(
            args.feature_extractor, train_dataset_imagenet, args.dataset, count, device=args.device
        ).get_coreset()
    
    coreset_weights = None
    weighted_coreset_accuracy = '-'
    
    if args.weighting_method == 'meta_weight_net':
        default_args = get_args()
        default_args.dataset = args.dataset
        default_args.num_classes = get_classes_count(args.dataset)
        coreset = CustomSubset(trainset, coreset_indices)
        num_meta = max(int(0.02 * len(coreset)), max(min(100, int(0.5 * len(coreset))), 1))
        default_args.num_meta = num_meta
        coreset_weights, weighted_coreset_accuracy = meta_weight_net(default_args, coreset, testset)
    elif args.weighting_method == 'uniform':
        coreset_weights = torch.ones(coreset_indices.shape[0], device=args.device)
    else:
        raise NotImplementedError
    
    model = get_model(args.arch, get_classes_count(args.dataset)).to(args.device)
    coreset = CustomSubset(trainset, coreset_indices)
    
    if args.broadcast_weights:
        # Train on the whole dataset
        if args.weighting_method == 'uniform':
            train(
                model, trainset, testset, torch.ones(len(trainset), device=args.device),
                args.batch_size, args.device, args.epochs, args.test_interval, args.lr, args.momentum, args.wd
            )
        elif args.weighting_method == 'meta_weight_net':
            coreset_imagenet = CustomSubset(train_dataset_imagenet, coreset_indices)
            weights = broadcast_weights(
                args.feature_extractor, coreset_weights, train_dataset_imagenet, coreset_imagenet,
                num_workers=int(args.num_workers), device=args.device
            )
            train(
                model, trainset, testset, weights, args.batch_size, args.device,
                args.epochs, args.test_interval, args.lr, args.momentum, args.wd
            )
        else:
            raise NotImplementedError
    else:
        # Train on the coreset only
        if args.weighting_method == 'uniform':
            train(
                model, coreset, testset, torch.ones(len(coreset), device=args.device),
                args.batch_size, args.device, args.epochs, args.test_interval, args.lr, args.momentum, args.wd
            )
        elif args.weighting_method == 'meta_weight_net':
            train(
                model, coreset, testset, coreset_weights, args.batch_size, args.device,
                args.epochs, args.test_interval, args.lr, args.momentum, args.wd
            )
        else:
            raise NotImplementedError
        
if __name__ == "__main__":
    main()
