import argparse
from feature_extractor import feature_extractors

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")


    # Model
    parser.add_argument("--arch", type=str, help="Model achitecture",
                        default="preactresnet18",
                        choices=["preactresnet18",
                                 "preactresnet34",
                                 "preactresnet50",
                                 "preactresnet101",
                                 "preactresnet152",
                                 "resnet20",
                                 "resnet32",
                                 "resnet44",
                                 "resnet56",
                                 "resnet110",
                                 "resnet1202"])
    
    
    # Feature extractor
    parser.add_argument("--feature-extractor", type=str, help="Feature extractor architecture", choices=feature_extractors)

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100"],
        help="Dataset for training and eval",
    )
    
    parser.add_argument(
        "--coreset-method",
        type=str,
        choices=["moderate_selection", "random"],
        help="Coreset selection method",
    )
    
    parser.add_argument(
        "--weighting-method",
        type=str,
        choices=["uniform", "meta_weight_net"],
        help="Weighting selection method",
    )
    
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=0.1,
        help="Coreset ratio",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        metavar="N",
    )
    
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for testing (default: 128)",
    )
    
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="whether to normalize the data",
    )
    
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="path to datasets"
    )
    
    parser.add_argument(
        "--epochs", type=int, default=100, metavar="N", help="number of epochs to train"
    )
    
    parser.add_argument(
        "--optimizer", type=str, default="sgd", choices=("sgd")
    )
    
    
    parser.add_argument("--wd", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="step",
        choices=("step"),
        help="Learning rate schedule",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--test-interval", type=int, default=5)

    parser.add_argument(
        "--device", type=str, default="cuda", choices=("cuda", "cpu")
    )
    
    parser.add_argument(
        '--broadcast-weights',
        type=str,
        choices=['True', 'False'],
        default='False',
        help='Broadcast weights to full dataset'
    )
        
    return parser.parse_args()