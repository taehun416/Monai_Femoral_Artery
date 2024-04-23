from monai.utils import set_determinism
import torch, os, glob, argparse
# from src.trainer import TrainUNet, TrainMultiHeaderUNet, TrainUNETR, TrainSwinUNETR, TrainUNETR2, TrainMultiHeaderSwinUNETR, Trainer
from src.trainer import Trainer

from src.transform.transform import TrainTransform, TrainTransformAll
from src.data_loader.dataloader import TrainThreadDataloader, TrainDataloader
from src.network.network_selector import NetworkSelector
from src.loss.lossfunction_selector import LossFunctionSelector

def args_input():
    p = argparse.ArgumentParser(description='cmd parameters')
    ##common argument
    p.add_argument('--gpu_index', '-gpu_index', type=int, default=0)
    p.add_argument('--defaults_path', '-defaults_path', type=str, default='/home/taehun/tae/2.Tutorial/1.MONAI_Spleen/monai_spleen_tutorial/')
    p.add_argument('--model_name', '-model_name', type=str, default= 'UNet', help="model list of UNet, UNETR, SwinUNETR, UNet_Multiheader")
    p.add_argument('--loss_function_name', '-loss_function_name', type=str, default= 'DiceLoss', help="loss function list of DiceLoss, DiceCELoss, DiceFocalLoss, SoftclDiceLoss, FGDTMloss")
    p.add_argument('--optimizer_name', '-optimizer_name', type=str, default= 'Adam', help = 'optimizer list of Adam, AdamW')
    p.add_argument('--cache_rate', '-cache_rate', type=float, default= 1.0)
    p.add_argument('--num_workers', '-num_workers', type=float, default= 4)
    p.add_argument('--set_determinism_seed', '-set_determinism_seed', type=float, default= 4)

    ## UNet argument
    p.add_argument('--max_epochs', '-max_epochs', type=int, default= 100)
    p.add_argument('--val_interval', '-val_interval', type=int, default= 2)

    ## SwinUNETR argument
    p.add_argument('--cache_num', '-cache_num', type=int, default= 30)
    p.add_argument('--num_samples', '-num_samples', type=int, default= 4)
    # p.add_argument('--split_json', '-split_json', type=str, default= 'dataset_spleen/dataset_spleen.json')
    p.add_argument('--split_json', '-split_json', type=str, default= 'dataset_femoral/dataset_femoral_all_case.json')
    p.add_argument('--max_iterations', '-max_iterations', type=int, default= 30000)
    p.add_argument('--eval_num', '-eval_num', type=int, default= 500)
    p.add_argument('--to_onehot', '-to_onehot', type=int, default= None)
    ## distance_weight argument
    p.add_argument('--distance_map_weight', '-distance_map_weight', type=float, default= 1.0)
    p.add_argument('--dist_flag', '-dist_flag', type=int, default= 0)
    p.add_argument('--valid_save_flag', '-valid_save_flag', type=int, default= 0)

    return p.parse_args()


if __name__ == '__main__':
    args = args_input()
    defaults_path = args.defaults_path
    max_epochs = args.max_epochs
    val_interval = args.val_interval
    model_name = args.model_name
    loss_function_name = args.loss_function_name
    optimizer_name = args.optimizer_name
    cache_rate = args.cache_rate
    num_workers = args.num_workers
    cache_num = args.cache_num
    num_samples = args.num_samples
    split_json = args.split_json
    max_iterations = args.max_iterations
    eval_num = args.eval_num
    to_onehot = args.to_onehot
    distance_map_weight = args.distance_map_weight
    set_determinism_seed = args.set_determinism_seed
    dist_flag = args.dist_flag
    valid_save_flag = args.valid_save_flag

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_determinism(seed=set_determinism_seed)

    # train_transforms, val_transforms = TrainTransformAll.transform()
    train_transforms, val_transforms = TrainTransform.transform()

    model = NetworkSelector(model_name, device).get_model()
    criterion = LossFunctionSelector(loss_function_name).get_loss_function()
    print("Loader :: TrainThreadDataloader")
    train_ds, train_loader, val_ds, val_loader = TrainThreadDataloader.dataloader(split_json, train_transforms, val_transforms, cache_num, cache_rate, num_workers)

#model_list = UNet, UNETR, SwinUNETR, SwinUNETR_Multiheader

    print(f'model: {model_name}, loss: {loss_function_name}')
    print('dist_flag::', dist_flag)
    if model_name == 'SwinUNETR':
        print("Pre_Train Load")
        weight = torch.load("./model_swinvit.pt")
        model.load_from(weights=weight)

    trainer = Trainer(model_name, model, loss_function_name, criterion, dist_flag, distance_map_weight, max_iterations, eval_num, optimizer_name, defaults_path, train_loader, val_loader, to_onehot, valid_save_flag).forward()

    
# ex) python train_test.py -model_name UNet -val_interval 1 -cache_rate 0.1 -gpu_index 0 -eval_num 1 -optimizer_name Adam -loss_function_name DiceLoss -cache_num 70 -num_workers 8 -split_json 'dataset_femoral/dataset_femoral_test.json'
# ex) python train_test.py -model_name SwinUNETR_Multiheader -val_interval 1 -cache_rate 0.1 -gpu_index 0 -dist_flag 1 -eval_num 1 -optimizer_name Adam -loss_function_name FGDTMloss -cache_num 70 -num_workers 8 -split_json 'dataset_femoral/dataset_TH.json'

#UNet
# ex) python train.py -model_name UNet -val_interval 1 -cache_rate 0.1 -optimizer_name Adam -loss_function_name DiceCELoss -cache_num 70 -num_workers 8 -split_json 'dataset_femoral/dataset_femoral_test.json'
# python train.py -model_name UNet -optimizer_name Adam -loss_function_name DiceCELoss -gpu_index 0 -cache_num 70 -num_workers 8 -split_json dataset_femoral/dataset_TH.json

#UNet_Multiheader
# ex) python train.py -model_name UNet_Multiheader -val_interval 1 -cache_rate 0.1 -optimizer_name Adam -loss_function_name FGDTMloss -dist_flag 1 -gpu_index 2 -cache_num 70 -num_workers 8 -split_json 'dataset_femoral/dataset_femoral_test.json'
# python train.py -model_name UNet_Multiheader -optimizer_name Adam -loss_function_name FGDTMloss -dist_flag 1 -gpu_index 2 -cache_num 70 -num_workers 8 -split_json dataset_femoral/dataset_TH.json


#UNETR
# ex) python train.py -model_name UNETR -cache_rate 0.1 -optimizer_name AdamW -eval_num 1 -loss_function_name DiceCELoss -cache_num 70 -num_workers 8 -split_json 'dataset_femoral/dataset_femoral_test.json'
# python train.py -model_name UNETR -optimizer_name AdamW -loss_function_name DiceCELoss -gpu_index 1 -cache_num 70 -num_workers 8 -split_json dataset_femoral/dataset_TH.json

#SwinUNETR
# ex) python train.py -model_name SwinUNETR -cache_rate 0.1 -optimizer_name AdamW -eval_num 1 -loss_function_name DiceCELoss -cache_num 70 -num_workers 8 -split_json 'dataset_femoral/dataset_femoral_test.json'
# python train.py -model_name SwinUNETR -optimizer_name AdamW -loss_function_name DiceCELoss -gpu_index 2 -cache_num 70 -num_workers 8 -split_json dataset_femoral/dataset_TH.json

#SwinUNETR_MultiHeader
# ex) python train.py -model_name SwinUNETR_Multiheader -val_interval 1 -cache_rate 0.1 -optimizer_name AdamW -loss_function_name FGDTMloss -dist_flag 1 -eval_num 1 -gpu_index 2 -valid_save_flag 1 -cache_num 70 -num_workers 8 -split_json 'dataset_femoral/dataset_femoral_test.json'
# python train.py -model_name SwinUNETR_Multiheader -optimizer_name AdamW -loss_function_name FGDTMloss -dist_flag 1 -gpu_index 2 -cache_num 70 -num_workers 8 -valid_save_flag 1 -split_json dataset_femoral/dataset_TH.json

