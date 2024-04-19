from monai.utils import set_determinism
import torch, os, glob, argparse

from src.transform.transform import TestTransform
from src.data_loader.dataloader import TestThreadDataloader, TestDataloader
from src.network.network_selector import NetworkSelector
from src.inference import Inference, InferenceMode

def args_input():
    p = argparse.ArgumentParser(description='cmd parameters')
    ##common argument
    p.add_argument('--gpu_index', '-gpu_index', type=int, default=0)
    p.add_argument('--defaults_path', '-defaults_path', type=str, default='/home/taehun/tae/2.Tutorial/1.MONAI_Spleen/monai_spleen_tutorial/')
    p.add_argument('--model_name', '-model_name', type=str, default= 'UNet', help="model list of UNet, UNETR, SwinUNETR")
    p.add_argument('--loss_function_name', '-loss_function_name', type=str, default= 'DiceLoss', help="loss function list of DiceLoss, DiceCELoss, DiceFocalLoss, SoftclDiceLoss")
    p.add_argument('--cache_rate', '-cache_rate', type=float, default= 1.0)
    p.add_argument('--num_workers', '-num_workers', type=float, default= 4)
    p.add_argument('--set_determinism_seed', '-set_determinism_seed', type=float, default= 4)
    p.add_argument('--overlap', '-overlap', type=float, default= 0.25)
    p.add_argument('--val_interval', '-val_interval', type=int, default= 2)
    p.add_argument('--cache_num', '-cache_num', type=int, default= 30)
    p.add_argument('--num_samples', '-num_samples', type=int, default= 4)
    p.add_argument('--split_json', '-split_json', type=str, default= 'dataset_femoral/dataset_femoral_2.json' )
    p.add_argument('--to_onehot', '-to_onehot', type=float, default= None)
    p.add_argument('--save_flag', '-save_flag', type=int, default= 0)
    p.add_argument('--evaluation_flag', '-evaluation_flag', type=int, default= 1)
    p.add_argument('--inference_mode', '-inference_mode', type=str, default= "constant", help="constant or gaussian")


    ## distance_weight argument
    p.add_argument('--distance_map_weight', '-distance_map_weight', type=float, default= 1.0)
    return p.parse_args()

if __name__ == '__main__':
    args = args_input()
    defaults_path = args.defaults_path
    val_interval = args.val_interval
    model_name = args.model_name
    loss_function_name = args.loss_function_name
    cache_rate = args.cache_rate
    num_workers = args.num_workers
    cache_num = args.cache_num
    num_samples = args.num_samples
    split_json = args.split_json
    to_onehot = args.to_onehot
    distance_map_weight = args.distance_map_weight
    set_determinism_seed = args.set_determinism_seed
    overlap = args.overlap
    save_flag = args.save_flag
    evaluation_flag = args.evaluation_flag
    inference_mode = args.inference_mode

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_determinism(seed=set_determinism_seed)

    val_transforms = TestTransform.transform()
    # val_ds, val_loader = TestThreadDataloader.dataloader(split_json, cache_num, cache_rate, val_transforms)
    print("Loader :: TestThreadDataloader")
    val_ds, val_loader = TestThreadDataloader.dataloader(split_json, cache_num, cache_rate, val_transforms)
    
    print(f'model: {model_name}, loss: {loss_function_name}')
    model = NetworkSelector(model_name, device).get_model()

    Inference(model, model_name, loss_function_name, defaults_path, val_loader, overlap, inference_mode, to_onehot, device, save_flag, evaluation_flag).forward()


## ex) python test.py -model_name UNet -cache_num 9 -cache_rate 0.2 -loss_function_name DiceCELoss -inference_mode constant -save_flag 1 -evaluation_flag 1 
#python test.py -model_name UNet -cache_num 9 -cache_rate 1.0 -loss_function_name DiceCELoss -gpu_index 0 -overlap 0.25 -split_json dataset_femoral/dataset_femoral_all_case.json -inference_mode constant -save_flag 1 -evaluation_flag 1


## ex) python test.py -model_name UNETR -cache_num 9 -cache_rate 0.2 -loss_function_name DiceCELoss -save_flag 1 -evaluation_flag 1
#python test.py -model_name UNETR -cache_num 9 -cache_rate 1.0 -loss_function_name DiceCELoss -gpu_index 1 -overlap 0.25 -split_json dataset_femoral/dataset_femoral_all_case.json -inference_mode constant -save_flag 1 -evaluation_flag 1


## ex) python test.py -model_name SwinUNETR -cache_num 9 -cache_rate 0.2 -loss_function_name DiceCELoss -save_flag 1 -evaluation_flag 1
#python test.py -model_name SwinUNETR -cache_num 9 -cache_rate 1.0 -loss_function_name DiceCELoss -gpu_index 2 -overlap 0.25 -split_json dataset_femoral/dataset_femoral_all_case.json -inference_mode constant -save_flag 1 -evaluation_flag 1


