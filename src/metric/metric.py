import nibabel
import numpy as np
import os
from skimage.measure import label
import SimpleITK as sitk
from skimage.measure import label
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure

def dice_coef(y_true, y_pred):
    gt = sitk.GetImageFromArray(y_true, isVector=False)
    my_mask = sitk.GetImageFromArray(y_pred, isVector=False)
    dice_dist = sitk.LabelOverlapMeasuresImageFilter()
    dice_dist.Execute(gt > 0.5, my_mask > 0.5)
    dice = dice_dist.GetDiceCoefficient()
    return dice


def HausdorffDistance(y_true, y_pred, spaceing):
    gt = sitk.GetImageFromArray(y_true, isVector=False)
    mask = sitk.GetImageFromArray(y_pred, isVector=False)
    gt.SetSpacing(spaceing)
    mask.SetSpacing(spaceing)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(gt > 0.5, mask > 0.5)
    AvgHD = hausdorffcomputer.GetAverageHausdorffDistance()
    HD = hausdorffcomputer.GetHausdorffDistance()
    return AvgHD, HD

def dice_coef2(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(
        y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth) # 1>= loss >0

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):

    result = np.atleast_1d(result.astype(np.float32))
    reference = np.atleast_1d(reference.astype(np.float32))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds

def hd95_(result, reference, voxelspacing=None, connectivity=1):

    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
        
    return hd95        


def get_region_num(img, get_num):
    '''
    :param img: 输入标签图像
    :param get_num: 保留最大连通域的个数
    :return: 提取连通域后的numpy数组
    '''
    conn_img = label(img, connectivity=3)
    i_max = conn_img.max()
    if get_num > i_max:
        get_num = i_max
    area_list = []
    for i in np.arange(1, i_max + 1):
        area_list.append(np.sum(conn_img == i))
    index = np.argsort(-np.array(area_list)) + 1
    record = np.zeros_like(img)
    for i in np.arange(get_num):
        record[conn_img == index[i]] = 1
    return record


class Cal_metrics:
    '''
    计算指标多进程版
    预测标签的目录为id/pre_label.nii.gz
    真实标签的目录为id/(label.nii.gz,image.nii.gz)
    '''
    def __init__(self, pre_path, true_path, con_num=0 ,pre_label_name='pre_label.nii.gz',is_use_prob=True):
        '''
        :param pre_path:  the path of prediction
        :param true_path: the path of true label
        :param con_num: the number of max connected domain would be reserved, 如果为0将保留所有连通域
        :param is_use_prob: 是否进行二值化操作
        '''
        self.pre_path = pre_path
        self.true_path = true_path
        self.con_num= con_num
        self.is_use_prob = is_use_prob
        self.pre_label_name=pre_label_name

    def calculate_dice(self, i):
        '''
        :param i: id号，id号必须存在于预测路径以及真实路径
        :return:
        '''

        pre_nii = nibabel.load(os.path.join(self.pre_path, i, self.pre_label_name))
        pre = pre_nii.get_fdata()

        if self.is_use_prob:
            pre[pre >= 0.5] = 1
            pre[pre < 0.5] = 0

        if self.con_num!=0:
            pre = get_region_num(pre, self.con_num)

        true = nibabel.load(os.path.join(self.true_path, i, 'label.nii.gz')).get_fdata()
        data_nii = nibabel.load(os.path.join(self.true_path, i, 'img.nii.gz'))

        header = data_nii.header
        spacing = header.get_zooms()
        spacing = tuple([float(spacing[0]), float(spacing[1]), float(spacing[2])])

        max_dice = dice_coef(true, pre)
        ahd, hd = HausdorffDistance(true, pre, spacing)
        # hd95 = hd95_(pre, true)
        # print(i,' dice:%f hd:%f'%(max_dice,hd))
        print(i,' dice:%f ahd: %f hd:%f'%(max_dice,ahd, hd))
        # print(i,' dice:%f ahd: %f hd:%f hd95:%f'%(max_dice,ahd, hd, hd95))

        return max_dice, ahd, hd
