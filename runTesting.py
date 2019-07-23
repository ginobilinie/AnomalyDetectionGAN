# from __future__ import print_function
import argparse, os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch
import torch.utils.data as data_utils
# from Unet2d_pytorch import UNet, ResUNet, UNet_LRes, ResUNet_LRes, Discriminator
from nnBuildUnits import  adjust_learning_rate, calc_gradient_penalty
from ganComponents import Generator, Generator_Z, Discriminator, Generator3D, Generator3D_Z, Discriminator3D
from utils import *
import time
# from dataClean import denoiseImg,denoiseImg_isolation,denoiseImg_closing
import SimpleITK as sitk

parser = argparse.ArgumentParser(description="PyTorch InfantSeg")

parser.add_argument("--NDim", type=int, default=3, help="the dimension of the shape, 1D, 2D or 3D?")
parser.add_argument("--isSegReg", action="store_true", help="is Seg and Reg?", default=False)
parser.add_argument("--isMultiSource", action="store_true", help="is multiple input modality used?", default=False)
parser.add_argument("--whichLoss", type=int, default=1, help="which loss to use: 1. LossL1, 2. lossRTL1, 3. MSE (default)")
parser.add_argument("--whichNet", type=int, default=4, help="which loss to use: 1. UNet, 2. ResUNet, 3. UNet_LRes and 4. ResUNet_LRes (default, 3)")
parser.add_argument("--lossBase", type=int, default=1, help="The base to multiply the lossG_G, Default (1)")
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")
parser.add_argument("--numOfChannel_singleSource", type=int, default=5, help="# of channels for a 2D patch for the main modality (Default, 5)")
parser.add_argument("--numOfChannel_allSource", type=int, default=5, help="# of channels for a 2D patch for all the concatenated modalities (Default, 5)")
parser.add_argument("--isResidualEnhancement", action="store_true", help="is residual learning operation enhanced?", default=False)
parser.add_argument("--isViewExpansion", action="store_true", help="is view expanded?", default=True)
parser.add_argument("--isAdLoss", action="store_true", help="is adversarial loss used?", default=True)
parser.add_argument("--isSpatialDropOut", action="store_true", help="is spatial dropout used?", default=False)
parser.add_argument("--isFocalLoss", action="store_true", help="is focal loss used?", default=False)
parser.add_argument("--isSampleImportanceFromAd", action="store_true", help="is sample importance from adversarial network used?", default=False)
parser.add_argument("--dropoutRate", type=float, default=0.25, help="Spatial Dropout Rate. Default=0.25")
parser.add_argument("--lambdaAD", type=float, default=0, help="loss coefficient for AD loss. Default=0")
parser.add_argument("--adImportance", type=float, default=0, help="Sample importance from AD network. Default=0")
parser.add_argument("--isFixedRegions", action="store_true", help="Is the organ regions roughly known?", default=False)
#parser.add_argument("--modelPath", default="/home/niedong/Data4LowDosePET/pytorch_UNet/model/resunet2d_pet_Aug_noNorm_lres_bn_lr5e3_base1_lossL1_0p01_0624_200000.pt", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--modelPath", default="/home/niedong/Data4LowDosePET/pytorch_UNet/model/resunet2d_dp_pet_BatchAug_sNorm_lres_bn_lr5e3_lrdec_base1_lossL1_0p005_0628_200000.pt", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="pred_resunet2d_dp_pet_Aug_sNorm_lres_lrdce_bn_lr5e3_base1_lossL1_0628_20w_", type=str, help="prefix of the to-be-saved predicted filename")
parser.add_argument("--how2normalize", type=int, default=6, help="how to normalize the data")
parser.add_argument("--resType", type=int, default=2, help="resType: 0: segmentation map (integer); 1: regression map (continuous); 2: segmentation map + probability map")

def main():
    opt = parser.parse_args()
    print opt

    path_test = '/home/niedong/Data4LowDosePET/data_niigz_scale/'

    if opt.NDim==2:
        netD = Discriminator()
    else:
        netD = Discriminator3D()

    netD.cuda()

    checkpoint = torch.load(opt.modelPath)
    netD.load_state_dict(checkpoint['model'])


    ids = [1,6,11,16,21,26,31,36,41,46] #in on folder, we test 10 which is the testing set
    ids = [1] #in on folder, we test 10 which is the testing set

    ids = ['1_QFZ','2_LLQ','3_LMB','4_ZSL','5_CJB','11_TCL','15_WYL','21_PY','25_LYL','31_CZX','35_WLL','41_WQC','45_YXM']
    for ind in ids:
        start = time.time()

        mr_test_itk = sitk.ReadImage(os.path.join(path_test,'%s_60s_suv.nii.gz'%ind))#input modality

        spacing = mr_test_itk.GetSpacing()
        origin = mr_test_itk.GetOrigin()
        direction = mr_test_itk.GetDirection()
        
        mrnp=sitk.GetArrayFromImage(mr_test_itk)
        
        ##### specific normalization #####
        mu = np.mean(mrnp)
        
        #for training data in pelvicSeg
        if opt.how2normalize == 1:
            maxV, minV = np.percentile(mrnp, [99 ,1])
            print 'maxV,',maxV,' minV, ',minV
            mrnp = (mrnp-mu)/(maxV-minV)
            print 'unique value: ',np.unique(mrnp)
        
        #for training data in pelvicSeg
        if opt.how2normalize == 2:
            maxV, minV = np.percentile(mrnp, [99 ,1])
            print 'maxV,',maxV,' minV, ',minV
            mrnp = (mrnp-mu)/(maxV-minV)
            print 'unique value: ',np.unique(mrnp)
        
        #for training data in pelvicSegRegH5
        if opt.how2normalize== 3:
            std = np.std(mrnp)
            mrnp = (mrnp - mu)/std
            print 'maxV,',np.ndarray.max(mrnp),' minV, ',np.ndarray.min(mrnp)
        
        if opt.how2normalize == 4:
            maxLPET = 149.366742
            maxPercentLPET = 7.76
            minLPET = 0.00055037
            meanLPET = 0.27593288
            stdLPET = 0.75747500
        
            matLPET = (mrnp - minLPET) / (maxPercentLPET - minLPET)
        
        if opt.how2normalize == 5:
            # for rsCT
            maxCT = 27279
            maxPercentCT = 1320
            minCT = -1023
            meanCT = -601.1929
            stdCT = 475.034
        
            print 'ct, max: ', np.amax(mrnp), ' ct, min: ', np.amin(mrnp)
        
            matLPET = mrnp
        
        if opt.how2normalize == 6:
            maxPercentPET, minPercentPET = np.percentile(mrnp, [99.5, 0])
        
            matLPET = (mrnp - minPercentPET) / (maxPercentPET - minPercentPET)
        
        
        
        matFA = matLPET
        #                 matGT = hpetnp
        
        print 'matFA shape: ',matFA.shape
        pred = testOneSubject4Cla(matFA,[64,64,64],[32,32,32],netD, opt.prefixModelName+'%d.pt'%iter)
        print 'predicted result is ', pred


if __name__ == '__main__':
#     testGradients()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
