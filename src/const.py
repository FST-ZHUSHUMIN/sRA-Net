import time
import torch
import socket as _socket

_hostname = str(_socket.gethostname())
print(_hostname)
mode = 'addcategory'   # #['attribute','addcategory','addlandmark']

name = time.strftime('%m-%d %H:%M:%S', time.localtime())
print(name)

# USE_NET = 'VGG16'
USE_NET = 'VGG16'

TRAIN_DIR = 'runs/' + name
VAL_DIR = 'runs/' + name

FASHIONET_LOAD_VGG16_GLOBAL = False

DATASET_PROC_METHOD_TRAIN = 'BBOXRESIZE'
DATASET_PROC_METHOD_VAL = 'BBOXRESIZE'

# 0: no sigmoid 1: sigmoid
VGG16_ACT_FUNC_IN_POSE = 0

MODEL_NAME = 'vgg16.pkl'

# # deepfashion path
base_df_path = '../dataset/DeepFashion/'
USE_df_CSV = 'info_attribute_created.csv'


# # technicialdrawing path
base_td_path = '../dataset/TechnicialDrawingcut2/'
USE_td_CSV = 'Women-Technical-drawing-updated-attribute-combined.xlsx'

# # iMaterialist path
base_im_path = '/home/shumin/Documents/new_folder/zhushumin-project/DeepFashionmaster/dataset/imaterialist-challenge-fashion-2018/'
USE_im_CSV = 'type_all_info.csv'



NUM_EPOCH = 12
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY = 0.9
BATCH_SIZE = 32
VAL_BATCH_SIZE = 32

WEIGHT_ATTR_NEG = 0.1
WEIGHT_ATTR_POS = 1
WEIGHT_LANDMARK_VIS_NEG = 0.5
WEIGHT_LANDMARK_VIS_POS = 0.5


# LOSS WEIGHT
WEIGHT_LOSS_CATEGORY = 1
WEIGHT_LOSS_ATTR = 500
WEIGHT_LOSS_VIS = 1
WEIGHT_LOSS_LM_POS = 1


# VAL
VAL_CATEGORY_TOP_N = (1, 3, 5)
VAL_ATTR_TOP_N = (3, 5)
VAL_LM_RELATIVE_DIS = 0.1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

lm2name = ['L.Col', 'R.Col', 'L.Sle', 'R.Sle', 'L.Wai', 'R.Wai', 'L.Hem', 'R.Hem']
attrtype2name = {0:'category', 1: 'texture', 2: 'fabric', 3: 'shape', 4: 'part', 5: 'style'}

iM_attrtype2name = {1:'Categoory', 2:'Color', 3:'Gender', 4:'Material', 5:'Neckline', 6:'Pattern', 7:'Sleeve', 8:'Style'}

VAL_WHILE_TRAIN = True


LM_TRAIN_USE = 'vis'
LM_EVAL_USE = 'vis'
