
import time as _time

from src.utils import df_Evaluator as _evaluator  
from backbone.df_sRAnet import df_sRAnet as _net   

_name = 'df_sRANet' 
_time = _time.strftime('%m-%d %H:%M:%S', _time.localtime())

# Dataset
mode = 'addcategory'       ##['attribute','addcategory','addlandmark']
gaussian_R = 8
DATASET_PROC_METHOD_TRAIN = 'BBOXRESIZE'
DATASET_PROC_METHOD_VAL = 'BBOXRESIZE'
BATCH_SIZE = 16
########

# Network
USE_NET = _net
# LM_BRANCH = _lm_branch
EVALUATOR = _evaluator
#################

# Learning Scheme

# auto
TRAIN_DIR = 'runs/%s/' % _name + _time
VAL_DIR = 'runs/%s/' % _name + _time

MODEL_NAME = '%s' % _name
#############
