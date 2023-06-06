import time as _time

from backbone.im_sRAnet import iMaterialist_sRAnet as _net

from src.utils import iMaterialist_Evaluator as _evaluator1 
from src.utils import iM_Evaluator_group as _evaluator2 

_name = 'im_RAnet_4'
mode = 'attribute'  
_time = _time.strftime('%m-%d %H:%M:%S', _time.localtime())
#  Dataset
BATCH_SIZE = 64

gaussian_R = 8
DATASET_PROC_METHOD_TRAIN = 'BBOXRESIZE'
DATASET_PROC_METHOD_VAL = 'BBOXRESIZE'
VAL_WHILE_TRAIN = True
########
# Network
USE_NET = _net
# LM_BRANCH = _lm_branch
EVALUATOR1 = _evaluator1
EVALUATOR2 = _evaluator2
#################

# Learning Scheme

# auto
TRAIN_DIR = 'runs/%s/' % _name + _time
VAL_DIR = 'runs/%s/' % _name + _time

MODEL_NAME = '%s' % _name
#############
