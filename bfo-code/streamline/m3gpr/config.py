from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# I/O Data Setup
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.FILE_TYPE = 'xlsx' ## input data file type
_C.DATA.NUM_TASKS = 3 ## The number of replicates
_C.DATA.NUM_OUTPUTS = 2 ## The number of replicates
_C.DATA.DROP_COLS = ['Molecular_Weight','task_ind'] #drop cols for fdt dataset

# -----------------------------------------------------------------------------
# Model Parameter
# -----------------------------------------------------------------------------
_C.MODEL = CN()
#_C.MODEL.Q = 2 ## The dimension of H
#_C.MODEL.GAP = 2

_C.MODEL.MODEL_NAME = 'MTMO'#'MTMO-LMGP','MTMO','MO'#'HGP'#'rf'#'singleGP'
_C.MODEL.CATE_TRANS = 'ohe'
_C.MODEL.X_SCALE = 'x-minmax'
_C.MODEL.Y_SCALE = 'y-stand'#'y-stand'#'no-y-scale'
_C.MODEL.NUM_TRAIN_ITERS = 100
_C.MODEL.SPLIT = 'by-task'#'combine'#'by-task'
_C.MODEL.CV = 'kfold'#none
_C.MODEL.N_CV = 5
_C.MODEL.METRIC = 'NMSE' #MAE

_C.MODEL.OUTPUT_RANK = 1 #if 0, no correlation between output
_C.MODEL.TASK_RANK = 2#if 0, no correlation between tasks
_C.MODEL.LIK_RANK = 4


# ---------------------------------------------------------------------------- #
# Paths
# ---------------------------------------------------------------------------- #
_C.PATH = CN()
_C.PATH.SAVING_GENERAL = '/Users/chenya68/Documents/GitHub/BFO' # General directory
#_C.PATH.LOSS = '/Loss/synthetic-same-input/Loss-HMOGPLV' # Output loss directory
#_C.PATH.PARAMETERS = '/Model_parameter/synthetic-same-input/Parameter-HMOGPLV' # Output parameters directory
#_C.PATH.PLOT = '/All-plot/synthetic-same-input/Plot-HMOGPLV' # Output plot directory
_C.PATH.RESULT = '/Users/chenya68/Documents/GitHub/BFO/output' # Output result directory
_C.PATH.DATA_PATH = '/Users/chenya68/Documents/GitHub/BFO/bfo-data/harpoon/harpoon-doe.xlsx'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.MISC = CN()
_C.MISC.DATA_NAME = 'Harpoon'
_C.MISC.DATE = '0925'

def get_cfg_defaults_all():
    return _C.clone()