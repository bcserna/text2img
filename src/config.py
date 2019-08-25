D_GF = 64
D_DF = 64
D_WORD = 256
D_HIDDEN = 128
D_Z = 100
D_COND = 100

GENERATOR_LR = 4e-5
DISCRIMINATOR_LR = 4e-5  # BATCH SIZE!

RESIDUALS = 2

IMG_WEIGHT_INIT_RANGE = 0.1

P_DROP = 0.4

GAN_BATCH = 8
BATCH = 32

BASE_SIZE = 64

BRANCH_NUM = 3

CAPTIONS = 10

CAP_MAX_LEN = 18

END_TOKEN = '<END>'
UNK_TOKEN = '<UNK>'

DEVICE = 'cuda'

GAMMA_1 = 4.0
GAMMA_2 = 5.0
GAMMA_3 = 10.0
LAMBDA = 1.0

MIN_WORD_FREQ = 3
