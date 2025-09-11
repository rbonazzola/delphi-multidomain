
import time

N_CV_DRUG_TOKENS = 132
N_NS_DRUG_TOKENS = 200
N_DRUG_TOKENS = N_CV_DRUG_TOKENS + N_NS_DRUG_TOKENS
GAPLEN = 3

out_dir = 'Delphi-cvns'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 # don't print too too often
seed = 42

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'delphi'
wandb_run_name = 'run' + str(time.time())

dataset = 'ukb_real_data'
batch_size = 128
block_size = 128
data_fraction = 1.0

n_layer = 12
n_head = 12
n_embd = 120
dropout = 0.1
weight_decay = 2e-1
vocab_size = 1270 + N_DRUG_TOKENS + GAPLEN + 1 # two death tokens

learning_rate = 6e-4 # with baby networks can afford to go a bit higher
max_iters = 100000
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr = 6e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = 5000 # not super necessary potentially

PADDING_TOKEN = 0
NO_EVENT_TOKEN = 1
SEX_TOKENS = [2,3]
LIFESTYLE_TOKENS = list(range(4, 13))
ICD10_TOKENS = list(range(13, 1400))

DEATH_TOKENS = [1400] 
CV_DRUG_TOKENS = list(range(DEATH_TOKENS[-1], DEATH_TOKENS[-1]+N_CV_DRUG_TOKENS))
NS_DRUG_TOKENS = list(range(DEATH_TOKENS[-1], DEATH_TOKENS[-1]+N_CV_DRUG_TOKENS+GAPLEN+N_NS_DRUG_TOKENS))

ignore_tokens = [PADDING_TOKEN] + SEX_TOKENS + LIFESTYLE_TOKENS

t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5
