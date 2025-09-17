# %%
import os, sys
import mlflow
from tqdm import tqdm
from easydict import EasyDict
from pathlib import Path
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib
import torch
import torch.nn.functional as F

display = print
import ipywidgets
import concurrent.futures

DELPHI_DIR = os.getenv("D", os.getenv("HOME") + "/repos/delphi")
DELPHI_DIR = Path(DELPHI_DIR) 
DATADIR = DELPHI_DIR / "data/transforms/ukb_real_data/"
os.chdir(DELPHI_DIR)
sys.path.append(os.getcwd())

import model
model = importlib.reload(model)
Delphi = model.Delphi
from cv_utils import get_data_partitions
from utils import *

# ———————————————————————————————————————————————————————————————

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = getattr(torch, 'float32')

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# ———————————————————————————————————————————————————————————————

# runs_df = mlflow.search_runs(experiment_ids = [exp.experiment_id for exp in mlflow.search_experiments()])
# runs_df = runs_df[runs_df['fold'].notnull()]
# grouped = runs_df.groupby('experiment_id')

# exp_ids = []

# for exp_id, group in grouped:
    
#     if group['fold'].nunique() == 5:
#         # print(group[['run_id', 'params.val_filename']])
#         print(f"Experiment ID: {exp_id}")
#         exp_ids.append(exp_id)

# runs_df = runs_df[runs_df['experiment_id'].isin(exp_ids)]
# display(runs_df)

# runinfo = runs_df.iloc[0]

mlflow.set_tracking_uri(DELPHI_DIR / "mlruns")

RUN_ID = "6b47df60154d4f67a70a7f0c94f712b6"

run = mlflow.get_run(RUN_ID)
runinfo = pd.Series({
    "run_id": run.info.run_id, 
    "status": run.info.status, 
    "artifact_uri": run.info.artifact_uri, 
    **run.data.params, 
    **run.data.metrics, 
    **run.data.tags}
)

labels = pd.read_csv("data/delphi_labels_chapters_colours_icd.csv")
ignored_tokens = get_ignored_tokens(runinfo)
tokens_of_interest = [ x for x in range(len(labels)) if x not in ignored_tokens ]

t_min = float(runinfo['t_min'])
mask_ties = ast.literal_eval(runinfo['mask_ties'])

best_ckpt_path    = get_best_ckpt(runinfo)
model = Delphi.from_checkpoint(best_ckpt_path).eval().to(device)

data = get_data_partitions(DATADIR / "ukb_real.bin", fold=1)
_, _, (test_data, test_p2i, test_ids) = data

count_tokens_df = get_top_counts(test_data, labels, N:=100, ignored_tokens)

# count_tokens_df.\
    # plot.barh(figsize=(10, 20), 
    #   title=f"Top {N} tokens in validation set", 
    #   xlabel="Count", ylabel="Token"
    # );

# %%
def get_batch_wrapper(token_stream):
    return get_batch(ix=token_stream, data=test_data, p2i=test_p2i, select='left', block_size=block_size, device=device, padding='random')

def model_forward(X, age_X):
    with torch.no_grad():
        return model(X, age_X)

mini_batch_size = 32
n_chunks = 512
block_size = 128

ix = torch.randint(len(test_p2i), (batch_size := n_chunks * mini_batch_size,))

with concurrent.futures.ThreadPoolExecutor() as executor:

    batches = list(executor.map(get_batch_wrapper, ix.chunk(n_chunks)))

    token_stream = torch.stack([b[0] for b in batches]).view(batch_size, block_size),
    age          = torch.stack([b[1] for b in batches]).view(batch_size, block_size),
    targets      = torch.stack([b[2] for b in batches]).view(batch_size, block_size),
    targets_age  = torch.stack([b[3] for b in batches]).view(batch_size, block_size)
    
    outputs = list(executor.map(lambda b: model_forward(b[0], b[1]), batches))

    # logits = torch.concat([ outputs[b][0] for b in range(len(outputs)) ])
    logits = torch.concat([ o[0] for o in outputs ])
    del outputs

token_stream = token_stream[0]
age          = age[0]
targets      = targets[0]
targets_age  = targets_age[0]

token_stream

# %%
id_to_token = dict(zip(labels.index-1, labels.name))
list( map(lambda x: id_to_token[x-1], ignored_tokens) )

# %%
with torch.no_grad():

    # if we are given some desired targets also calculate the loss
    # ignored_tokens = self.config.ignore_tokens.copy()    
    
    # "filter" columns (setting to -Inf)
    if (validation_loss_mode := True):
        logits = model.blackout_ignored(logits, ignored_tokens)

    # filter rows
    pass_tokens = model.get_allowed_tokens_mask(targets, ignored_tokens)
    
    loss_ce = model.cross_entropy_loss(logits, targets, pass_tokens, agg='per_disease')

    loss_dt = model.time_to_event_loss(
        logits, delta_t:= targets_age-age, 
        pass_tokens, attn_mask, mask_ties, t_min, agg=None
    )

    loss = dict(loss_ce=loss_ce, loss_dt=loss_dt)

# %%
(  
   tokens_of_interest := labels.iloc[tokens_of_interest].\
    sort_values("count", ascending=False).\
    index.tolist()
)

# %%
loss, loss_ce, loss_dt = EasyDict(), EasyDict(), EasyDict()

# def 

loss_ce_per_disease = model.cross_entropy_loss(logits, targets)


# %%
def loss_dt_for_token(logits, targets, delta_age, token, t_min):
    
    # dt = flattened_targets_age - flattened_age

    targets = targets.reshape(-1)
    
    pass_tokens = targets != -1
    pass_tokens *= targets == token

    if pass_tokens.sum() == 0:
        return 0

    delta_age = delta_age.reshape(-1)
    delta_age = delta_age[pass_tokens]

    dt_for_token = - torch.log(torch.clamp(delta_age, min=1.0) + t_min)

    flattened_logits = logits.reshape(-1, logits.size(-1))

    lse = torch.logsumexp(logits, -1)
    flattened_lse = lse.reshape(-1)

    flat_lse_pass = flattened_lse[pass_tokens]
    loss_for_token = - (flat_lse_pass - torch.exp(flat_lse_pass - dt_for_token)).sum() / len(logits) ## Exponential log-likelihood (real statistics, TM)
    
    return loss_for_token.item()

   
# %%     

loss_dt = dict()

with torch.no_grad():

    targets = targets.reshape(-1)
    lse = torch.logsumexp(logits, -1)

    flattened_logits = logits.reshape(-1, logits.size(-1))
    flattened_targets_age = targets_age.reshape(-1)
    flattened_age = age.reshape(-1)
    flattened_lse = lse.reshape(-1)

    total_loss = 0
    for token_of_interest in tqdm(tokens_of_interest):
               
        loss_for_token = loss_dt_for_token(logits, targets, targets_age-age, token=token_of_interest, t_min=t_min)
        loss_dt[str(token_of_interest)] = loss_for_token

        # loss_dt = loss_dt_for_token()

        total_loss += loss_for_token

        # pass_tokens = targets != -1
        # pass_tokens *= targets == allowed_token

        # if pass_tokens.sum() > 0:

            # flat_targets_age_pass = flattened_targets_age[pass_tokens]
            # flat_age_pass = flattened_age[pass_tokens]
            # flat_lse_pass = flattened_lse[pass_tokens]
            
            # dt = flattened_targets_age - flattened_age
            # dt_for_token = - torch.log(torch.clamp(dt[pass_tokens], min=1.0) + t_min)
            # dt_for_token = - (flat_lse_pass - torch.exp(flat_lse_pass - dt_for_token)).sum() / len(logits) ## Exponential log-likelihood (real statistics, TM)
            # loss[f'loss_dt_{allowed_token}'] = dt_for_token
            # loss_dt[str(allowed_token)] = dt_for_token
            

            # ce_for_token = -torch.log(F.softmax(flattened_logits[pass_tokens])[:, allowed_token]).sum().item() / len(logits)
            # loss[f'loss_ce_{allowed_token}'] = ce_for_token
            # loss_ce[str(allowed_token)] = ce_for_token
            # print(f"{ce_for_token:.4f}")
            # print(f"{dt_for_token:.4f}")
            # total_loss += ce_for_token
            
            # print(total_loss)

# %%
{ id_to_token.get(int(k), k): v for k, v in loss_dt.items() }

# %%
labels['ce'] = labels['index'].apply(lambda x: loss_ce.get(str(x), 0))
labels.sort_values('ce', ascending=False).head(600)

# %%
print(allowed_token, pass_tokens.sum(), ce_for_token)      

# %%
loss_ce = F.cross_entropy(
    logits.reshape(-1, logits.size(-1))[pass_tokens], 
    targets[pass_tokens], ignore_index=-1
)

# %%
F.cross_entropy(flattened_logits[pass_tokens], targets[pass_tokens], weight=1/sum(pass_tokens))
sum(pass_tokens)
