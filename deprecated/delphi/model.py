"""
Full definition of a GPT Language Model, all of it in this single file.
    References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import warnings

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        B = config.block_size
        self.register_buffer("bias", 
            torch.tril(torch.ones(B, B)).\
            view(1, 1, B, B)
        )

        '''
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = False #hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if not self.flash:
            # print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
        '''

    def forward(self, x, attn_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        '''
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
        '''

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att.masked_fill(attn_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, att

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        y, att = self.attn(self.ln_1(x), attn_mask) 
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att

class AgeEncoding(nn.Module):

    def __init__(self, config, max_dim: int = 1024):
        super().__init__()
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd))
        self.register_buffer('div_term', div_term)
        self.n_embd = config.n_embd
        self.linear = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
        y[..., 0::2] = torch.sin(x / 365.25 * self.div_term) #* (1-self.div_term)
        y[..., 1::2] = torch.cos(x /365.25 * self.div_term) #* (1-self.div_term)
        y = self.linear(y)
        
        #x = self.wae[:x.size(0)]
        return y #self.dropout(x)
    
@dataclass
class DelphiConfig:
    block_size: int = 1024
    vocab_size: int = 2048 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    token_dropout: float = 0.0
    t_min: float = 1.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    mask_ties: bool = False
    ignore_tokens: list = field(default_factory=lambda: [0])

class Delphi(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wae = AgeEncoding(config),
            token_drop = nn.Dropout(config.token_dropout),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        #if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def build_attention_mask(self, idx, age, targets, targets_age, mask_ties):

        # causal self-attention mask, to ensure that attention is only applied to the left in the input sequence
        device = idx.device
        B, L = idx.size(0), idx.size(1)

        PADDING_TOKEN = 0
        # Do not attend to padded positions
        attn_mask = (idx != PADDING_TOKEN).view(B, 1, 1, L) * (idx != PADDING_TOKEN).view(B, 1, L, 1)  
        
        attn_mask *= torch.tril(torch.ones(L, L, device=device))[None, None, :, :] > 0
        
        #self.transformer.h[0].attn.bias[:,:,:idx.size(1),:idx.size(1)] > 0
        
        # if targets is not None and self.config.mask_ties:
        if targets is not None and mask_ties:
            # Mask co-occuring tokens
            attn_mask *= ((age.view(idx.size(0),1,1,idx.size(1)) != targets_age.view(idx.size(0),1,idx.size(1),1))) 
            attn_mask += (attn_mask.sum(-1, keepdim=True)==0) * torch.diag(torch.ones(idx.size(1), device=device)) > 0
        
        # Except for padding
        attn_mask = attn_mask + (idx==PADDING_TOKEN).view(B, 1, 1, L) * torch.diag(torch.ones(L, device=device)) > 0 

        attn_mask *= torch.tril(torch.ones(L, L, device=device))[None,None,:,:] > 0 # self.transformer.h[0].attn.bias[:,:,:idx.size(1),:idx.size(1)] > 0
        
        return attn_mask
        

    def transformer_block(self, x, attn_mask, return_attentions=False):
        """
        Forward pass through the transformer block, with optional attention return.
        """
        if return_attentions:
            att = []
            for block in self.transformer.h:
                x, a = block(x, attn_mask)
                att.append(a)
            x = self.transformer.ln_f(x)
            att = torch.stack(att)
        else:
            for block in self.transformer.h:
                x, _ = block(x, attn_mask)
            x = self.transformer.ln_f(x)
            att = None
        return x, att


    def cross_entropy_loss(self, logits, targets, pass_tokens, agg=None):
        '''
        Cross entropy loss for the next token prediction.
        Arguments:
            logits: Tensor, shape [batch_size, sequence_length, vocab_size]
            targets: Tensor, shape [batch_size, sequence_length]
            pass_tokens: Tensor of bools, batch_size * sequence_length
            agg: one of None, "mean" or "sum"
        '''

        n_classes = logits.size(-1)            
        if agg == "per_token":
            log_softmax = F.log_softmax(logits.view(-1, n_classes)[pass_tokens])
            loss_ce_per_token = log_softmax[torch.arange(log_softmax.size(0)), targets.view(-1)[pass_tokens]]
            return loss_ce_per_token
        if agg == "per_disease":
            loss_ce_per_token = self.cross_entropy_loss(logits, targets, pass_tokens, agg="per_token")
            loss_ce_agg_per_disease = pd.DataFrame([loss_ce_per_token, targets.view(-1)[pass_tokens].cpu().numpy()]).T.\
                set_axis(["log_p", "token_id"], axis=1).\
                astype({"token_id": int}).\
                groupby("token_id").sum().\
                log_p.apply(lambda x: x.item())
            return loss_ce_agg_per_disease / pass_tokens.sum().item()
        elif agg is None:
            loss_ce = F.cross_entropy(
                logits.reshape(-1, n_classes)[pass_tokens], 
                targets[pass_tokens], 
                ignore_index=-1
            )
        else:
            raise ValueError("agg should be in [None, 'per_disease']")

        return loss_ce
                                  


    def time_to_event_loss(self, logits, time_to_next, pass_tokens, attn_mask, mask_ties, t_min, agg=None):       
        '''
        '''
        
        lse = torch.logsumexp(logits,-1) ## More forgiving than using torch.max() for the most likely next event
        lse = - torch.log(torch.exp(-lse) + t_min)

        dt  = torch.clamp(time_to_next, min=1.0)
        dd = dict(device=logits.device, dtype=torch.float32)
        block_size = attn_mask.size(-1)

        if mask_ties:
            # Use time from last untied token
            dt = torch.gather(
                dt, -1, (attn_mask * torch.arange(0, block_size, **dd).view(1, 1, 1, -1)).max(-1).indices.squeeze((1, 2))
            )  

        log_dt = - torch.log(dt + t_min).view(-1)        

        ## Exponential log-likelihood (real statistics, TM)
        loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - log_dt.reshape(-1))) 

        if agg is None:
            pass
        elif agg == "mean":
            loss_dt = (loss_dt[pass_tokens]).mean()
        elif agg == "sum":
            loss_dt = (loss_dt[pass_tokens]).sum()
        elif agg == "per_disease":
            raise NotImplementedError
          
        return loss_dt
    

    def get_allowed_tokens_mask(self, targets, ignore_tokens):

        targets = targets.reshape(-1)
        pass_tokens = targets != -1 
        for k in ignore_tokens: # and gender
            pass_tokens *= targets != k

        return pass_tokens

    def set_valid_loss_mode(self, validation_loss_mode):
        self.validation_loss_mode = validation_loss_mode


    def blackout_ignored(self, logits, ignore_tokens):

        if self.validation_loss_mode:
            ignore_tokens += [1]
            logits[..., ignore_tokens] = -torch.inf

        return logits

        
    def forward(self, token_stream, age, targets=None, targets_age=None, validation_loss_mode=False, return_attentions=False, return_embeddings=False):
        """
        If return_embeddings=True, the method will return the final hidden states (embeddings) before the logits.
        The return value will be a tuple: (logits, loss, att, embeddings)
        Otherwise, the return value is (logits, loss, att) as before.
        """

        device = token_stream.device
        b, t = token_stream.size()
        ignore_tokens = self.config.ignore_tokens.copy()
        t_min = self.config.t_min
        
        self.set_valid_loss_mode(validation_loss_mode)
        
        # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(token_stream) # token embeddings of shape (b, t, n_embd)
        age_emb = self.transformer.wae(age.unsqueeze(-1)) # age embeddings of shape (b, t, n_embd)

        x = self.transformer.token_drop(tok_emb) * (1-self.config.token_dropout) 
        x = x + age_emb
        x = self.transformer.drop(x)

        mask_ties = self.config.mask_ties

        attn_mask = self.build_attention_mask(token_stream, age, targets, targets_age, mask_ties) 
        x, att = self.transformer_block(x, attn_mask, return_attentions=return_attentions)
        embeddings = x if return_embeddings else None
        
        if targets is None:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, :, :]) # note: using list [-1] to preserve the time dim
            loss = None
        else:
            # if we are given some desired targets also calculate the loss
            # next token cross entropy loss, padding masked
            logits = self.lm_head(x)
                
            # mask for logits' rows containing predicted tokens (to be included in the loss)
            pass_tokens = self.get_allowed_tokens_mask(targets, ignore_tokens)

            '''
            targets = targets.reshape(-1)
            pass_tokens = targets != -1
            for k in ignored_tokens: # and gender
                pass_tokens *= targets != k
            '''
               
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss_ce = self.cross_entropy_loss(logits_flat, targets_flat, ignore_tokens)
            loss_dt = self.time_to_event_loss(logits, targets_age-age, pass_tokens, attn_mask, mask_ties, t_min, agg='mean')

            '''
            # time to next event loss, padding masked
            lse = torch.logsumexp(logits,-1) ## More forgiving than using torch.max() for the most likely next event
            lse = - torch.log(torch.exp(-lse) + self.config.t_min)
            dt  = torch.clamp(targets_age - age, min=1.0)
            if self.config.mask_ties:
                # Use time from last untied token
                dt = torch.gather(
                    dt, -1, 
                    (attn_mask * torch.arange(0, token_stream.size(1), device=device, dtype=torch.float32).\
                     view(1, 1, 1, -1)).\
                     max(-1).\
                     indices.\
                     squeeze((1, 2))
                )  
            ldt = - torch.log(dt + self.config.t_min).view(-1)
            
            loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1))) ## Exponential log-likelihood (real statistics, TM)
            loss_dt = torch.mean(loss_dt[pass_tokens]) 
            
            # Both losses combined
            # loss = loss_ce + loss_dt
            '''

            loss = {'loss_ce': loss_ce, 'loss_dt': loss_dt}
            
            #loss += 5.0 * F.mse_loss(lse.view(-1)*(ldt != 0), ldt) ## Adds MSE for log time difference to next observed event
            
        return logits, loss, att, embeddings


    def get_embeddings(self, idx, age, targets=None, targets_age=None, validation_loss_mode=False, return_attentions=True, return_embeddings=True):
        """
        If return_embeddings=True, the method will return the final hidden states (embeddings) before the logits.
        The return value will be a tuple: (logits, loss, att, embeddings)
        Otherwise, the return value is (logits, loss, att) as before.
        """
        return self(idx, age, targets, targets_age, validation_loss_mode, return_attentions, return_embeddings)[-1]


    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
            
    def adjust_block_size(self, block_size):
        for block in self.transformer.h:
            block.attn.bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    @torch.no_grad()
    def generate(self, idx, age, max_new_tokens=100, max_age=85*365.25, no_repeat=True, termination_tokens=None, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Selected parameters:
        --------------------

        termination_tokens: list[int] -  a list of tokens that indicate the and of the trajectory.
        Usually it is the "Death" token, but could be several tokens e.g. to indicate different
        death reasons.
        top_k: None, does nothing
        """
        if termination_tokens is None:
            warnings.warn('When using a custem dataset, consider changing the `termination_tokens` argument.')
            termination_tokens = [1269]
        
        termination_tokens = torch.tensor(termination_tokens, dtype=torch.int64, device=idx.device)
        mask_time = -10000

        if max_new_tokens == -1:
            max_new_tokens = 128

        for _ in range(max_new_tokens):
            logits, _, _, _ = self(idx, age)
            logits = logits[:, -1, :]
            logits[:,self.config.ignore_tokens] = -torch.inf

            if no_repeat:
                fill = idx.clone()
                fill[fill == 1] = 0
                logits = logits.scatter_(1, fill, -torch.inf)
            
            # sample from exponential distributions for each disease using the inverse CDF method, then take min
            t_next = torch.clamp(-torch.exp(-logits) * torch.rand(logits.shape, device=idx.device).log(), min=0, max=365*80).min(1)
            idx_next = t_next[1][:,None] # the index of the min sampled time
            age_next = age[...,[-1]] + t_next[0][:,None] # the value of the min sampled time
            
            print(age_next)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)
            
            if torch.logical_or(torch.isin(idx, termination_tokens).any(-1), age_next > max_age).all():
                break
        
        is_termination_token = (torch.cumsum(torch.cumsum(torch.isin(idx, termination_tokens), 1).bool().int(), 1) > 1)
        print(f"{is_termination_token=}")
        
        overage = age > max_age
        print(f"{overage=}")
        
        pad = is_termination_token + overage
        
        print(f"{pad=}")

        logits, _, _, _ = self(idx, age)
        idx[pad] = 0
        age[pad] = mask_time

        if no_repeat:
            fill = idx + 0
            fill[fill == 1] = 0
            logits = torch.stack([logits[:,j].scatter_(1, fill[:,:j+1], -torch.inf) for j in range(fill.shape[1])]).transpose(0,1)

        return idx, age, logits


    @classmethod
    def from_checkpoint(cls, ckpt_path, device=None):
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(ckpt_path, map_location=device)        
        conf = DelphiConfig(**checkpoint['model_args'])        
        model = Delphi(conf)
        
        state_dict = checkpoint['model']
        state_dict = { k.replace("_orig_mod.", ""): v for k, v in state_dict.items() }

        model.load_state_dict(state_dict)         
        model = model.to(device)

        return model