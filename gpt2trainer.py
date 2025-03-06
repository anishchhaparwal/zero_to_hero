from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import math
import tiktoken
import time 
import inspect

@dataclass
class GPTConfig:
    block_size:int = 1024
    vocab_size:int = 50257  # ugly number we want to replace with a power of 2. eg: 50304  
    n_layer:int = 12 # no of attention block
    n_head:int = 12 # attention head
    n_embd:int = 768 # embedding size
    
class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.gpt2__init = 1
        # this is masked encoding so model doesnt have context from future. using bias to follow naming convetion from openai.
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() # B no of egs, T is batch_size, C is n_embd
        qkv = self.c_attn(x)
        q, k ,v = qkv.split(self.n_embd, 2) # B T C
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # B, n_head, T, n_embd/n_heads
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        
        # flash attention to speed up. torch.compile currently doesnt optamize this. it should.
        # y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias[:,:,:T,:T], dropout_p=0.0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # attn = (q @ k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1))) # (B, n_head, T, n_embd/n_heads) @ (B, n_head,  n_embd/n_heads, T) = (B, n_head, T, T)
        # attn = attn.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        # attn = F.softmax(attn, dim=-1)
        # y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y        
        
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.gpt2__init = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)    
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer=nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # input and logits embedding should be same. same input distribution, same output distribution.
        self.transformer.wte.weight = self.lm_head.weight
        
        # init params:
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'gpt2__init'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets = None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"T cannot be greater than block size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        
        tok_embd = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        pos_embd = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        x = tok_embd + pos_embd
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        # n_layer, n_head and n_embd are determined from model_type
        # config_args = {
        #     'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        #     'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        #     'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        #     'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        # }[model_type]
        # config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        # config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        
        config = GPTConfig()
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizer(self, weight_decay, learning_rate, device):
        # all paramas that require grad
        param_dict = {pn:p for pn,p in self.named_parameters() if p.requires_grad}
        
        # we want to decay all embedding and matmuls w+b but not for layernorm.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay':  0.0}
        ]
        # used fused which should be used by default to speed things up but not always available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters  
        
        print(f"num decayed: {sum(p.numel() for p in decay_params):,} | num non-decayed: {sum(p.numel() for p in nondecay_params):,} | setting fused in optim to: {fused_available}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8, fused=fused_available)
        return optimizer
    
# -----------------------------------------
# dataloader
class Dataloaderlite():
    def __init__(self, batch, block_size):
        self.B = batch
        self.T = block_size
        
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, device=device)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {(len(self.tokens) // (self.B * self.T))} batches")

        self.current_position = 0
        
    def next_batch(self):
        buf = self.tokens[self.current_position: self.current_position + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        
        self.current_position += self.B * self.T 
        
        if self.current_position + (self.B * self.T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y         
# -----------------------------------------
# get_lr

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10 # 715
max_steps = 50 # 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# -----------------------------------------
torch.set_float32_matmul_precision('high')
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

device="cpu"
if torch.cuda.is_available():
    device="cuda"

# model init
model = GPT(GPTConfig(vocab_size=50304))
model.eval()
model.to(device)
model = torch.compile(model)

total_batch_size  = 2**19
no_of_eg = 16
context_window = 1024
assert total_batch_size % (no_of_eg * context_window) == 0, "make total_batch_size divisible by no_of_eg * context_window"
grad_accum_steps = total_batch_size // (no_of_eg * context_window)
data = Dataloaderlite(batch=16, block_size=1024)

# optimizer = torch.optim.AdamW(model.parameters(),lr = 3e-4, betas=(0.9,0.95), eps=1e-8)
optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

# training steps
for step in range(max_steps):
    t0=time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = data.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss /grad_accum_steps # we do this because in normal F.cross_entropy we take the mean value. here we are accumulating across multiple batches so we devide by no of step we are accumunating by,
        loss_accum += loss.detach()
        loss.backward()
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determining lr
    lr = get_lr(step)
    for param in optimizer.param_groups:
        param['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time diff in ms
    tokens_processed = data.B * data.T * grad_accum_steps
    tokens_per_second = tokens_processed / (t1 - t0) 
    print(f" step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_second:.2f}")

print("bye")
# model = model.to('cpu')
# del model
# torch.cuda.empty_cache()
# print(torch.__version__)
# import sys;``
# sys.stdout.flush()
# sys.stderr.flush()
# sys.exit(0)


# resampling_count = 5
# max_sample_length=30
# import tiktoken 
# enc=tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(resampling_count, 1)
# print(f"shape of tokens {tokens.shape}")
# x = tokens.to(device)

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_sample_length:
#     with torch.no_grad():
#         logits = model(x)
#         last_timestamp = logits[:,-1,:]
#         probs = F.softmax(last_timestamp, dim=-1)
#         topk_probs, topk_indices = torch.topk(probs, 50,-1)
#         ix = torch.multinomial(topk_probs,1)
#         xcol = torch.gather(topk_indices, -1, ix)
#         x = torch.cat((x,xcol), dim=1)
        
# for i in range(resampling_count):
#     tokens = x[i,:max_sample_length].tolist()
#     decode = enc.decode(tokens)
#     print(f"> {decode}")