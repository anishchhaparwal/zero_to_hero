# numactl --physcpubind=64-96 torchrun --standalone --nproc_per_node=3 trainer.py

from dataclasses import dataclass
import math
import tiktoken
import time 
import inspect
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from clearml import Task
   
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
        
        # print(f"num decayed: {sum(p.numel() for p in decay_params):,} | num non-decayed: {sum(p.numel() for p in nondecay_params):,} | setting fused in optim to: {fused_available}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8, fused=fused_available)
        return optimizer
        
    def get_num_params(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
        
    def estimate_mfu(self, fwdbwd_per_iter, dt, world_size=1):
        """Estimate model flops utilization (MFU) in units of 3090 GPU FLOPs."""
        # First estimate the number of flops we do per iteration
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # Express flops throughput as ratio of 3090 FLOPs peak
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 142e12  # 142 TFLOPS per 3090 GPU
        mfu = flops_achieved / flops_promised
        return mfu

class FineWebIterableDataset(IterableDataset):
    """PyTorch IterableDataset for FineWeb data that yields batches continuously."""
    
    def __init__(self, data_dir, batch_size, block_size, 
                 process_id, num_processes, split='train'):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.block_size = block_size
        self.process_id = process_id # ddp_rank / global rank
        self.num_processes = num_processes
        
        # Get shard filenames
        shards = [f for f in os.listdir(data_dir) if split in f and f.endswith('.npy')]
        self.shards = sorted([os.path.join(data_dir, s) for s in shards])
        
        assert len(self.shards) > 0, f"No shards found for split {split} in {data_dir}"
        if process_id == 0:
            print(f"Found {len(self.shards)} shards for split {split}")
    
    def __iter__(self):
        # Handle worker_info for DataLoader's multiprocessing
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:  # single-process loading
            worker_id = 0
            num_workers = 1
        else:  # multi-process loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        # Calculate the effective process ID and total processes
        
        # to distribute data uniquely across all processes and workers, avoiding overlap.
        # here effective_process_id is setting the starting point of each iter in a uniform manner on a line that loops.
        effective_process_id = self.process_id * num_workers + worker_id 
        
        # previously only one gpu. now each gpu has multiple num_processes so we move those many blocks ahead.
        # Total number of gpus across cluster * number of processes per gpu
        effective_num_processes = self.num_processes * num_workers # total no of processes across all gpus. so think stride.
       
        # Initial state
        shard_idx = 0
        tokens = np.load(self.shards[shard_idx], mmap_mode='r')
        position = self.batch_size * self.block_size * effective_process_id 
        
        
        # The code has two distinct phases:
        # Initialization phase: Everything before the while True loop. personal init for each worker.
        # Working phase: Actual loading of data from shrads.
        while True:
            B, T = self.batch_size, self.block_size
            
            # Check if we need to load the next shard
            if position + (B * T + 1) > len(tokens):
                shard_idx = (shard_idx + 1) % len(self.shards)
                tokens = np.load(self.shards[shard_idx], mmap_mode='r')
                position = B * T * effective_process_id
            
            # Get batch of tokens
            buf_np = tokens[position:position + B*T + 1]
            
            # Sanity check - should rarely happen
            if len(buf_np) < B*T + 1:
                shard_idx = (shard_idx + 1) % len(self.shards)
                tokens = np.load(self.shards[shard_idx], mmap_mode='r')
                position = B * T * effective_process_id
                continue
            
            # Convert to PyTorch tensor
            buf = torch.tensor(buf_np.astype(np.int32) , dtype=torch.long)
            
            # Create input and target batches (B, T)
            x = buf[:-1].reshape(B, T)
            y = buf[1:].reshape(B, T)
            
            yield x, y
            
            # Advance position for next batch, respecting distributed processes
            position += B * T * effective_num_processes

class DataLoaderLitePyTorch:
    """
    Drop-in replacement for DataLoaderLite using PyTorch DataLoader.
    
    Features:
    - Uses PyTorch's efficient DataLoader with multi-processing
    - Memory maps files to reduce RAM usage
    - Pin memory for faster CPU->GPU transfer
    - Automatic shard preloading
    """
    
    def __init__(self, B, T, process_rank, num_processes, split="train", 
                 data_dir="/cache/fast_data_nas8/llm_setup_dataset/fineweb_10b", num_workers=2):
        """
        Args:
            B: Batch size
            T: Sequence length (context length)
            process_rank: Process rank for distributed training
            num_processes: Number of processes for distributed training
            split: 'train' or 'val'
            data_dir: Directory containing the data shards
            num_workers: Number of DataLoader workers for parallel loading
        """
        self.B = B
        self.T = T
        self.split = split
        
        # Create dataset
        dataset = FineWebIterableDataset(
            data_dir=data_dir,
            split=split,
            batch_size=B,
            block_size=T,
            process_id=process_rank,
            num_processes=num_processes
        )
        
        # Create dataloader with PyTorch's parallel loading capabilities
        self.dataloader = DataLoader(
            dataset,
            batch_size=None,  # Already batched in dataset
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        # Initialize iterator
        self.reset()
    
    def next_batch(self):
        """Get the next batch of data, matching original DataLoaderLite interface."""
        try:
            return next(self.iterator)
        except StopIteration:
            # This should rarely happen with our infinite IterableDataset
            self.reset()
            return next(self.iterator)
    
    def reset(self):
        """Reset the data loader (creates a new iterator)."""
        self.iterator = iter(self.dataloader)

def worker_init_fn(worker_id):
    worker_seed = 1337 # torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# -----------------------------------------
# Performance metrics tracking
tokens_per_second_history = []
mfu_history = []

# -----------------------------------------
# DDP, Device, Seed
torch.set_float32_matmul_precision('high')
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # global rank of the process across all nodes and GPUs in the distributed setup
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # in the node which gpu are we refering too
    ddp_world_size = int(os.environ['WORLD_SIZE']) # total no of gpus across nodes and multiple cards
    device = f'cuda:{ddp_local_rank}' # Assign GPU based on local rank
    torch.cuda.set_device(device) # Set the current CUDA device
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"


# -----------------------------------------
# log
# create the log directory we will write checkpoints to and log to
task = Task.init(project_name='testing', task_name='test_run_5')


log_dir = "/cache/fast_data_nas8/llm_setup_dataset/model_runs/run_3"
os.makedirs(log_dir, exist_ok=True)
# Create histograms directory
if master_process:
    os.makedirs(os.path.join(log_dir, "histograms"), exist_ok=True)
    tb_writer = SummaryWriter(log_dir)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

def log_histograms(model, step, log_dir, tb_writer=None):
    if not master_process:
        return

    logger_dict = {}
    layer_types = {'attention': [], 'mlp': [], 'embeddings': []}

    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim >= 2:
            if 'h.0.' in name or 'h.5.' in name or 'h.11.' in name or 'wte' in name or 'lm_head' in name:
                stats = {'weight_mean': param.mean().item(), 'weight_std': param.std().item()}
                if param.grad is not None:
                    stats.update({
                        'grad_mean': param.grad.mean().item(),
                        'grad_std': param.grad.std().item(),
                        'ratio': param.grad.std().item() / param.std().item()
                    })
                logger_dict[name] = stats
                if 'attn' in name:
                    layer_types['attention'].append(stats)
                elif 'mlp' in name:
                    layer_types['mlp'].append(stats)
                else:
                    layer_types['embeddings'].append(stats)

    # Print averages per layer type
    for layer_type, stats_list in layer_types.items():
        if stats_list:
            avg_weight_std = sum(s['weight_std'] for s in stats_list) / len(stats_list)
            avg_grad_std = sum(s['grad_std'] for s in stats_list if 'grad_std' in s) / len([s for s in stats_list if 'grad_std' in s]) if any('grad_std' in s for s in stats_list) else 0
            avg_ratio = sum(s['ratio'] for s in stats_list if 'ratio' in s) / len([s for s in stats_list if 'ratio' in s]) if any('ratio' in s for s in stats_list) else 0
            print(f"{layer_type}: avg weight std {avg_weight_std:.4e}, avg grad std {avg_grad_std:.4e}, avg ratio {avg_ratio:.4e}")

    # TensorBoard logging
    if tb_writer:
        for name, stats in logger_dict.items():
            tb_writer.add_scalar(f"weights/mean/{name}", stats['weight_mean'], step)
            tb_writer.add_scalar(f"weights/std/{name}", stats['weight_std'], step)
            if 'grad_mean' in stats:
                tb_writer.add_scalar(f"grads/mean/{name}", stats['grad_mean'], step)
                tb_writer.add_scalar(f"grads/std/{name}", stats['grad_std'], step)
                tb_writer.add_scalar(f"grad_param_ratio/{name}", stats['ratio'], step)
                # Log the target ratio as a reference line
                tb_writer.add_scalar("grad_param_ratio/target", 1e-3, step)
# -----------------------------------------
# get_lr
max_steps = 17000 # 19073 # 16,955 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
max_steps = 1700
max_lr = 6e-4 * 3
min_lr = max_lr * 0.1
warmup_steps = min(620,int(max_steps*0.1)) # 715

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
@dataclass
class GPTConfig:
    block_size:int = 1024
    vocab_size:int = 50257  # ugly number we want to replace with a power of 2. eg: 50304  
    n_layer:int = 12 # no of attention block
    n_head:int = 12 # attention head
    n_embd:int = 768 # embedding size
    
# model init
total_batch_size  = 589824 #540672 #2**19
no_of_eg = 16
context_window = 1024


model = GPT(GPTConfig(vocab_size=50304, block_size=context_window))
model.eval()
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model


assert total_batch_size % (no_of_eg * context_window * ddp_world_size) == 0, "make total_batch_size divisible by no_of_eg * context_window * ddp_world_size"
grad_accum_steps = total_batch_size // (no_of_eg * context_window * ddp_world_size)
if master_process:
    print(f"total batch size: {total_batch_size}. total fwd&bck in one trip: {(no_of_eg * context_window * ddp_world_size)}, Total grad accum steps: {grad_accum_steps}")  
# print(f"gpu rank: {ddp_rank}")

# data = Dataloaderlite(batch=no_of_eg, block_size=context_window, global_rank_of_gpu= ddp_rank, total_gpu_across_cluster = ddp_world_size)
# https://grok.com/share/bGVnYWN5_ef9de139-74dd-40eb-a7ea-b62916fe41ea
# https://grok.com/chat/d58e0736-dd0b-4164-9adc-b584d4958a3c
train_loader  = DataLoaderLitePyTorch(
        B=no_of_eg,              # Batch size
        T=context_window,           # Sequence length
        process_rank=ddp_rank,   # Process rank for distributed training
        num_processes=ddp_world_size,  # Total number of gpus across cluster
        split="train",    # 'train' or 'val'
        num_workers=8     # DataLoader workers (0 for single-process)
    ) 
val_loader = DataLoaderLitePyTorch(B=no_of_eg, T=context_window, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", num_workers=2)

# optimizer = torch.optim.AdamW(model.parameters(),lr = 3e-4, betas=(0.9,0.95), eps=1e-8)
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

# -----------------------------------------
# evaluate_model
def save_checkpoint(model, optimizer, step, val_loss, log_dir):
    """
    Save model checkpoint.
    
    Args:
        model: The raw model (not DDP wrapped)
        optimizer: The optimizer
        step: Current training step
        val_loss: Validation loss
        log_dir: Directory to save checkpoint to
    """
    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
    checkpoint = {
        'model': model.state_dict(),
        'config': model.config,
        'optimizer': optimizer.state_dict(),
        'step': step,
        'val_loss': val_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


# training steps
for step in range(max_steps):
    t0=time.time()
    last_step = (step == max_steps - 1)
    
    # Validation and checkpoint logic
    # once in a while evaluate our validation loss
    if step % 25 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            
            # Log to TensorBoard
            if 'tb_writer' in locals() or 'tb_writer' in globals():
                tb_writer.add_scalar("loss/val", val_loss_accum.item(), step)
                
            
            # Log histograms less frequently to avoid overhead
            if step % 100 == 0 or last_step:
                log_histograms(raw_model, step, log_dir, tb_writer if 'tb_writer' in locals() or 'tb_writer' in globals() else None)

            # If we have performance metrics, report a summary
            if len(tokens_per_second_history) > 0:
                # Calculate performance stats
                avg_tokens_per_second = sum(tokens_per_second_history) / len(tokens_per_second_history)
                avg_mfu = sum(mfu_history) / len(mfu_history)
                max_tokens_per_second = max(tokens_per_second_history)
                max_mfu = max(mfu_history)
                
                # Log summary
                print(f"\nPERFORMANCE SUMMARY at step {step}:")
                print(f"Avg tokens/sec: {avg_tokens_per_second:.2f} | Max tokens/sec: {max_tokens_per_second:.2f}")
                print(f"Avg MFU: {avg_mfu:.2%} | Max MFU: {max_mfu:.2%}")
                
            if step > 0 and (step % 500 == 0 or last_step):
                save_checkpoint(
                    model=raw_model,  # Use raw_model, not DDP-wrapped model
                    optimizer=optimizer, step=step, val_loss=val_loss_accum.item(), log_dir=log_dir )

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp: # syncing loss across all gpu in all nodes only on grad_accum_steps -1 steps and not every step. 
            model.require_backward_grad_sync = (micro_step == (grad_accum_steps -1))
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss /grad_accum_steps # we do this because in normal F.cross_entropy we take the mean value. here we are accumulating across multiple batches so we devide by no of step we are accumunating by,
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determining lr
    lr = get_lr(step)
    for param in optimizer.param_groups:
        param['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time diff in ms
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_second = tokens_processed / (t1 - t0)
    
    if master_process:
        # Calculate MFU
        mfu = raw_model.estimate_mfu(fwdbwd_per_iter=train_loader.B * grad_accum_steps, dt=t1-t0, world_size=ddp_world_size)
        tokens_per_second_history.append(tokens_per_second)
        mfu_history.append(mfu)
        
        # Calculate rolling averages over last 50 steps
        window_size = min(50, len(tokens_per_second_history))
        avg_tokens_per_second = sum(tokens_per_second_history[-window_size:]) / window_size
        avg_mfu = sum(mfu_history[-window_size:]) / window_size
        
        # Print to console
        print(f" step {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_second:.2f} | avg tok/sec: {avg_tokens_per_second:.2f} | MFU: {mfu:.2%}")
        with open(log_file, "a") as f:
            f.write(f"{step} train loss={loss_accum.item():.6f} tok_per_sec={tokens_per_second:.2f} avg_tok_per_sec={avg_tokens_per_second:.2f} mfu={mfu:.6f} avg_mfu={avg_mfu:.6f}\n")
        
        # Log to TensorBoard
        if 'tb_writer' in locals() or 'tb_writer' in globals():
            tb_writer.add_scalar("loss/train", loss_accum.item(), step)
            tb_writer.add_scalar("loss/reference_gpt2_target", 2.85, step) # GPT-2 target is 2.85
            tb_writer.add_scalar("lr", lr, step)
            tb_writer.add_scalar("grad_norm", norm, step)
            tb_writer.add_scalar("performance/tokens_per_second", tokens_per_second, step)
            tb_writer.add_scalar("performance/mfu", mfu, step)
    
if ddp:
    destroy_process_group()

# Close TensorBoard writer
if master_process and 'tb_writer' in locals():
    tb_writer.close()