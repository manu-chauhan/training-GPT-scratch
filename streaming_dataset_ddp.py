import inspect
import os 
import math 
import platform
import sys
import time 
import tiktoken 
from dataclasses import dataclass 
import torch 
import torch.nn as nn 
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from src import HindiTokenizer
import utilities

parallel_in_block = False
apply_first_skip_at_last_h_block = False 
add_org_x_before_lm_head = False

ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    print("\nDDP")
    assert torch.cuda.sis_available(), "CUDA not available"
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    decide = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_head = config.n_head
        n_embd = config.n_embd
        
        assert n_embd % n_head == 0
        
        # query, key, value prjections all combined
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        
        # output projection, after `v` is already multiplied with attention_scores
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        block_size = config.block_size
        
        self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        
        self.n_embd = n_embd
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size() # batch_size, sequence_len, embedding_dim (n_embd)
        # total dim = n_head * head_size
        # example GPT2 has 12 heads with each hs = 64 thus C= 12*64 = 768

        qkv = self.c_attn(x) # get combined qkv matix B, T, n_embd * 3(768*3=2304)

        q, k, v = qkv.split(self.n_embd, dim=2) # each item gets n_embd size, dimension against two 

        # b, seq, n_embd -> b, seq, n_heads, head_size -> b, n_heads, seq_len, head_size
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        # final-> bs, n_heads, seq_len, mini-n_head_embd

        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        
        # print(f"shape of q: {q.shape}... shape of k : {k.shape}")
        
        attn = (q @ k.transpose(-2, -1))/(math.sqrt(k.shape[-1]))

        # apply masked fill at places where mask ==0, remember tril is lower triangle
        attn = attn.masked_fill(mask = self.bias[ : , : , :T, :T] == 0, value=float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        y = attn @ v # B, n_heads, T/seq, T @ B, n_heads, T/Seq, head_size) -> B, n_heads, T, head_size

        # transpose y to merge all n_heads. B, n_heads, T, head_size -> transpose B, T, n_heads, head_size -> view B, T, Channel_size/n_emb 768 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # out projection, B, T, C -> B, T, C
        y = self.c_proj(y)
        
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):

        if parallel_in_block:
            x = x + self.attn(self.ln_1(x)) + self.mlp(self.ln_2(x))
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)
            ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size() # batch , seq_len

        # check if incoming seq_len of idx is within limits
        assert T <= self.config.block_size, f"Cannot proceed as your Sequence len : {T} is more than {self.config.block_size}"

        # forward for token and position encodings
        # shape (T)
        pos = torch.arange(0, T, dtype=torch.int32, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # position embds of shape (T, n_embd)
        token_emb = self.transformer.wte(idx) # token embds of shape (Batch, T/seq_len, n_embd)

        x = pos_emb + token_emb
        
        x_org = x.clone()

        # now forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)
               
        # pass through final layernorm
        x = self.transformer.ln_f(x)

        if add_org_x_before_lm_head:
            x = nn.LayerNorm(x + x_org)

        # pass through final LM_HEAD
        logits = self.lm_head(x) # shape (Batch_size, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


@dataclass
class GPTConfig:
    n_layer:int = 12
    block_size: int = 1024
    vocab_size: int = 50304
    n_head : int = 12
    n_embd: int = 768


class DataLoader:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.over = False
        
        # read data here
        # with open('input.txt', 'r') as f:
        #     data = f.read()
    
        # encode the data
        # enc = tiktoken.get_encoding('gpt2')
        tokenizer = HindiTokenizer()
        tokenizer.load("saved_vocabs/batch_80_Hindi_EN_Bengali_Tokenizer-test-all_batches-160_000_batchsize-initial_vocab_size_5000.model")
        
       
        
        def _streaming_batch_encode():
            all_files = utilities.get_all_text_dataset("dataset")
            
            if not len(all_files)> 0:
                raise FileNotFoundError("dataset not found")
            
            result = utilities.read_from_all_files(all_files, batch_size=B, batch_num=None)
            text = ""
            while True:
                for data_batch in result:
                    # print(data_batch)
                    text += " ".join(data_batch)
                    
                    encoded_text = tokenizer.encode(text)
                    # print(encoded_text)

                    if len(encoded_text) > (B * T * num_processes)+1:
                        to_yeild = encoded_text[: (B *T * num_processes)+1]
                        yield torch.tensor(to_yeild, device=device)
                        text = text[len(to_yeild):]  
                self.over = True

        # assign tokens from encoded data
        self.tokens = _streaming_batch_encode()
        
        # state
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T

        # buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        buf = next(self.tokens)
        
        print(f"=====buf len: {len(buf)}")

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        self.current_position += B * T * self.num_processes

        # # if self.current_position + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
        # if self.over:
        #     # self.current_position = self.B * self.T * self.process_rank
        #     self.tokens = self._streaming_batch_encode()
        if self.current_position + (self.B * self.T * self.num_processes + 1) > len(buf):
            self.current_position = self.B * self.T * self.process_rank

        return x, y 
    
max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 10
max_steps = 5000

def get_lr(iteration):
    if iteration < warmup_steps:
        return max_lr * (iteration + 1) / warmup_steps
    if iteration > max_steps:
        return min_lr
    
    decay_ratio = (iteration - warmup_steps) / (max_steps - warmup_steps)

    assert 0<= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

total_batch_size = 1024
B = 4
T = 32
grad_acum_steps = total_batch_size // (B*T * ddp_world_size)

train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)


torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig()).to(device=device)

# Get the name of the OS
os_name = platform.system()
if "nix" in os_name:
    model = torch.compile(model, mode='default')

if ddp:
    print("\n\n====================================\nDDP")
    model = DDP(module=model,device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)

if master_process:
    print(f"\ntotal batch size: {total_batch_size}")
    print(f"\n gradient accumulation steps: {grad_acum_steps}")

for step in range(50):
    t0 = time.time()
    
    optimizer.zero_grad()

    grad_acum_loss = 0.0

    for micro_step in range(grad_acum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # print(f"\n==== data at : {step} and micro step : {micro_step} \n:X:\n {x}\ny:\n: {y}")
        # print(x.shape)
        # sys.exit(0)
        with torch.autocast(device_type='cpu', dtype=torch.float16):
            logits, loss = model(x, y)
        
        loss = loss/grad_acum_steps
        
        grad_acum_loss += loss.detach()
        
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_acum_steps - 1)

        loss.backward() # adds up gradients += 
    
    if ddp:
        dist.all_reduce(grad_acum_loss, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    t1 = time.time()
    
    dt = t1-t0

    tokens_processed = train_loader.B * train_loader.T * grad_acum_steps * ddp_world_size
    tokens_per_sec = tokens_processed/dt
    if master_process:
        print(f"step: {step} | time: {dt:.4f} | loss: {grad_acum_loss.item():.6f} | norm: {norm:4f} | dt: {dt * 1000} ms | tokens_per_sec: {tokens_per_sec} ")

if ddp:
    destroy_process_group()
