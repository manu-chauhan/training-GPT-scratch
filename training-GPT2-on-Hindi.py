# DDP

import hindi
import utilities
import os
import math
import tiktoken
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

torch.set_float32_matmul_precision('high')

ddp = int(os.environ.get('RANK', -1)) != -1  # to check if this is a ddp run

if ddp:
    print('\nDDP==============')
    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE!"
    init_process_group(backend='nccl')
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
    device = 'cpu'
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
        self.c_proj.NANOGPT_SCALE_INIT = 1

        block_size = config.block_size

        self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        self.n_embd = n_embd
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()  # batch_size, sequence_len, embedding_dim (n_embd)
        # total dim = n_head * head_size
        # example GPT2 has 12 heads with each hs = 64 thus C= 12*64 = 768

        qkv = self.c_attn(x)  # get combined qkv matix B, T, n_embd * 3(768*3=2304)

        q, k, v = qkv.split(self.n_embd, dim=2)  # each item gets n_embd size, dimension against two

        # b, seq, n_embd -> b, seq, n_heads, head_size -> b, n_heads, seq_len, head_size
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # final-> bs, n_heads, seq_len, mini-n_head_embd

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # print(f"shape of q: {q.shape}... shape of k : {k.shape}")

        # attn = (q @ k.transpose(-2, -1)) / (math.sqrt(k.shape[-1]))
        #
        # # apply masked fill at places where mask ==0, remember tril is lower triangle
        # attn = attn.masked_fill(mask=self.bias[:, :, :T, :T] == 0, value=float('-inf'))
        #
        # attn = F.softmax(attn, dim=-1)
        #
        # y = attn @ v  # B, n_heads, T/seq, T @ B, n_heads, T/Seq, head_size) -> B, n_heads, T, head_size

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # Flash attention

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
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd)
            ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weights = self.lm_head.weight

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()  # batch , seq_len

        # check if incoming seq_len of idx is within limits
        assert T <= self.config.block_size, f"Cannot proceed as your Sequence len : {T} is more than {self.config.block_size}"

        # forward for token and position encodings
        # shape (T)
        pos = torch.arange(0, T, dtype=torch.int32, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # position embds of shape (T, n_embd)
        token_emb = self.transformer.wte(idx)  # token embds of shape (Batch, T/seq_len, n_embd)

        x = pos_emb + token_emb

        # now forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # pass through final layernorm
        x = self.transformer.ln_f(x)

        # pass through final LM_HEAD
        logits = self.lm_head(x)  # shape (Batch_size, T, vocab_size)

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
    block_size: int = 1024  # this is max sequence len
    vocab_size: int = 50304  # total vocab including 256 bytes + 1 special token (<|endoftext|>) and 1000-257 BPE merges
    n_layer: int = 12  # number of layers
    n_head: int = 12  # total number of attention heads
    n_embd: int = 768  # embedding dimension


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        all_files = utilities.get_all_text_dataset("dataset")
        print(f"\nDATALOADERLITE...")

        # with open('input.txt', 'r') as f:
        #     text = f.read()

        result = utilities.read_from_all_files(all_files)
        # enc = tiktoken.get_encoding('gpt2')
        text = ""
        for i, data_batch in enumerate(result):
            print(f"\n\n:{i}")
            print(len(data_batch))
            batch_text = "".join(data_batch)
            batch_text = batch_text.split(' ')[:T]
            mini_text = ''
            for tok in batch_text:
                mini_text += ' ' + tok
            text += mini_text
            if i > B:
                break
        enc = hindi.load_tokenizer()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded len : {len(self.tokens)}')
        # print(f'1 epoch = :{len(self.tokens)}//{B * T} = {len(self.tokens) // (B * T)} batches ')

        # modified to accommodate multiple processes
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + (B * T) + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # now each advancement is by B*T * total processes as we slide to new window to fetch New Tokens,
        # i.e. moving by an entire chunk of window at a time
        self.current_position += (B * T * self.num_processes)

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y


if __name__ == "__main__":

    max_lr = 6e-4*3
    min_lr = max_lr * 0.15
    warmup_steps = 100
    max_steps = 5000


    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and
                                                                            torch.backends.mps.is_available() else 'cpu')
    model = GPT(GPTConfig()).to(device=device)

    total_batch_size = 5_24_288  # 2**19 = 5,24,288, tokens
    B = 20
    T = 1024

    gradient_accumulation_steps = total_batch_size // (B * T * ddp_world_size)

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

    # Load your custom tokenizer
    tokenizer = hindi.load_tokenizer()  # Assuming your load_tokenizer() function is defined and returns the tokenizer object

    # # Path to the directory containing your .txt files
    # dataset_dir = 'dataset'

    # # Create the dataset
    # train_dataset = HindiTextDataset(directory=dataset_dir, tokenizer=tokenizer)

    # # Create the DataLoader with DistributedSampler
    # train_sampler = DistributedSampler(train_dataset)
    # train_loader = DataLoader(train_dataset, batch_size=B, sampler=train_sampler)

    if master_process:
        print(f"\nTotal effective batch size: {total_batch_size}")
        print(f"\nGrad accumulation steps: {gradient_accumulation_steps}")

    # try:
    #     model = torch.compile(model, mode='max-autotune')
    # except Exception as e:
    #     model = torch.compile(model, mode='default')
    model = torch.compile(model, mode='default')

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)

    for step in range(max_steps):
        # for idx, mini_batch in enumerate(train_loader):
        t0 = time.time()
        optimizer.zero_grad()

        model.require_backward_grad_sync = False  # being explicit to disable grad sync as we want to accumulate first

        loss_accumulator = torch.zeros(size=(1, 1), device=device)
        for micro_step in range(gradient_accumulation_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device=device), y.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)

            loss = loss / gradient_accumulation_steps

            loss_accumulator += loss.detach()

            if ddp:
                model.require_backward_grad_sync = (
                        micro_step == gradient_accumulation_steps - 1)  # enable only on last step

            loss.backward()  # keep updating grad here (adding up)

        # average loss_accumulator now
        if ddp:
            # fetches loss_accumulator from all ranks, averages those and updates all
            dist.all_reduce(loss_accumulator, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        lr = get_lr(step)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        t1 = time.time()
        dt = (t1 - t0)

        tokens_per_sec = (train_loader.B * train_loader.T * gradient_accumulation_steps * ddp_world_size) / (dt)

        if master_process:
            print(
                f'step : {step + 1} | loss: {loss_accumulator.item()} | lr: {lr} | dt: {dt * 1000:.2f} ms | tokens/sec: {tokens_per_sec:_.2f} | norm:{norm:.3f}')
            if step>100 and step% 50 == 0:
                chk = {'model':model.state_dict()}
                torch.save(chk, f"model_epoch_{step}.pth")
if ddp:
    destroy_process_group()
