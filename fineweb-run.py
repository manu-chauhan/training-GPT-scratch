# DDP

import os
import math

import numpy as np
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

import hellaswag
from hellaswag import iterate_examples, render_example

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


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


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


def load_tokens(filename):
    np_tokens = np.load(filename)
    torch_tokens = torch.tensor(np_tokens, dtype=torch.long)
    return torch_tokens


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {'train', 'val'}

        # get shard filenames
        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]

        self.shards = shards

        assert len(shards) > 0, f"no shards found for the provided split: {split}"

        if master_process:
            print(f"\nFound {len(shards)} for split {split}")

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + (B * T) + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # now each advancement is by B*T * total processes as we slide to new window to fetch New Tokens,
        # i.e. moving by an entire chunk of window at a time
        self.current_position += (B * T * self.num_processes)

        # now check if loading next batch leads to out of bounds here then move to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(filename=self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and
                                                                            torch.backends.mps.is_available() else 'cpu')
    model = GPT(GPTConfig()).to(device=device)
    to_compile = False

    if to_compile:
        model = torch.compile(model, mode='default')

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)

    total_batch_size = 544288  # 2**19 = 5,24,288, tokens
    B = 14
    T = 1024

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
    enc = tiktoken.get_encoding('gpt2')

    gradient_accumulation_steps = total_batch_size // (B * T * ddp_world_size)

    if master_process:
        print(f"\nTotal effective batch size: {total_batch_size}")
        print(f"\nGrad accumulation steps: {gradient_accumulation_steps}")

    max_lr = 6e-4 * 3
    min_lr = max_lr * 0.15
    warmup_steps = 100  # 715  # GPT3 paper warms up LR over 375 million tokens ; 375e6/2^19 = 715.2
    max_steps = 5000  # 19_703  # steps result of 10^9/2^19 (2^19 tokens from total_batch_size and 10^9 tokens in fineweb dataset)


    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    with open(log_file, 'w') as f:
        pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        if step % 500 == 0 or last_step:
            model.eval()
            val_loader.reset()

            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 500 == 0 or last_step):
                    # save model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    torch.save(checkpoint, checkpoint_path)

        # generate some stuff once in a while
        if (step > 1 and step % 500 == 0) or last_step:
            model.eval()
            num_return_sequences = 3
            max_length = 32

            tokens = enc.encode("Hello, I am a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

            x_generate = tokens.to(device)

            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)

            while x_generate.size(1) < max_length:
                with torch.no_grad():
                    logits, loss = model(x_generate)

                    # get logits from last position
                    logits = logits[:, -1, :]  # this is (Batch , vocab_size) dim now

                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)

                    # perform topk sampling with size 50, HF default is 5, 50
                    topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)

                    # select a random token from topk, this is (Batch_size, 1) dim
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)

                    xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)

                    x_generate = torch.cat([x_generate, xcol], dim=1)

                # print generated sequence
                for i in range(num_return_sequences):
                    tokens = x_generate[i, :max_length].tolist()
                    decoded = enc.decode(tokens)
                    print(f"rank: {ddp_rank} sample {i} : {decoded}")

        if step % 250 == 0 or last_step and not to_compile:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue
                # render examples into tokens and labels

                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get logits
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()

            acc_norm = num_correct_norm / num_total

            if master_process:
                print(f"HellaSwag accuracy == {num_correct_norm}/{num_total} = {acc_norm:.4f}")
                with open(log_file, 'a') as f:
                    f.write(f"\n{step} hellaswag accuracy: {acc_norm:.4f}")

        # normal training code
        model.train()
        optimizer.zero_grad()

        model.require_backward_grad_sync = False  # being explicit to disable grad sync as we want to accumulate first

        loss_accumulator = 0.0
        for micro_step in range(gradient_accumulation_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device=device), y.to(device)

            if ddp:
                model.require_backward_grad_sync = (
                        micro_step == gradient_accumulation_steps - 1)  # enable only on last step

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)

            loss = loss / gradient_accumulation_steps

            loss_accumulator += loss.detach()

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
                f'step : {step + 1} | loss: {loss_accumulator.item()} | dt: {dt * 1000:.2f} ms | tokens/sec: {tokens_per_sec:_.2f} | LR:{lr:.6f} | norm:{norm:.3f}')

            with open(log_file, 'a') as f:
                f.write(f"\n{step} train  {loss_accumulator.item():.6f}")

if ddp:
    destroy_process_group()
