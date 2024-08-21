import os
import math
import tiktoken
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


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

        attn = (q @ k.transpose(-2, -1)) / (math.sqrt(k.shape[-1]))

        # apply masked fill at places where mask ==0, remember tril is lower triangle
        attn = attn.masked_fill(mask=self.bias[:, :, :T, :T] == 0, value=float('-inf'))

        attn = F.softmax(attn, dim=-1)

        y = attn @ v  # B, n_heads, T/seq, T @ B, n_heads, T/Seq, head_size) -> B, n_heads, T, head_size

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

    @classmethod
    def from_pretrained(cls, model_name):
        """for loading pre-trained GPT model weights from HuggingFace"""
        assert model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained GPT model: {model_name}")

        # n_layer, n_head, n_embd for each model_name
        config_args = {
            'gpt2': dict(n_layer=12, n_embd=768, n_head=12),  # has 124M params
            'gpt2-medium': dict(n_layer=24, n_embd=1024, n_head=16),  # 350M params
            'gpt2-large': dict(n_layer=36, n_embd=1280, n_head=20),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_embd=1600, n_head=25)  # 1558M params

        }[model_name]

        config_args['vocab_size'] = 50257  # same for all GPT2 checkpoints
        config_args['block_size'] = 1024  # max seq len 1024 for all GPT2 checkpoints

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [key for key in sd_keys if not key.endswith('.attn.bias')]  # discard this mask, not a parameter

        # initialize transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_name)
        sd_hf = model_hf.state_dict()
        sd_hf_keys = [key for key in sd_hf.keys() if
                      not key.endswith('.attn.masked_bias')]  # to discard, not a parameter
        sd_hf_keys = [key for key in sd_hf.keys() if not key.endswith('.attn.bias')]  # to discard, not a param

        # transposing these to match openai's Conv1d usage with Linear layer
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        print("=======\nDifference in keys: ", set(sd_keys) - set(sd_hf_keys))
        assert len(sd_keys) == len(
            sd_hf_keys), f"mismatched keys: sd_keys {len(sd_keys)} != sd_hd_keys {len(sd_hf_keys)}"

        for key in sd_hf_keys:
            if any(key.endswith(w) for w in transposed):
                assert sd_hf[key].shape[::-1] == sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key].t())
            else:
                # simple copy for other params
                assert sd_hf[key].shape == sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key])

        return model


@dataclass
class GPTConfig:
    block_size: int = 1024  # this is max sequence len
    vocab_size: int = 50257  # total vocab including 256 bytes + 1 special token (<|endoftext|>) and 1000-257 BPE merges
    n_layer: int = 12  # number of layers
    n_head: int = 12  # total number of attention heads
    n_embd: int = 768  # embedding dimension


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded len : {len(self.tokens)}')
        print(f'1 epoch = :{len(self.tokens)}//{B * T} = {len(self.tokens) // (B * T)} batches ')
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + (B * T) + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += (B * T)

        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends,'mps') and
                                                                            torch.backends.mps.is_available() else 'cpu')
    model = GPT(GPTConfig()).to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_loader = DataLoaderLite(B=2, T=1024)

    model.to(torch.device(device))

    for i in range(50):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device=device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        print(f'step : {i+1}, loss: {loss.item()}')
        t1 = time.time()
        dt = (t1 - t0)
        tokens_per_sec = (train_loader.B * train_loader.T) / (dt)
        print(
            f'step : {i + 1} | loss: {loss.item()} | dt: {dt * 1000:.2f} ms | tokens/sec: {tokens_per_sec:.2f}')