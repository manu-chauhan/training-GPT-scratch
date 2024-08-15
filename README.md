# GPT Training from scratch

- Training a GPT model from scratch on Shakespeare data
- Multiple optimizations:
  - flash (sdpa) attention from Pytorch
  - torch.compile
  - nice number for vocab size 50257 to 50304
  - Optimizer AdamW with parameters used during GPT2 and GPT3 training
  - grad clipping 
  - cosine LR
- Ran training for 5000 steps on `input.txt` 
- model.pt [GDrive link](https://drive.google.com/file/d/11uCyn_PwFyP43t35ongpaLVfXDTKYTV9/view?usp=share_link)
