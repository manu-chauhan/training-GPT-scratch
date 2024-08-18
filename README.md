# Training GPT2 from scratch (and adding optimizations)

1. Baseline 

2. Speed-up-1 : 

   ```python
   weight sharing
   self.transformer.wte.weights = self.lm_head.weight
   ```

3. Speed-up-2: `torch.set_float32_matmul_precision('high')` this trades-off precision for speed (default is 'highest')

   Supports three settings:

   > - “highest”, float32 matrix multiplications use the float32 datatype (24 mantissa bits with 23 bits explicitly stored) for internal computations.
   > - “high”, float32 matrix multiplications either use the TensorFloat32 datatype (10 mantissa bits explicitly stored) or treat each float32 number as the sum of two bfloat16 numbers (approximately 16 mantissa bits with 14 bits explicitly stored), if the appropriate fast matrix multiplication algorithms are available. Otherwise float32 matrix multiplications are computed as if the precision is “highest”. See below for more information on the bfloat16 approach.
   > - “medium”, float32 matrix multiplications use the bfloat16 datatype (8 mantissa bits with 7 bits explicitly stored) for internal computations, if a fast matrix multiplication algorithm using that datatype internally is available. Otherwise float32 matrix multiplications are computed as if the precision is “high”.

4. Speed-up-3: 

   ```python
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)
   ```

   - Instances of [torch.autocast](https://pytorch.org/docs/stable/amp.html#autocasting) serve as context managers that allow regions of your script to run in mixed precision.
   - In these regions, CUDA ops run in a dtype chosen by autocast to improve performance while maintaining accuracy. See the [Autocast Op Reference](https://pytorch.org/docs/stable/amp.html#autocast-op-reference) for details on what precision autocast chooses for each op, and under what circumstances.

   

5. Speed-up-4: `# Solving for residual std scaling issue` 

   ```python
   x = torch.zeros(768)
   
   n = 100
   
   for i in range(n):
       x = x + torch.randn(768)
   
   print(x.std()) # tensor(9.8123)
   ```

   ```python
   x = torch.zeros(768)
   
   n = 100
   
   for i in range(n):
       x = x + n**-0.5 *  torch.randn(768)
   
   print(x.std()) # tensor(1.0066)
   ```

   

6. Speed-up-5: `torch.compile` 

7. Speed-up-6: `Pytorch Flash attention`

   1. ```python
      F.scaled_dot_product_attention(q, k, v, is_causal=True) 
      ```

      

8. Speed-up-7: `power of 2 (nice numbers), vocab size changed`

9. Speed-up-8: 

   ```python
   GPT-3 Paper
   model training, hyper-parameters
   Adam W
   gradient clipping
   ```

10. Speed-up-9: 

    ```python
    GPT-3 Paper
    add cosing delay for LR
    ```

11. Speed-up-10: `optimizer modified and fused version checked`


- Ran training for 5000 steps on `input.txt` 
- model.pt [GDrive link](https://drive.google.com/file/d/11uCyn_PwFyP43t35ongpaLVfXDTKYTV9/view?usp=share_link)
