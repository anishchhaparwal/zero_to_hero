we wnat to implement logging to ternsorboard and adding history grams once in a while:

# Simple TensorBoard Histogram Monitoring Implementation

I'll implement a lightweight histogram monitoring solution that stays close to your codebase's style and avoids complex hooks. This will have minimal impact on training speed while providing valuable insights.

## Implementation

First, add these imports at the top of your file:

```python
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
```

Then, initialize a TensorBoard writer after creating your log directory:

```python
# After your log directory setup
if master_process:
    os.makedirs(os.path.join(log_dir, "histograms"), exist_ok=True)
    tb_writer = SummaryWriter(log_dir)
```

Next, add this simple function for histogram visualization:

```python
def log_histograms(model, step, log_dir, tb_writer=None):
    """
    Log histograms of model parameters and gradients.
    Simple approach without hooks to visualize weight and gradient distributions.
    """
    # Skip if not running on master process
    if not master_process:
        return
        
    # Group parameters by type for plotting
    weights_dict = {}
    grads_dict = {}
    
    # Collect parameters and gradients from key layers
    sample_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim == 2:  # Only matrix params
            # Select representative layers to avoid too many histograms
            if ('h.0.' in name or 'h.5.' in name or 'h.11.' in name or 
                'wte' in name or 'lm_head' in name):
                
                # Add param to weights dict
                weights_dict[name] = param.detach().cpu().float()
                
                # Add gradient to grads dict if available
                if param.grad is not None:
                    grads_dict[name] = param.grad.detach().cpu().float()
                    
                    # Print statistics like in the example
                    print(f'weight {name}: mean {param.mean().item():+.4f}, std {param.std().item():.4e}, '
                          f'grad:data ratio {param.grad.std().item() / param.std().item():.4e}')
                
                sample_count += 1
                if sample_count >= 8:  # Limit number of histograms for clarity
                    break
    
    # Plot weight distributions (multiple on same graph)
    if weights_dict:
        plt.figure(figsize=(20, 8))
        for name, param in weights_dict.items():
            hy, hx = torch.histogram(param.flatten(), bins=50, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
        plt.legend(list(weights_dict.keys()), fontsize='small')
        plt.title(f'Weight Distribution (Step {step})')
        plt.xlabel('Weight Value')
        plt.ylabel('Density')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(log_dir, f"histograms/weights_{step}.png"))
        plt.close()
    
    # Plot gradient distributions (multiple on same graph)
    if grads_dict:
        plt.figure(figsize=(20, 8))
        for name, grad in grads_dict.items():
            hy, hx = torch.histogram(grad.flatten(), bins=50, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
        plt.legend(list(grads_dict.keys()), fontsize='small')
        plt.title(f'Gradient Distribution (Step {step})')
        plt.xlabel('Gradient Value')
        plt.ylabel('Density')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(log_dir, f"histograms/gradients_{step}.png"))
        plt.close()
    
    # Log to TensorBoard if provided
    if tb_writer:
        for name, param in weights_dict.items():
            tb_writer.add_histogram(f"weights/{name}", param, step)
        
        for name, grad in grads_dict.items():
            tb_writer.add_histogram(f"gradients/{name}", grad, step)
```

Now integrate this into your validation section in the training loop:

```python
if step % 25 == 0 or last_step:
    # ... existing validation code ...
    
    if master_process:
        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        
        # Log histograms less frequently to avoid overhead
        if step % 100 == 0 or last_step:
            log_histograms(raw_model, step, log_dir, tb_writer)
            
        # If we have performance metrics, report a summary
        # ... rest of your existing code ...
```

Finally, close the writer at the end of your script:

```python
if ddp:
    destroy_process_group()

# Close TensorBoard writer
if master_process and 'tb_writer' in locals():
    tb_writer.close()
```

## Benefits of this Implementation

1. **Simplicity** - No hooks or complex instrumenting, just directly samples model parameters
2. **Efficiency** - Minimal overhead by:
   - Only running on master process
   - Only logging periodically (every 100 steps)  
   - Only sampling a few representative layers
3. **Visualization** - Multiple distributions plotted on the same graph for easy comparison
4. **Dual Output** - Both saves image files and logs to TensorBoard
5. **Minimal Code Changes** - Stays close to your existing code style

To view the TensorBoard logs after training:
```bash
tensorboard --logdir=log
```

This implementation provides the key insights from Karpathy's examples (weight/gradient distributions and statistics) without adding significant training overhead.

add instruction to readme.md:

Viewing TensorBoard Logs
After implementing these changes, you can view the TensorBoard logs by running:
tensorboard --logdir=log
