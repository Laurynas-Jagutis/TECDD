import numpy as np
import torch


class PrefixEmbedder:
    """Returns the decoder output z of the frozen transformer for a batch of inputs"""
    
    def __init__(self, model, device, batch_size=256):
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.cache = None
        self.hook_handle = self.model.decoder.register_forward_hook(self._hook)
        
    def _hook(self, module, inputs, output):
        val = output[0] if isinstance(output, tuple) else output
        if val.dim() == 3 and val.size(1) == 1:
            val = val.squeeze(1)
        self.cache = val.detach()      
        
    def remove_hook(self):
        self.hook_handle.remove()
        
    @torch.no_grad()
    def get_embeddings(self, X_acts, X_times):
        X_acts = X_acts.to(dtype=torch.long)
        X_times = X_times.to(dtype=torch.float32)
        
        all_embeddings = []
        for i in range(0, X_acts.size(0), self.batch_size):
            self.cache = None
            self.model(X_acts[i:i + self.batch_size].to(self.device), X_times[i:i + self.batch_size].to(self.device))
            all_embeddings.append(self.cache.cpu())
            
        if not all_embeddings:
            return np.array([])
        return torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
