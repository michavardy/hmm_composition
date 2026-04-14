# initializers/base.py
import torch



class BaseInitializer:
    """All initializers must implement `__call__`."""
    def __call__(self, shape, device="cpu"):
        raise NotImplementedError

class StickyInitializer(BaseInitializer):
    """
    Creates a matrix (e.g., transition logits) with a bias toward the diagonal.
    For square matrices (num_states x num_states), the diagonal elements are higher.
    """
    def __call__(self, shape, device="cpu"):
        if len(shape) == 1:
            # If it's a vector (like initial logits), just zeros
            return torch.zeros(shape, device=device)
        elif len(shape) == 2 and shape[0] == shape[1]:
            # square matrix: add a bias to diagonal
            mat = torch.zeros(shape, device=device)
            mat += torch.eye(shape[0], device=device)  # 1 on diagonal
            return mat
        else:
            # For non-square matrices, fallback to zeros
            return torch.zeros(shape, device=device)

class StickyNoisyInitializer(BaseInitializer):
    """
    Sticky diagonal with optional noise. 
    noise_coeff controls magnitude of added noise.
    """
    def __init__(self, noise_coeff=0.01):
        self.noise_coeff = noise_coeff

    def __call__(self, shape, device="cpu"):
        mat = StickyInitializer()(shape, device=device)
        noise = self.noise_coeff * torch.randn(shape, device=device)
        return mat + noise
    
class ZerosInitializer(BaseInitializer):
    def __call__(self, shape, device="cpu"):
        return torch.zeros(shape, device=device)
    
class RandomInitializer(BaseInitializer):
    def __call__(self, shape, device="cpu"):
        return torch.randn(shape, device=device)

class SmallRandomInitializer(BaseInitializer):
    def __call__(self, shape, device="cpu"):
        return 0.01 * torch.randn(shape, device=device)
    
InitializerType = ZerosInitializer | RandomInitializer | SmallRandomInitializer | StickyInitializer | StickyNoisyInitializer

initializer_mapping = {
    "zeros": ZerosInitializer(),
    "random": RandomInitializer(),
    "small_random": SmallRandomInitializer(),
    "sticky": StickyInitializer(),
    "sticky_noisy": StickyNoisyInitializer(),
}