import torch
import torch.nn.functional as F
import numpy as np
import math

from nodes import common_ksampler

# Import required modules
import comfy.sample
import comfy.samplers
import comfy.utils
# Prepare callback for progress
import latent_preview



NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def create_gaussian_filter(height, width, sigma, device):
    """Create Gaussian filter in frequency domain"""
    # Create frequency grids
    u = torch.linspace(-0.5, 0.5, height, device=device)
    v = torch.linspace(-0.5, 0.5, width, device=device)
    U, V = torch.meshgrid(u, v, indexing='ij')
    
    # Calculate squared distance from center
    D2 = U**2 + V**2
    
    # Gaussian filter formula: exp(-2 * pi^2 * sigma^2 * D2)
    pi = math.pi
    gaussian = torch.exp(-2 * (pi**2) * (sigma**2) * D2)
    
    return gaussian


def HighFrequencyEnhancer(latent, high_freq_mult, sigma, denoise_threshold, mask_hardness, hf_pre_blur_sigma):
    # 1. Extract the latent samples tensor
    samples = latent["samples"].clone()
    batch_size, channels, height, width = samples.shape[:4]
    
    # Handle WAN format if needed
    is_wan = False
    if samples.ndim == 5:
        samples = samples.squeeze(2)
        is_wan = True
        batch_size, channels, height, width = samples.shape

    # --- Frequency Separation using FFT ---
    device = samples.device
    
    # Create Gaussian filter in frequency domain
    gaussian_filter = create_gaussian_filter(height, width, sigma, device)
    # Expand filter dimensions to match latent tensor [1, 1, H, W]
    gaussian_filter = gaussian_filter.view(1, 1, height, width)
    
    # Apply FFT to latent samples
    fft_latent = torch.fft.fft2(samples, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft_latent, dim=(-2, -1))
    
    # Apply low-pass filter
    low_freq_fft = fft_shifted * gaussian_filter
    
    # Inverse FFT to get low frequency component
    low_freq_shifted = torch.fft.ifftshift(low_freq_fft, dim=(-2, -1))
    low_freq = torch.fft.ifft2(low_freq_shifted, dim=(-2, -1)).real
    
    # High frequency is residual
    high_freq_original = samples - low_freq

    # --- Smart Mask Generation ---
    if denoise_threshold > 0:
        # Pre-blur for noise grouping if enabled
        if hf_pre_blur_sigma > 0.0:
            # Calculate kernel size based on sigma
            k_size = int(2 * round(3 * hf_pre_blur_sigma) + 1)
            if k_size % 2 == 0:
                k_size += 1
            
            # Create 1D Gaussian kernel
            kernel_1d = torch.arange(k_size, device=device) - k_size // 2
            kernel_1d = torch.exp(-0.5 * (kernel_1d / hf_pre_blur_sigma) ** 2)
            kernel_1d = kernel_1d / kernel_1d.sum()
            
            # Create 2D kernel
            kernel_2d = kernel_1d.view(1, 1, -1, 1) * kernel_1d.view(1, 1, 1, -1)
            
            # Apply to each channel separately
            high_freq_detection = torch.zeros_like(high_freq_original)
            for c in range(channels):
                channel = high_freq_original[:, c:c+1]
                # Manual convolution with reflection padding
                padded = F.pad(channel, (k_size//2, k_size//2, k_size//2, k_size//2), mode='reflect')
                high_freq_detection[:, c:c+1] = F.conv2d(padded, kernel_2d, padding=0)
        else:
            high_freq_detection = high_freq_original

        # Calculate magnitude and create soft gate mask
        magnitude = torch.abs(high_freq_detection)
        mask = torch.sigmoid((magnitude - denoise_threshold) * mask_hardness)
        
        # Apply mask and detail enhancement
        high_freq_final = high_freq_original * mask * high_freq_mult
        
    else:
        # Passthrough if no denoising requested
        mask = torch.ones_like(samples)
        high_freq_final = high_freq_original * high_freq_mult

    # --- Recombination ---
    enhanced_samples = low_freq + high_freq_final
    
    # --- Output Preparation ---
    # Prepare Latent Output
    enhanced_latent = latent.copy()
    if is_wan:
        enhanced_samples = enhanced_samples.unsqueeze(2)
    enhanced_latent["samples"] = enhanced_samples
    
    # Prepare Mask Preview Output
    # Calculate mean across channels to get grayscale intensity
    mask_preview = torch.mean(mask, dim=1, keepdim=True)  # [Batch, 1, H, W]
    mask_preview = mask_preview.repeat(1, 3, 1, 1)        # [Batch, 3, H, W]
    mask_preview = mask_preview.permute(0, 2, 3, 1)       # [Batch, H, W, 3]

    return (enhanced_latent, mask_preview)

class LatentFrequencyEnhancer_lrzjason:
    """
    ComfyUI Node for selective latent denoising and enhancement using FFT.
    Uses frequency separation via FFT to isolate details and a Sigmoid Soft-Gate mask 
    to smoothly eliminate background noise while preserving sharp features.
    Outputs the processing mask as a preview image.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "high_freq_mult": ("FLOAT", {
                    "default": 1.15, 
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.01,
                    "label": "Detail Strength (HF Mult)"
                }),
                "sigma": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1,
                    "label": "Frequency Split Sigma"
                }),
                "denoise_threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "label": "Noise Threshold"
                }),
                "mask_hardness": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 1.0,
                    "label": "Mask Hardness (Transition)"
                }),
                "hf_pre_blur_sigma": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Noise Grouping (Pre-Blur)"
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("enhanced_latent", "mask_preview")
    FUNCTION = "enhance"
    CATEGORY = "latent/enhancement"
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def enhance(self, latent, high_freq_mult, sigma, denoise_threshold, mask_hardness, hf_pre_blur_sigma):
        return HighFrequencyEnhancer(latent, high_freq_mult, sigma, denoise_threshold, mask_hardness, hf_pre_blur_sigma)
        
# Register the node
NODE_CLASS_MAPPINGS["LatentFrequencyEnhancer_lrzjason"] = LatentFrequencyEnhancer_lrzjason
NODE_DISPLAY_NAME_MAPPINGS["LatentFrequencyEnhancer_lrzjason"] = "Latent Frequency Enhancer (lrzjason)"


class HFEPostProcessor:
    """
    Custom sampler with high-frequency enhancement during sampling process.
    Runs sampling steps in a loop and applies HighFrequencyEnhancer at specified steps.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        # "add_noise": (["enable", "disable"], ),
                        "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                        "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                        "latent_image": ("LATENT", ),
                        "hfe_steps": ("INT", {
                            "default": 2,
                            "min": 1,
                            "max": 100,
                            "step": 1,
                            "label": "Start HFE Step"
                        }),
                        "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),
                        "high_freq_mult": ("FLOAT", {
                            "default": 1.05, 
                            "min": 1.0,
                            "max": 2.0,
                            "step": 0.01,
                            "label": "Detail Strength (HF Mult)"
                        }),
                        "sigma": ("FLOAT", {
                            "default": 5.0,
                            "min": 0.01,
                            "max": 20.0,
                            "step": 0.01,
                            "label": "Frequency Split Sigma"
                        }),
                        "denoise_threshold": ("FLOAT", {
                            "default": 0.05,
                            "min": 0.0,
                            "max": 1.0,
                            "step": 0.001,
                            "label": "Noise Threshold"
                        }),
                        "mask_hardness": ("FLOAT", {
                            "default": 2.0,
                            "min": 0.01,
                            "max": 100.0,
                            "step": 0.01,
                            "label": "Mask Hardness (Transition)"
                        }),
                        "hf_pre_blur_sigma": ("FLOAT", {
                            "default": 0.5, 
                            "min": 0.0,
                            "max": 1.0,
                            "step": 0.01,
                            "label": "Noise Grouping (Pre-Blur)"
                        }),
                    }
                }

    # RETURN_TYPES = ("LATENT","LATENT","LATENT","LATENT","LATENT","LATENT","LATENT","LATENT",)
    RETURN_TYPES = ("LATENT", )
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, 
            #    add_noise, 
               noise_seed, 
               steps, 
               latent_image, 
               hfe_steps,
               cfg, sampler_name, scheduler, positive, negative, 
            #    start_at_step, end_at_step, 
            #    return_with_leftover_noise, 
               denoise=1.0,
               high_freq_mult=1.05, sigma=2, denoise_threshold=0.05, mask_hardness=2, hf_pre_blur_sigma=0.5
            ):
        disable_noise = False
        latent = latent_image
        
        force_full_denoise = True
        
        start_hfe_step = steps
        steps = start_hfe_step + hfe_steps
        for i in range(start_hfe_step, steps):
            start_at_step = i
            end_at_step = i + 1
            if end_at_step >= steps:
                end_at_step = 10000

            latent = HighFrequencyEnhancer(latent, high_freq_mult, sigma, denoise_threshold, mask_hardness, hf_pre_blur_sigma)[0]
                                    
            latent = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)[0]    
            
        return (latent, )


NODE_CLASS_MAPPINGS["HFEPostProcessor (lrzjason)"] = HFEPostProcessor
NODE_DISPLAY_NAME_MAPPINGS["HFEPostProcessor (lrzjason)"] = "HFEPostProcessor (lrzjason)"