"""
Dual Encoder: SigLIP2 + DINOv3
Extracts features from both models and concatenates them: C = C1 + C2
"""
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoModel, AutoImageProcessor


def _tensor_batch_to_pil_list(images):
    """Convert (B, C, H, W) tensor in [0,1] to list of PIL Images."""
    if images.dim() == 3:
        images = images.unsqueeze(0)
    pil_list = []
    for i in range(images.shape[0]):
        img = images[i].cpu()
        if img.max() <= 1.0:
            img = (img * 255).clamp(0, 255).byte()
        img = TF.to_pil_image(img)
        pil_list.append(img)
    return pil_list


class DualEncoder(torch.nn.Module): # dual 双
    """Combines SigLIP2 (vision+text) and DINOv3 (vision only)."""

    def __init__(self, siglip_model_name="google/siglip2-base-patch16-224",
                 dinov3_model_name="MFY111/dinov3-vits16"):
        super().__init__()
        self.use_dinov3 = dinov3_model_name is not None and str(dinov3_model_name).strip() != ""

        self.siglip_model = AutoModel.from_pretrained(siglip_model_name)
        self.siglip_image_processor = AutoImageProcessor.from_pretrained(siglip_model_name)
        from transformers import Siglip2Tokenizer
        self.siglip_tokenizer = Siglip2Tokenizer.from_pretrained(siglip_model_name)
        self.siglip_model.eval()

        self.c1 = self.siglip_model.config.vision_config.hidden_size
        if self.use_dinov3:
            self.dinov3_model = AutoModel.from_pretrained(dinov3_model_name)
            self.dinov3_processor = AutoImageProcessor.from_pretrained(dinov3_model_name)
            self.dinov3_model.eval()
            self.c2 = self.dinov3_model.config.hidden_size
        else:
            self.dinov3_model = None
            self.dinov3_processor = None
            self.c2 = 0
        self.c = self.c1 + self.c2

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def encode_image(self, images):
        """
        images: (B, 3, H, W) - expected from dataloader with resize/center_crop, [0,1] or [0,255]
        Returns: (B, C) with C = C1 + C2, L2 normalized per vector
        """
        with torch.no_grad():
            device = next(self.siglip_model.parameters()).device
            pil_list = _tensor_batch_to_pil_list(images)

            # SigLIP2 - use image processor only
            siglip_inputs = self.siglip_image_processor(images=pil_list, return_tensors="pt")
            siglip_pixels = siglip_inputs["pixel_values"].to(device)
            siglip_out = self.siglip_model.get_image_features(pixel_values=siglip_pixels)
            siglip_feat = siglip_out.pooler_output if hasattr(siglip_out, "pooler_output") and siglip_out.pooler_output is not None else siglip_out.last_hidden_state[:, 0, :]
            siglip_feat = siglip_feat / siglip_feat.norm(dim=-1, keepdim=True)

            if self.use_dinov3:
                dinov3_inputs = self.dinov3_processor(images=pil_list, return_tensors="pt")
                dinov3_inputs = {k: v.to(device) for k, v in dinov3_inputs.items()}
                dinov3_out = self.dinov3_model(**dinov3_inputs).pooler_output
                dinov3_feat = dinov3_out / dinov3_out.norm(dim=-1, keepdim=True)
                features = torch.cat([siglip_feat, dinov3_feat], dim=-1)
                features = features / features.norm(dim=-1, keepdim=True)
            else:
                features = siglip_feat
        return features

    def get_text_weights(self, classnames, template, device):
        """
        Get zero-shot classifier weights from SigLIP2 text encoder.
        Returns (C, N): pad SigLIP text emb (C1,N) with zeros for DINO dims -> (C1+C2, N)
        """
        with torch.no_grad():
            weights = []
            for classname in classnames:
                classname = classname.replace("_", " ").lower()
                texts = [t.format(classname) for t in template]
                inputs = self.siglip_tokenizer(
                    texts,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = self.siglip_model.get_text_features(**inputs)
                # get_text_features returns BaseModelOutputWithPooling, extract pooler_output
                text_feat = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, -1, :]
                text_feat = text_feat.mean(dim=0) / text_feat.mean(dim=0).norm()
                weights.append(text_feat)
            siglip_weights = torch.stack(weights, dim=1)
            if self.use_dinov3:
                zeros = torch.zeros(self.c2, siglip_weights.shape[1], device=device, dtype=siglip_weights.dtype)
                return torch.cat([siglip_weights, zeros], dim=0)
            return siglip_weights
