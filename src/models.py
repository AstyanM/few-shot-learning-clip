"""
Model architectures for CoOp and CoCoOp prompt learning.
"""

from collections import OrderedDict

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """CLIP text encoder wrapper that operates on learnable prompt embeddings.

    Instead of tokenizing raw text, this encoder takes pre-built prompt
    embeddings (with learnable context vectors) and produces normalised
    text features.

    Args:
        clip_model: A loaded CLIP model instance.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """Encode prompt embeddings into text features.

        Args:
            prompts: Tensor of shape ``[n_cls, n_tokens, ctx_dim]``.
            tokenized_prompts: Original tokenized prompts (used to locate EOS token).

        Returns:
            Text feature tensor of shape ``[n_cls, proj_dim]``.
        """
        device = prompts.device
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Extract features from the end-of-text token
        x = x[torch.arange(x.shape[0], device=device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    """Learnable soft-prompt generator for CoOp.

    Replaces hand-crafted text prompts with learnable context vectors that
    are optimised via back-propagation on few-shot data.

    Args:
        classnames: List of class name strings.
        clip_model: A loaded CLIP model instance.
        n_ctx: Number of context tokens.
        ctx_init: Optional initialisation text (e.g. ``"a photo of a"``).
        noise_scale: If set, adds Gaussian noise during training as regularisation.
    """

    def __init__(self, classnames, clip_model, n_ctx=4, ctx_init="", noise_scale=None):
        super().__init__()
        self.noise_scale = noise_scale
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        device = clip_model.ln_final.weight.device

        # Initialize context vectors
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words: {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p).to(device) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])   # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        if self.training and self.noise_scale:
            max_val = ctx.abs().max().item()
            std = self.noise_scale * max_val
            noise = torch.randn_like(ctx) * std
            ctx = ctx + noise

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts


class CoOp(nn.Module):
    """Context Optimization (CoOp) model.

    Combines a frozen CLIP image encoder with a learnable prompt generator
    and a text encoder to perform few-shot classification.

    Args:
        classnames: List of class name strings.
        clip_model: A loaded CLIP model instance.
        n_ctx: Number of context tokens.
        ctx_init: Optional initialisation text.
        noise_scale: Gaussian noise scale for regularisation.
    """

    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init="", noise_scale=0.15):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, n_ctx, ctx_init, noise_scale)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


class CoCoOpPromptLearner(nn.Module):
    """Conditional Context Optimization prompt learner.

    Extends :class:`PromptLearner` with a lightweight Meta-Network that
    generates image-conditional biases for the context vectors, improving
    generalisation to novel classes.

    Args:
        classnames: List of class name strings.
        clip_model: A loaded CLIP model instance.
        n_ctx: Number of context tokens.
        ctx_init: Optional initialisation text.
        noise_scale: If set, adds Gaussian noise and uses a deeper Meta-Net
            with BatchNorm and Dropout.
    """

    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init="", noise_scale=None):
        super().__init__()
        self.noise_scale = noise_scale
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        device = clip_model.ln_final.weight.device

        # Initialize context vectors (same as CoOp)
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words: {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        # Meta-network architecture depends on noise_scale
        if noise_scale:
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                ("bn1", nn.BatchNorm1d(vis_dim // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("dropout", nn.Dropout(p=0.3)),
                ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
            ]))
        else:
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
            ]))

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p).to(device) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def construct_prompts(self, ctx, prefix, suffix):
        """Concatenate prefix, context, and suffix into full prompt embeddings."""
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self, im_features):
        """Generate image-conditional prompts.

        Args:
            im_features: Image features ``[batch_size, vis_dim]``.

        Returns:
            Prompt tensor ``[batch_size, n_cls, n_tokens, ctx_dim]``.
        """
        prefix = self.token_prefix
        suffix = self.token_suffix

        ctx = self.ctx
        bias = self.meta_net(im_features)
        bias = bias.unsqueeze(1)
        ctx = ctx.unsqueeze(0)

        if self.training and self.noise_scale:
            max_val = ctx.abs().max().item()
            std = self.noise_scale * max_val
            noise = torch.randn_like(ctx) * std
            ctx = ctx + noise

        ctx_shifted = ctx + bias

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        return prompts


class CoCoOp(nn.Module):
    """Conditional Context Optimization (CoCoOp) model.

    Combines CLIP's visual encoder with an image-conditional prompt learner
    and text encoder for improved few-shot generalisation.

    Args:
        classnames: List of class name strings.
        clip_model: A loaded CLIP model instance.
        n_ctx: Number of context tokens.
        ctx_init: Optional initialisation text.
        noise_scale: Gaussian noise scale for regularisation.
    """

    def __init__(self, classnames, clip_model, n_ctx=4, ctx_init="", noise_scale=None):
        super().__init__()
        self.prompt_learner = CoCoOpPromptLearner(classnames, clip_model, n_ctx, ctx_init, noise_scale)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, self.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)

        logits = torch.stack(logits)
        return logits
