import sd1_clip
import open_clip
import torch

class SD2ClipModel(torch.nn.Module, sd1_clip.ClipTokenWeightEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate",
        "hidden"
    ]
    #version="laion2b_s32b_b79k"
    def __init__(self, arch="ViT-H-14", device="cpu", max_length=77,
                 freeze=True, layer="penultimate", layer_idx=None):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'))
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        self.empty_tokens = [[49406] + [49407] + [0] * 75]
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        elif self.layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < 24
            self.clip_layer(layer_idx)
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def clip_layer(self, layer_idx):
        #layer_idx should have the same logic as the one for SD1
        if abs(layer_idx) >= 24:
            self.layer_idx = 0
        else:
            if layer_idx < 0:
                self.layer_idx = -(layer_idx + 1)
            else:
                self.layer_idx = 24 - (layer_idx + 1)

    def forward(self, tokens):
        tokens = torch.LongTensor(tokens).to(self.device)
        z = self.encode_with_transformer(tokens)
        return z

    def encode_with_transformer(self, tokens):
        x = self.model.token_embedding(tokens)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, tokens):
        return self(tokens)



class SD2Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, tokenizer_path=None):
        super().__init__(tokenizer_path, pad_with_end=False)


