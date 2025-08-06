import torch
from mmengine.model import BaseModule
from mmengine.logging import MMLogger

import clip

from seg.models.utils.load_checkpoint import load_checkpoint_with_prefix

class CLIPTextEncoder(BaseModule):
    def __init__(
        self,
        model_name: str = 'RN50x16',
        fix: bool = True,
        init_cfg=None,
    ):
        assert init_cfg is not None and init_cfg['type'] == 'clip_pretrain', f"{init_cfg['type']} is not supported."
        pretrained = init_cfg['checkpoint']
        super().__init__(init_cfg=None)
        self.init_cfg = init_cfg
        self.logger = MMLogger.get_current_instance()

        self.model, _ = clip.load(model_name, device='cpu')
        self.tokenizer = clip.tokenize

        if pretrained == 'openai':
            # CLIP is already pretrained, no need to load weights
            pass
        else:
            # Load custom weights if provided
            state_dict = load_checkpoint_with_prefix(pretrained, prefix='text_encoder')
            self.model.load_state_dict(state_dict, strict=True)

        self.fix = fix
        if self.fix:
            self.train(mode=False)
            for param in self.model.parameters():
                param.requires_grad = False
                
    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        # 确保输入的 tokens 是在正确的设备上
        tokens = tokens.to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        return text_features

    def forward(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text).to(self.device)
        return self.encode_text(tokens)
    
    @property
    def device(self):
        return next(self.model.parameters()).device

    def init_weights(self):
        self.logger.info(f"Init Config for {self.__class__.__name__}")
        self.logger.info(self.init_cfg)

    def train(self: torch.nn.Module, mode: bool = True) -> torch.nn.Module:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if self.fix:
            super().train(mode=False)
        else:
            super().train(mode=mode)
        return self

    def forward(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        return text_features