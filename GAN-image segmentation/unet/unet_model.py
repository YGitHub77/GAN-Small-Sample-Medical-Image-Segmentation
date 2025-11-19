""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp



class UNetWithDeepSupervision(nn.Module):
    """
    基于 smp.Unet(resnet34, imagenet)
    - forward(x)        -> main_logits（用于验证/推理/对抗）
    - forward_with_aux  -> main_logits, aux_logits_list（用于训练深监督）
    通过对 decoder.blocks 注册 forward hook，收集多尺度特征做 aux 头。
    """
    def __init__(self, in_channels=3, out_channels=1, n_aux=3):
        super().__init__()
        self.net = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=out_channels,
            activation=None,  # logits
        )
        self.n_aux = n_aux
        self._dec_feats = []
        self._hooks = []
        self.aux_heads = nn.ModuleList()
        self._aux_ready = False

        self._register_decoder_hooks()

    def _register_decoder_hooks(self):
        blocks = getattr(self.net.decoder, 'blocks', None)
        if blocks is None or len(blocks) == 0:
            # 兜底：没有 blocks 就只拿最后输出（此时 n_aux 形同关闭）
            def _hook(module, inp, out):
                self._dec_feats = [out]
            self._hooks.append(self.net.decoder.register_forward_hook(_hook))
            return

        # 依次 hook 每层 block 的输出（从粗到细 append）
        for blk in blocks:
            def _hook(module, inp, out):
                self._dec_feats.append(out)
            self._hooks.append(blk.register_forward_hook(_hook))

    def _clear_dec_feats(self):
        self._dec_feats = []

    def _build_aux_if_needed(self, main_logits):
        if self._aux_ready:
            return
        if self.n_aux <= 0 or len(self._dec_feats) == 0:
            self._aux_ready = True
            return

        # 按空间分辨率从小到大排，取最后 n_aux 个（分辨率最高）
        feats_sorted = sorted(self._dec_feats, key=lambda t: (t.shape[2], t.shape[3]))
        used = feats_sorted[-self.n_aux:] if self.n_aux > 0 else []

        in_chs = [t.shape[1] for t in used]
        self.aux_heads = nn.ModuleList([nn.Conv2d(c, 1, kernel_size=1, bias=True) for c in in_chs]).to(main_logits.device)
        self._aux_ready = True

        used_str = ', '.join(f"{c}ch@{t.shape[2]}x{t.shape[3]}" for c, t in zip(in_chs, used))
        print(f"[DeepSup] Aux heads built on {len(used)} dec features: {used_str}")

    def forward_with_aux(self, x):
        self._clear_dec_feats()
        feats = self.net.encoder(x)
        dec_out = self.net.decoder(*feats)           # hooks 会填充 _dec_feats
        main_logits = self.net.segmentation_head(dec_out)

        self._build_aux_if_needed(main_logits)

        aux_logits_list = []
        if self.n_aux > 0 and len(self.aux_heads) > 0:
            feats_sorted = sorted(self._dec_feats, key=lambda t: (t.shape[2], t.shape[3]))
            used = feats_sorted[-self.n_aux:]
            for head, f in zip(self.aux_heads, used):
                aux = head(f)
                aux_up = F.interpolate(aux, size=main_logits.shape[2:], mode='bilinear', align_corners=False)
                aux_logits_list.append(aux_up)
        return main_logits, aux_logits_list

    def forward(self, x):
        feats = self.net.encoder(x)
        dec_out = self.net.decoder(feats)
        main_logits = self.net.segmentation_head(dec_out)
        return main_logits

    @property
    def encoder(self):
        return self.net.encoder

    def remove_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []



