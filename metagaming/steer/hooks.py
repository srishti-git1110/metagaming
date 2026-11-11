"""Model loading, residual-stream reading and steering hooks (plain transformers; works for dense Qwen3.5/3.6
checkpoints, which are vision-language `*ForConditionalGeneration` classes whose text layers live under
model.model.language_model.layers, and for ordinary causal LMs whose layers live under model.model.layers)."""
import json
from pathlib import Path

import torch
from torch import nn


def load_model(name, dtype='bfloat16', device=None):
    import transformers
    device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    dt = getattr(torch, dtype)
    tok = transformers.AutoTokenizer.from_pretrained(name)
    cfg = transformers.AutoConfig.from_pretrained(name)
    loader = transformers.AutoModelForImageTextToText if getattr(cfg, 'text_config', None) is not None else transformers.AutoModelForCausalLM
    model = loader.from_pretrained(name, dtype=dt, device_map=device)
    model.eval()
    return model, tok


def find_layers(model) -> nn.ModuleList:
    """The decoder-layer ModuleList (residual stream is the output of each entry)."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.endswith('layers') and len(mod) > 4 and hasattr(mod[0], 'forward'):
            return mod
    raise RuntimeError('no decoder layer list found')


def render_chat(tok, messages, enable_thinking=None):
    kw = {} if enable_thinking is None else {'enable_thinking': enable_thinking}
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)


@torch.no_grad()
def residuals(model, tok, texts, batch_size=4, max_len=4096):
    """Per text: dict with 'last' [L, d] (last prompt token) and 'mean' [L, d] (mean over non-pad prompt tokens),
    where L = number of decoder layers (hidden_states[1:]; index i = output of layer i). Returned on CPU, float32."""
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors='pt', padding=True, truncation=True, max_length=max_len,
                  add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[1:]
        h = torch.stack(hs, 1).float()                      # [B, L, T, d]
        mask = enc['attention_mask'][:, None, :, None].float()
        mean = (h * mask).sum(2) / mask.sum(2).clamp(min=1)
        last = h[:, :, -1, :]                              # left padding: last position is the last real token
        for b in range(h.shape[0]):
            out.append({'last': last[b].cpu(), 'mean': mean[b].cpu()})
        del hs, h
    return out


class Steer:
    """Adds coef * vec[layer] to the residual stream at the output of decoder layer `layer`.
    scope: 'all' = every token position (prompt and generation); 'generation' = only decode steps (sequence length 1,
    i.e. after the prompt has been prefilled)."""

    def __init__(self, model, vec: torch.Tensor, layer: int, coef: float, scope='all'):
        self.layers = find_layers(model)
        self.vec = vec.to(device=model.device, dtype=next(model.parameters()).dtype)
        self.layer, self.coef, self.scope = int(layer), float(coef), scope
        self.handle = None

    def _hook(self, module, args, output):
        if self.coef == 0:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if self.scope == 'generation' and h.shape[1] != 1:
            return output
        h = h + self.coef * self.vec
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def __enter__(self):
        self.handle = self.layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def load_vectors(path):
    """vectors.pt from build_vectors.py: {'directions': {name: tensor [L, d]}, 'meta': {...}}."""
    d = torch.load(path, map_location='cpu')
    return d['directions'], d['meta']


def save_vectors(path, directions, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'directions': {k: v.cpu() for k, v in directions.items()}, 'meta': meta}, path)
    Path(path).with_suffix('.json').write_text(json.dumps(meta, indent=1))
