"""Inspect model provider `steer/<hf model>`: Inspect's local HuggingFace provider plus an activation-steering hook.

Model args (-M):
  vectors=<vectors.pt>   from metagaming.steer.build_vectors
  direction=<name>       e.g. eval_generic:last, rendering:last, graded:last, random_0:last
  layer=<int>            decoder layer whose output gets coef * direction[layer] added
  coef=<float>           0 = no steering (the baseline run uses the same provider so batching and template match)
  rel=true|false         true: coef is a fraction of the mean residual-stream norm at that layer (meta residual_norm
                         from build_vectors; the direction is first normalised to unit length), the convention of the
                         Claude system cards and Read et al.; false (default): coef multiplies the raw diff-of-means vector
  scope=all|generation   token positions steered (default all)
  prefill=<text>         text placed at the start of the model's reasoning (after the chat template's <think> opener), for
                         the prefill experiment; it is put back in front of the reasoning in the log so judges see it
  plus any HuggingFace provider args, e.g. auto_model_class=AutoModelForImageTextToText dtype=bfloat16 enable_thinking=true

The `<think>...</think>` block of Qwen replies is split into ContentReasoning so the CoT scorers see it.
"""
import re
import sys
from pathlib import Path
from typing import Any

from inspect_ai.model import ChatMessageAssistant, ContentReasoning, ContentText, GenerateConfig, ModelAPI, modelapi
from inspect_ai.model._providers.hf import HuggingFaceAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metagaming.steer.hooks import Steer, load_vectors  # noqa: E402

THINK = re.compile(r'^\s*(?:<think>)?(.*?)</think>\s*(.*)$', re.S)


def split_think(text):
    m = THINK.match(text or '')
    if not m:
        return None, text
    return m.group(1).strip(), m.group(2)


@modelapi(name='steer')
def steer_provider():
    class SteerAPI(HuggingFaceAPI):
        def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None,
                     config: GenerateConfig = GenerateConfig(), **model_args: Any):
            vectors = model_args.pop('vectors', None)
            direction = model_args.pop('direction', 'eval_generic:last')
            layer = int(model_args.pop('layer', 0))
            coef = float(model_args.pop('coef', 0.0))
            scope = model_args.pop('scope', 'all')
            rel = str(model_args.pop('rel', 'false')).lower() in ('1', 'true', 'yes')
            self.prefill = str(model_args.pop('prefill', '') or '')
            super().__init__(model_name, base_url, api_key, config, **model_args)
            self.steer_args = {'vectors': vectors, 'direction': direction, 'layer': layer, 'coef': coef, 'rel': rel,
                               'scope': scope, 'prefill': self.prefill}
            self.steer = None
            if vectors and coef != 0:
                directions, meta = load_vectors(vectors)
                vec = directions[direction][layer]
                coef_abs = coef
                if rel:
                    vec = vec / vec.norm().clamp(min=1e-6)
                    coef_abs = coef * float(meta['residual_norm'][layer])
                self.steer_args['coef_abs'] = coef_abs      # what is actually added: coef_abs * (unit or raw) vector
                self.steer = Steer(self.model, vec, layer, coef_abs, scope).__enter__()

        def hf_chat(self, messages, tools):
            chat = super().hf_chat(messages, tools)
            return chat + self.prefill if self.prefill else chat

        async def generate(self, input, tools, tool_choice, config):
            out = await super().generate(input, tools, tool_choice, config)
            for choice in out.choices:
                msg = choice.message
                text = msg.text if isinstance(msg, ChatMessageAssistant) else None
                if text is None:
                    continue
                if self.prefill:
                    text = self.prefill + text
                reasoning, final = split_think(text)
                if reasoning is None and self.enable_thinking and '</think>' not in text:
                    reasoning, final = text.strip(), ''   # thought never closed: no final reply (counted as empty)
                if reasoning is not None:
                    msg.content = [ContentReasoning(reasoning=reasoning), ContentText(text=final)]
            out.metadata = dict(out.metadata or {}, steer=self.steer_args)
            return out

    return SteerAPI
