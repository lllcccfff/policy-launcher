from __future__ import annotations

import base64
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Sequence

from PIL import Image


OpenEMMABackendName = Literal["gpt", "qwen", "llava", "llama"]

OFFICIAL_MODEL_IDS: dict[OpenEMMABackendName, str] = {
    "gpt": "gpt-4o-2024-11-20",
    "qwen": "Qwen/Qwen2-VL-7B-Instruct",
    "llava": "liuhaotian/llava-v1.6-mistral-7b",
    "llama": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}

class OpenEMMAOfficialVLM:
    """OpenEMMA's official backend-specific model loading and generation paths."""

    def __init__(self, backend: OpenEMMABackendName, model_id: str | None = None):
        self.backend = backend
        self.model_id = OFFICIAL_MODEL_IDS[backend] if model_id is None else model_id
        self._model: Any | None = None
        self._processor: Any | None = None
        self._tokenizer: Any | None = None
        self._client: Any | None = None

        if backend == "gpt":
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif backend == "qwen":
            self._load_qwen()
        elif backend == "llava":
            self._load_llava()
        elif backend == "llama":
            self._load_llama()
        else:
            raise ValueError(f"Unsupported OpenEMMA backend: {backend}")

    @staticmethod
    def _decode_image(image_base64: str) -> Image.Image:
        return Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")

    def _load_qwen(self) -> None:
        import torch
        from transformers import (
            AutoConfig,
            AutoProcessor,
            Qwen2VLForConditionalGeneration,
        )

        config = AutoConfig.from_pretrained(self.model_id)
        model_type = str(config.model_type)
        if model_type != "qwen2_vl":
            raise ValueError(f"Qwen backend requires a Qwen2-VL model, got model_type={model_type!r}")
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def _load_llama(self) -> None:
        import torch
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        self._model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def _load_llava(self) -> None:
        source_root = Path(__file__).resolve().parents[2] / "OpenEMMA"
        if not source_root.is_dir():
            raise FileNotFoundError(f"Official OpenEMMA LLaVA source is missing: {source_root}")
        source_path = str(source_root)
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        self._tokenizer, self._model, self._processor, _ = load_pretrained_model(
            self.model_id,
            None,
            "llava-v1.6-mistral-7b",
        )

    def infer(self, prompt: str, image_base64s: Sequence[str], sys_message: str | None = None) -> str:
        if self.backend == "gpt":
            return self._infer_gpt(prompt, image_base64s, sys_message)
        if self.backend == "qwen":
            return self._infer_qwen(prompt, image_base64s)
        if self.backend == "llava":
            return self._infer_llava(prompt, image_base64s)
        if self.backend == "llama":
            return self._infer_llama(prompt, image_base64s)
        raise ValueError(f"Unsupported OpenEMMA backend: {self.backend}")

    def _infer_gpt(self, prompt: str, image_base64s: Sequence[str], sys_message: str | None) -> str:
        if not image_base64s:
            raise ValueError("OpenEMMA GPT inference requires at least one camera image")
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            for image in image_base64s
        ]
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        if sys_message is not None:
            messages.append({"role": "system", "content": sys_message})
        result = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=400,
        )
        return result.choices[0].message.content

    def _infer_qwen(self, prompt: str, image_base64s: Sequence[str]) -> str:
        from qwen_vl_utils import process_vision_info

        image = self._decode_image(image_base64s[-1])
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        generated_ids = self._model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _infer_llama(self, prompt: str, image_base64s: Sequence[str]) -> str:
        image = self._decode_image(image_base64s[-1])
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        input_text = self._processor.apply_chat_template(message, add_generation_prompt=True)
        inputs = self._processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self._model.device)
        output = self._model.generate(**inputs, max_new_tokens=2048)
        output_text = self._processor.decode(output[0])
        return re.findall(
            r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>",
            output_text,
            re.DOTALL,
        )[0].strip()

    def _infer_llava(self, prompt: str, image_base64s: Sequence[str]) -> str:
        import torch

        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_PLACEHOLDER, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in prompt:
            if self._model.config.mm_use_im_start_end:
                prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt)
            else:
                prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
        elif self._model.config.mm_use_im_start_end:
            prompt = image_token_se + "\n" + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv = conv_templates["mistral_instruct"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(
            conv.get_prompt(),
            self._tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).cuda()
        image = self._decode_image(image_base64s[-1])
        image_tensor = process_images([image], self._processor, self._model.config)[0]
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                max_new_tokens=2048,
                use_cache=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
