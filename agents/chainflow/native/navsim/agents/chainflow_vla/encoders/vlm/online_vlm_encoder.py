# ------------------------------------------------------------------------
# Copyright (c) 2026 AFARI-Model. All Rights Reserved.
# ------------------------------------------------------------------------
"""Online InternVL 2B VLM feature extractor.

Provides `vlm_hidden_state` ([L, D], default [2800, 1536]) on the fly during
inference/evaluation as a drop-in replacement for the offline `.gz` cache.

Design notes:
- This class is intentionally a plain ``object`` (NOT ``nn.Module``). The agent
  stores it as a plain attribute so the InternVL weights never enter the agent
  ``state_dict`` (keeps strict checkpoint loading intact).
- Image preprocessing (dynamic tiling) is a pure-CPU function that runs inside
  the DataLoader workers; the heavy GPU forward runs in the agent's main
  process (one InternVL copy per rank).
- Prompt construction and forward logic mirror the offline cache builder
  (``InternVLHiddenStateFeatureBuilder``) so online features match cached ones.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from navsim.agents.chainflow_vla.encoders.vlm.utils.conversation import (
    get_conv_template,
)
from navsim.agents.chainflow_vla.encoders.vlm.utils.internvl_preprocess import (
    build_transform,
    dynamic_preprocess,
)

logger = logging.getLogger(__name__)

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"

SYSTEM_MESSAGE = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""


def _format_number(n: float, decimal_places: int = 2) -> str:
    """String format matching the offline HF hidden-state caching prompts."""
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


def build_vlm_pixel_values(
    image: "np.ndarray | Image.Image | str | os.PathLike",
    force_image_size: int = 448,
    max_dynamic_patch: int = 12,
    dynamic_image_size: bool = True,
    use_thumbnail: bool = True,
) -> torch.Tensor:
    """Pure-CPU InternVL preprocessing (safe to call inside DataLoader workers).

    :param image: front-camera image as a numpy array (H, W, 3), PIL image, or
        a path to an image file (str / PathLike).
    :return: pixel_values tensor of shape [num_patches, 3, force_image_size, force_image_size].
    """
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image).convert("RGB")
    elif isinstance(image, (str, os.PathLike)):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    transform = build_transform(input_size=force_image_size)
    if dynamic_image_size:
        tiles = dynamic_preprocess(
            pil_image,
            image_size=force_image_size,
            use_thumbnail=use_thumbnail,
            max_num=max_dynamic_patch,
        )
    else:
        tiles = dynamic_preprocess(
            pil_image,
            image_size=force_image_size,
            use_thumbnail=False,
            max_num=1,
        )
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values


def build_vlm_prompt(
    history_trajectory: torch.Tensor,
    high_command_one_hot: torch.Tensor,
) -> str:
    """Reconstruct the exact prompt used for offline InternVL caching."""
    navigation_commands = ["turn left", "go straight", "turn right"]
    command_str = next(
        (navigation_commands[i] for i, v in enumerate(high_command_one_hot) if float(v) == 1),
        "unknown",
    )
    history_str = " ".join(
        [
            f"   - t-{3 - i}: ({_format_number(history_trajectory[i, 0].item())}, "
            f"{_format_number(history_trajectory[i, 1].item())}, "
            f"{_format_number(history_trajectory[i, 2].item())})"
            for i in range(4)
        ]
    )
    prompt = (
        f"<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
        f"1. Visual perception from front camera view\n"
        f"2. Historical motion context (last 4 timesteps):{history_str}\n"
        f"3. Active navigation command: [{command_str.upper()}]"
    )
    output_requirements = (
        "\nOutput requirements:\n- Predict 8 future trajectory points\n"
        "- Each point format: (x:float, y:float, heading:float)\n"
        "- Use [PT, ...] to encapsulate the trajectory\n"
        "- Maintain numerical precision to 2 decimal places"
    )
    return f"{prompt}{output_requirements}"


class OnlineVLMEncoder:
    """Loads InternVL 2B once (per rank) and produces `vlm_hidden_state` online.

    NOTE: plain object on purpose (see module docstring).
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "internvl",
        device: str = "cuda",
        num_image_token: int = 256,
        max_length: int = 2800,
    ):
        from transformers import AutoModel, AutoTokenizer

        if model_type.lower() != "internvl":
            raise ValueError(
                f"Unsupported vlm_online_model_type: '{model_type}'. Only 'internvl' is supported."
            )
        if not checkpoint_path:
            raise ValueError(
                "OnlineVLMEncoder requires a valid checkpoint path "
                "(agent.config.vlm_online_checkpoint_path)."
            )

        self.device = device
        self.num_image_token = num_image_token
        self.max_length = max_length

        logger.info("Loading online InternVL VLM from: %s (device=%s)", checkpoint_path, device)
        self.model = AutoModel.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=True,
            device_map=device,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model.system_message = SYSTEM_MESSAGE
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Online InternVL VLM loaded on device '%s'.", device)

    @torch.no_grad()
    def encode(
        self,
        pixel_values: torch.Tensor,
        history_trajectory: torch.Tensor,
        high_command_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        """Run InternVL on one sample and return `last_hidden_state` [max_length, D].

        :param pixel_values: [num_patches, 3, H, W] tensor for the front camera.
        :param history_trajectory: [4, 3] history poses for prompt reconstruction.
        :param high_command_one_hot: [4] driving command one-hot for the prompt.
        :return: last hidden state of shape [max_length, hidden_dim], float32, detached.
        """
        device = next(self.model.parameters()).device
        num_patches = int(pixel_values.shape[0])
        question = build_vlm_prompt(history_trajectory, high_command_one_hot)

        template = get_conv_template("internvl2_5")
        template.system_message = SYSTEM_MESSAGE
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        image_tokens = (
            IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        )
        query = query.replace("<image>", image_tokens, 1)

        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(
            [query], return_tensors="pt", padding="max_length", max_length=self.max_length
        )
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        image_flags = torch.ones(num_patches, dtype=torch.long, device=device)

        outputs = self.model(
            pixel_values=pixel_values.to(device).bfloat16(),
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags.squeeze(-1),
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = outputs.hidden_states[-1]  # [1, max_length, D]
        return last_hidden_state.squeeze(0).float().detach()

    @torch.no_grad()
    def encode_batch(
        self,
        pixel_values_list: List[torch.Tensor],
        history_trajectory_batch: torch.Tensor,
        high_command_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch sample-by-sample and stack into [B, max_length, D]."""
        hidden_states = []
        for i, pixel_values in enumerate(pixel_values_list):
            hidden_states.append(
                self.encode(
                    pixel_values,
                    history_trajectory_batch[i],
                    high_command_batch[i],
                )
            )
        return torch.stack(hidden_states, dim=0)
