# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# openpi_fast model factory

import glob
import os

import torch
from omegaconf import DictConfig

import safetensors.torch as safetensors_torch

from rlinf.models.embodiment.openpi_fast.openpi_fast_action_model import (
    OpenPiFastConfig,
    OpenPiFastForRLActionPrediction,
)


def get_model(cfg: DictConfig, torch_dtype=None) -> OpenPiFastForRLActionPrediction:
    """Build and return an ``OpenPiFastForRLActionPrediction`` model.

    Mirrors the pattern in ``rlinf/models/embodiment/openpi/__init__.py``:

    1. Resolve the openpi ``TrainConfig`` by ``config_name`` (with optional
       ``model_path`` and ``data_kwargs`` overrides).
    2. Build ``OpenPiFastConfig`` from the resolved train config, then apply
       any per-key overrides from ``cfg.openpi``.
    3. Instantiate ``OpenPiFastForRLActionPrediction``.
    4. Optionally freeze PaliGemma when ``train_expert_only=True``.
    5. Load weights from the checkpoint directory (supports both
       ``full_weights.pt`` checkpoints saved by FSDP and ``.safetensors``
       files from the original openpi release).
    6. Wire input/output transforms using openpi data-config machinery and
       norm stats loaded from the checkpoint.

    Args:
        cfg: Hydra DictConfig for the actor model.  Expected keys:

            - ``cfg.openpi.config_name``: openpi config name string.
            - ``cfg.model_path``: Path to checkpoint or weight directory.
            - ``cfg.openpi.*``: Optional per-field overrides for
              ``OpenPiFastConfig``.
            - ``cfg.openpi_data`` (optional): Extra data-config kwargs.

        torch_dtype: Unused; kept for API consistency.

    Returns:
        Initialised ``OpenPiFastForRLActionPrediction`` with weights loaded and
        transforms set up.
    """
    import openpi.shared.download as download
    import openpi.transforms as transforms
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
    )

    # openpi_model_config is the Pi0FASTConfig from openpi (has model_type=PI0_FAST),
    # used for data config creation so transforms dispatch correctly.
    openpi_model_config = actor_train_config.model

    # Build OpenPiFastConfig from defaults, then apply per-key overrides from cfg.
    fast_model_config = OpenPiFastConfig(
        config_name=config_name,
    )
    override_kwargs = cfg.openpi
    if override_kwargs is not None:
        for key, val in override_kwargs.items():
            if hasattr(fast_model_config, key):
                fast_model_config.__dict__[key] = val

    # Instantiate model.
    model = OpenPiFastForRLActionPrediction(fast_model_config)

    if fast_model_config.train_expert_only:
        model.freeze_vlm()

    # -----------------------------------------------------------------------
    # Load weights
    # -----------------------------------------------------------------------
    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    if os.path.exists(full_weights_path):
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    else:
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        all_state_dict: dict = {}
        for weight_path in weight_paths:
            state_dict = safetensors_torch.load_file(weight_path, device="cpu")
            all_state_dict.update(state_dict)
        model.load_state_dict(all_state_dict, strict=False)

    # Re-apply bfloat16 precision after weight loading (safetensors may load
    # parameters as float32 depending on the stored dtype).
    model.to_bfloat16_for_selected_params("bfloat16")

    # -----------------------------------------------------------------------
    # Set up input / output transforms
    # -----------------------------------------------------------------------
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, openpi_model_config
    )
    norm_stats = None
    if data_config.asset_id is None:
        raise ValueError(
            "Asset id is required to load norm stats for Pi0-Fast.  "
            "Set assets_dir or asset_id in the data config."
        )
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)

    repack_transforms = transforms.Group()
    # default_prompt is None here; set to a string to inject a fixed task prompt
    # for all samples (useful for single-task fine-tuning).
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
