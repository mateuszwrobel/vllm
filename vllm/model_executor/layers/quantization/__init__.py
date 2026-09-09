# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, get_args

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)

QuantizationMethods = Literal[
    "awq",
    "auto_awq",
    "fp8",
    "fbgemm_fp8",
    "fp_quant",
    "modelopt",
    "modelopt_fp4",
    "modelopt_mxfp8",
    "modelopt_mixed",
    "auto_gptq",
    "gptq",
    "gptq_marlin",
    "awq_marlin",
    "humming",
    "compressed-tensors",
    "experts_int8",
    "quark",
    "moe_wna16",
    "torchao",
    "inc",
    "mxfp4",
    "gpt_oss_mxfp4",
    "deepseek_v4_fp8",
    "online",
    # Below are online quant shorthand names (see vllm.config.quantization).
    # Listed here as strings to avoid a circular import; kept in sync with
    # _ONLINE_SHORTHANDS by the assertion in get_quantization_config().
    "fp8_per_tensor",
    "fp8_per_block",
    "fp8_per_channel",
    "int8_per_channel_weight_only",
    "nvfp4_per_token",
    "mxfp8",
]
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

DEPRECATED_QUANTIZATION_METHODS = [
    "fbgemm_fp8",
    "fp_quant",
]

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from vllm.model_executor.layers.quantization import (
        ...     register_quantization_config,
        ... )
        >>> from vllm.model_executor.layers.quantization import get_quantization_config
        >>> from vllm.model_executor.layers.quantization.base_config import (
        ...     QuantizationConfig,
        ... )
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            logger.debug(
                "The quantization method '%s' already exists and will be "
                "overwritten by the quantization config %s.",
                quantization,
                quant_config_cls,
            )
        else:
            QUANTIZATION_METHODS.append(quantization)
            # Automatically assume the custom quantization config is supported
            if sq := current_platform.supported_quantization:
                sq.append(quantization)

        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # lazy import to avoid triggering `torch.compile` too early
    from vllm.config.quantization import _ONLINE_SHORTHANDS
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    from vllm.models.deepseek_v4 import DeepseekV4FP8Config

    from .auto_awq import AutoAWQConfig
    from .auto_gptq import AutoGPTQConfig
    from .compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from .experts_int8 import ExpertsInt8Config
    from .fbgemm_fp8 import FBGEMMFp8Config
    from .fp8 import Fp8Config
    from .fp_quant import FPQuantConfig
    from .humming import HummingConfig
    from .inc import INCConfig
    from .modelopt import (
        ModelOptFp8Config,
        ModelOptMixedPrecisionConfig,
        ModelOptMxFp8Config,
        ModelOptNvFp4Config,
    )
    from .moe_wna16 import MoeWNA16Config
    from .mxfp4 import GptOssMxfp4Config, Mxfp4Config
    from .online.base import OnlineQuantizationConfig
    from .torchao import TorchAOConfig

    method_to_config: dict[str, type[QuantizationConfig]] = {
        "awq": AutoAWQConfig,
        "awq_marlin": AutoAWQConfig,
        "auto_awq": AutoAWQConfig,
        "fp8": Fp8Config,
        "fbgemm_fp8": FBGEMMFp8Config,
        "fp_quant": FPQuantConfig,
        "modelopt": ModelOptFp8Config,
        "modelopt_fp4": ModelOptNvFp4Config,
        "modelopt_mxfp8": ModelOptMxFp8Config,
        "modelopt_mixed": ModelOptMixedPrecisionConfig,
        "auto_gptq": AutoGPTQConfig,
        "gptq": AutoGPTQConfig,
        "gptq_marlin": AutoGPTQConfig,
        "compressed-tensors": CompressedTensorsConfig,
        "experts_int8": ExpertsInt8Config,
        "quark": QuarkConfig,
        "moe_wna16": MoeWNA16Config,
        "torchao": TorchAOConfig,
        "inc": INCConfig,
        "mxfp4": Mxfp4Config,
        "gpt_oss_mxfp4": GptOssMxfp4Config,
        "deepseek_v4_fp8": DeepseekV4FP8Config,
        "humming": HummingConfig,
        "online": OnlineQuantizationConfig,
        # MiniMax-style checkpoints tag `quant_method: "mxfp8"`; load with the
        # ModelOpt MXFP8 config (same format). The "mxfp8" online shorthand
        # below only applies to the `--quantization mxfp8` CLI path.
        "mxfp8": ModelOptMxFp8Config,
    }

    # Register online shorthands (e.g. "fp8_per_tensor") as quant methods.
    # setdefault so a shorthand that is also a checkpoint method (e.g. "mxfp8")
    # keeps its checkpoint config; the shorthand still works via the
    # `--quantization` CLI path in `resolve_quantization_config`.
    for shorthand in _ONLINE_SHORTHANDS:
        method_to_config.setdefault(shorthand, OnlineQuantizationConfig)

    # Update the `method_to_config` with customized quantization methods.
    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)

    return method_to_config[quantization]


__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]

# radiance: register the gfx1201 AutoRound int4 W4A8 config. Must happen before ModelConfig
# resolves the checkpoint's quant_method, otherwise vLLM's INC config claims "auto-round" and the
# engine dies with "inc quantization is currently not supported in rocm". Import failures are
# reported and swallowed: a broken side module must not take down every other quantization method.
import os as _radiance_os  # noqa: E402

if _radiance_os.environ.get("RADIANCE_AUTOROUND", "0") == "1":
    try:
        from vllm.radiance import radiance_autoround  # noqa: F401,E402
    except Exception as _radiance_exc:  # pragma: no cover
        import sys as _radiance_sys  # noqa: E402
        _radiance_sys.stderr.write(
            f"[radiance.autoround] registration FAILED: {_radiance_exc!r}\n")

# radiance: register the gfx1201 escha (EXL3 trellis) W2 config. Must happen before ModelConfig
# resolves the checkpoint's quant_method; "escha" is unknown to vLLM, so without this the engine
# exits during argument parsing. Import failures are reported and swallowed: a broken side module
# must not take down every other quantization method.
import os as _radiance_escha_os  # noqa: E402

if _radiance_escha_os.environ.get("RADIANCE_ESCHA", "0") == "1":
    try:
        from vllm.radiance import radiance_escha as _radiance_escha_mod  # noqa: E402
        _radiance_escha_mod.register()
    except Exception as _radiance_escha_exc:  # pragma: no cover
        import sys as _radiance_escha_sys  # noqa: E402
        _radiance_escha_sys.stderr.write(
            f"[radiance.escha] registration FAILED: {_radiance_escha_exc!r}\n")
