"""
Framework factory utilities.
Automatically builds registered framework implementations
based on configuration.

Each framework module (e.g., ManiFlowMultiModal.py) should register itself:
    from hvla.model.tools import FRAMEWORK_REGISTRY

    @FRAMEWORK_REGISTRY.register("ManiFlowMultiModal")
    class ManiFlowMultiModalFramework(baseframework):
        ...
"""

import pkgutil
import importlib
from hvla.model.tools import FRAMEWORK_REGISTRY

from hvla.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

try:
    pkg_path = __path__
except NameError:
    pkg_path = None

# Auto-import all framework submodules to trigger registration
if pkg_path is not None:
    for _, module_name, _ in pkgutil.iter_modules(pkg_path):
        try:
            importlib.import_module(f"{__name__}.{module_name}")
        except Exception as e:
            logger.warning(f"Failed to import framework module {module_name}: {e}")
        
def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: Identifier string (e.g. "ManiFlowMultiModal")
    Returns:
        nn.Module: Instantiated framework model.
    """

    if not hasattr(cfg.framework, "name"): 
        cfg.framework.name = cfg.framework.framework_py  # Backward compatibility for legacy config yaml
        
    if cfg.framework.name == "QwenOFT":
        from hvla.model.framework.QwenOFT import Qwenvl_OFT
        return Qwenvl_OFT(cfg)
    elif cfg.framework.name == "QwenFast":
        from hvla.model.framework.QwenFast import Qwenvl_Fast
        return Qwenvl_Fast(cfg)
    elif cfg.framework.name == "NeuroVLA":
        from hvla.model.framework.NeuroVLA import NeuroVLA
        return NeuroVLA(cfg)

    # auto detect from registry
    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(f"Framework {cfg.framework.name} is not implemented. Plz, python yourframework_py to specify framework module.")
    
    MODLE_CLASS = FRAMEWORK_REGISTRY[framework_id]
    return MODLE_CLASS(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
