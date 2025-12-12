import os
import sys
import warnings
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))  # ADPolicy/rl_framework/rl_modules/
uniad_ft = os.path.dirname(os.path.dirname(current_dir)) # ADPolicy/
uniad_path = os.path.join(uniad_ft, 'UniAD_SIM')

from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model

def create_uniad(uniad_config: dict):

    config_path = uniad_config['config_path']
    checkpoint_path = uniad_config['checkpoint_path']
    device = uniad_config['device']
    warnings.filterwarnings("ignore")

    try:
        from mmdet.core.bbox.match_costs.builder import MATCH_COST

        if hasattr(MATCH_COST, '_module_dict') and 'DiceCost' in MATCH_COST._module_dict:
            del MATCH_COST._module_dict['DiceCost']
            print("✅ Cleared DiceCost from MATCH_COST")

        from mmdet.models.builder import LOSSES
        if hasattr(LOSSES, '_module_dict') and 'DiceLoss' in LOSSES._module_dict:
            del LOSSES._module_dict['DiceLoss']
            print("✅ Cleared DiceLoss from LOSSES")
        
    except Exception as e:
        print(f"⚠️ Registry cleanup failed: {e}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading config from: {config_path}")
    print(f"Loading checkpoint from: {checkpoint_path}")

    uniad_root = uniad_path
    if uniad_root not in sys.path:
        sys.path.insert(0, uniad_root)
        print(f"Added UniAD root to sys.path: {uniad_root}")

    original_cwd = os.getcwd()
    print(f"Original working directory: {original_cwd}")
    
    config_path_abs = os.path.abspath(config_path)
    checkpoint_path_abs = os.path.abspath(checkpoint_path)
    
    try:
        os.chdir(uniad_root)
        print(f"Changed working directory to: {os.getcwd()}")
        
        config_rel_path = os.path.relpath(config_path_abs, uniad_root)
        print(f"Config relative path: {config_rel_path}")
        
        cfg = Config.fromfile(config_rel_path)

        if cfg.get('custom_imports', None):
            try:
                from mmcv.utils import import_modules_from_strings
                import_modules_from_strings(**cfg['custom_imports'])
                print("Custom imports loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load custom imports: {e}")

        if hasattr(cfg, 'plugin') and cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(config_rel_path)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
            print("CUDNN benchmark enabled")

        print("Building model...")
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        print("Model built successfully")

        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
            print("FP16 enabled")

        checkpoint = load_checkpoint(model, checkpoint_path_abs, map_location='cpu')
        print("Checkpoint loaded successfully")

        model = fuse_conv_bn(model)
        print("Conv-BN fusion completed")

        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']

        device_ids = [int(device.split(':')[1])]
        model = MMDataParallel(model, device_ids=device_ids).eval()
        
        print("UniAD model loaded successfully!")
        return model
    finally:
        os.chdir(original_cwd)
        print(f"Restored working directory to: {os.getcwd()}")