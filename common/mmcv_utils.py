from typing import Tuple


def cleanup_mmcv_registries() -> Tuple[bool, bool]:
    """Remove stale DiceCost/DiceLoss entries that can break MMCV registry imports."""
    cleared_cost = False
    cleared_loss = False
    try:
        from mmdet.core.bbox.match_costs.builder import MATCH_COST
        if hasattr(MATCH_COST, '_module_dict') and 'DiceCost' in MATCH_COST._module_dict:
            del MATCH_COST._module_dict['DiceCost']
            print("✅ Cleared DiceCost from MATCH_COST")
            cleared_cost = True

        from mmdet.models.builder import LOSSES
        if hasattr(LOSSES, '_module_dict') and 'DiceLoss' in LOSSES._module_dict:
            del LOSSES._module_dict['DiceLoss']
            print("✅ Cleared DiceLoss from LOSSES")
            cleared_loss = True
    except Exception as e:
        print(f"⚠️ Registry cleanup failed: {e}")

    return cleared_cost, cleared_loss
