"""
Main orchestrator for Cross-Reasoning Consistency Auto-CoT experiment.
Handles configuration and invokes inference script.
"""

import sys
import subprocess
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for experiment orchestration.
    
    This function:
    1. Loads Hydra configuration
    2. Applies mode-specific overrides
    3. Invokes inference script as a subprocess
    """
    
    # Print configuration
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Validate configuration
    if not hasattr(cfg, 'run') or not hasattr(cfg.run, 'run_id'):
        raise ValueError("Missing run configuration. Use run=<run_id> in command line.")
    
    # Apply mode-specific overrides
    mode = cfg.mode
    print(f"Running in {mode} mode")
    
    # For this inference-only task, we directly call the inference function
    # instead of using subprocess (simpler for LLM API-based experiments)
    from src.inference import run_inference
    
    try:
        accuracy = run_inference(cfg)
        print(f"\nExperiment completed successfully!")
        print(f"Final accuracy: {accuracy:.4f}")
        return 0
    except Exception as e:
        print(f"\nExperiment failed with error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
