from src.config import HF_TOKEN, WANDB_API_KEY
import huggingface_hub
import wandb

def signin() -> None:
    """
    Imports tokens from config and, if the respective tokens exist, logs into HuggingFace and Wandb.
    """
    if HF_TOKEN:
        huggingface_hub.login()
    if WANDB_API_KEY:
        wandb.login()