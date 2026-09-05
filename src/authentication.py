from src.config import HF_TOKEN, WANDB_API_KEY
import huggingface_hub
import wandb

def signin():
    if HF_TOKEN:
        huggingface_hub.login()
    if WANDB_API_KEY:
        wandb.login()