from src.config import HF_TOKEN, WANDB_API_KEY
import huggingface_hub
import wandb

def signin():
    if HF_TOKEN:
        huggingface_hub.login(HF_TOKEN)
    if WANDB_API_KEY:
        wandb.login(WANDB_API_KEY)