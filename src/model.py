import torch
import os

CHECKPOINT_PATH = "checkpoints/gpt.pt"


def load_model():
    """
    Load a trained model checkpoint safely.
    Supports multiple checkpoint formats.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        print("⚠️ No checkpoint found — running in policy-only mode")
        return None

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    # Case 1: checkpoint IS the model
    if hasattr(checkpoint, "eval"):
        checkpoint.eval()
        return checkpoint

    # Case 2: checkpoint is a dict with state_dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            print("⚠️ state_dict found but no model class — policy-only mode")
            return None

        if "model" in checkpoint:
            model = checkpoint["model"]
            model.eval()
            return model

    print("⚠️ Unknown checkpoint format — policy-only mode")
    return None


def generate_action(model, prompt: str) -> str:
    """
    Advisory AI signal only.
    Policy engine ALWAYS overrides.
    """

    text = prompt.lower()

    if "privilege" in text or "sudo" in text or "admin" in text:
        return "ESCALATE"

    if "exfiltrate" in text or "dump" in text or "export":
        return "BLOCK"

    return "LOG"
