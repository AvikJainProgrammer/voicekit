def resolve_device(device: str | None = None) -> str:
    """Resolve a user-supplied device string, defaulting to cuda when available."""
    if device and device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
