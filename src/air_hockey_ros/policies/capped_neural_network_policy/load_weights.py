import importlib
import pickle

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Allow torch to fully restore tensor types
        if module.startswith("torch"):
            return getattr(importlib.import_module(module), name)

        # Replace unknown classes (e.g. training.PPOAgentPolicy) with dummy
        class Dummy: pass
        return Dummy

def _collect_tensors(mod, prefix="", out=None):
    if out is None:
        out = {}
    dot = "" if prefix == "" else "."
    # parameters
    params = getattr(mod, "_parameters", None)
    if isinstance(params, dict):
        for k, v in params.items():
            if v is not None:
                out[f"{prefix}{dot}{k}" if prefix else k] = v
    # buffers (e.g., running_mean/var if present)
    bufs = getattr(mod, "_buffers", None)
    if isinstance(bufs, dict):
        for k, v in bufs.items():
            if v is not None:
                out[f"{prefix}{dot}{k}" if prefix else k] = v
    # children
    children = getattr(mod, "_modules", None)
    if isinstance(children, dict):
        for name, child in children.items():
            _collect_tensors(child, f"{prefix}{dot}{name}" if prefix else name, out)
    return out

def load_ac_state_dict_from_pkl(pkl_path, map_location="cpu"):
    with open(pkl_path, "rb") as f:
        obj = SafeUnpickler(f).load()
    net = obj.network  # Dummy wrapper, but holds real torch tensors inside
    raw = _collect_tensors(net)  # flat dict of tensors with proper dotted names
    # keep only the keys your real model expects
    sd = {}
    for k, v in raw.items():
        # keep module params/buffers and 'actor_logstd' param on the wrapper
        if k.startswith(("shared_net.", "actor_mean.", "critic.")) or k == "actor_logstd":
            sd[k] = v.to(map_location)
    return sd