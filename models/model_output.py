def make_model_output(x1=None, x2=None, x3=None, output=None, mask_logits=None, features=None):
    """Build the shared model-output dictionary used by training/eval/predict."""
    if output is None:
        output = x3
    return {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "output": output,
        "mask_logits": mask_logits,
        "features": features or {},
    }


def ensure_model_output(value):
    """Accept the new dict output and legacy tuple output during migration."""
    if isinstance(value, dict):
        if "output" not in value and "x3" in value:
            value = dict(value)
            value["output"] = value["x3"]
        if "features" not in value:
            value = dict(value)
            value["features"] = {}
        return value

    if isinstance(value, (tuple, list)):
        if len(value) == 5:
            x1, x2, x3, output, mask_logits = value
            return make_model_output(x1, x2, x3, output, mask_logits)
        if len(value) == 4:
            x1, x2, x3, mask_logits = value
            return make_model_output(x1, x2, x3, x3, mask_logits)

    raise TypeError(f"Unsupported model output type: {type(value)!r}")
