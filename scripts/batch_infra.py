import contextlib
import io
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from models import Custom, Custom3Tags, IEResNet, IEXResNet, Release, Release_lightly, Release_lightly_extra, ResNet
from models.model_output import ensure_model_output
from utils import get_transformer, maskToTensor, save_tensor_as_image


CONFIG_PATH = (PROJECT_ROOT / "config" / "config.yaml").resolve()
# 批量推理图片目录：图片直接平铺放在该目录下，不递归子目录。
# 设为 None 时，默认使用 config.predict.input_file_path 所在目录。
# 也可以直接改成具体目录，例如：
# BATCH_IMAGE_DIR = (PROJECT_ROOT / "datasets" / "your_batch_images").resolve()
BATCH_IMAGE_DIR = None
# 批量推理输出目录：每张图输出到 output/图片名/ 目录下
BATCH_OUTPUT_DIR = (PROJECT_ROOT / "output").resolve()

OVERWRITE_OUTPUT = True
SAVE_INPUT_IMAGE = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_NAME_ALIASES = {
    "Release_quantized": "Release",
    "resnet": "ResNet",
    "ieresnet": "IEResNet",
    "iexresnet": "IEXResNet",
}


class TensorCanvas:
    def __init__(self, channels, height, width):
        self.tensor = torch.zeros((channels, height, width), dtype=torch.float32)

    def add(self, left, top, block):
        if block.dim() == 4:
            if block.shape[0] != 1:
                raise ValueError("only support batch size 1 when saving prediction blocks")
            block = block.squeeze(0)

        block = block.detach().cpu().float()
        _, block_h, block_w = block.shape

        max_h = max(0, self.tensor.shape[1] - top)
        max_w = max(0, self.tensor.shape[2] - left)
        if max_h == 0 or max_w == 0:
            return

        use_h = min(block_h, max_h)
        use_w = min(block_w, max_w)
        self.tensor[:, top:top + use_h, left:left + use_w] = block[:, :use_h, :use_w]

    def merge(self):
        return self.tensor


def strip_inline_comment(line):
    in_single_quote = False
    in_double_quote = False

    for index, char in enumerate(line):
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == "#" and not in_single_quote and not in_double_quote:
            return line[:index]

    return line


def parse_scalar(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def load_config_without_yaml():
    root = {}
    stack = [(-1, root)]

    for raw_line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        line = strip_inline_comment(raw_line).rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()

        if content.startswith("- "):
            continue

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]

        if content.endswith(":"):
            key = content[:-1].strip()
            current[key] = {}
            stack.append((indent, current[key]))
            continue

        if ":" not in content:
            continue

        key, value = content.split(":", 1)
        current[key.strip()] = parse_scalar(value.strip())

    return root


def load_config():
    try:
        from config.config_utils import LoadConfig
        return LoadConfig()
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        print("[WARN] 当前环境缺少 `yaml`，改用内置简化解析器读取 config/config.yaml。")
        return load_config_without_yaml()


def resolve_config_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        # Keep relative paths consistent with running scripts from the project root.
        path = (PROJECT_ROOT / path_value).resolve()
    return path


def resolve_batch_input_dir(config):
    if BATCH_IMAGE_DIR is not None:
        return Path(BATCH_IMAGE_DIR).resolve()

    predict_input_path = resolve_config_path(config["predict"]["input_file_path"])
    if predict_input_path.is_dir():
        return predict_input_path
    return predict_input_path.parent


def list_flat_images(input_dir):
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_offsets(length, tile_size):
    if length <= tile_size:
        return [0]

    offsets = list(range(0, length - tile_size + 1, tile_size))
    last_offset = length - tile_size
    if offsets[-1] != last_offset:
        offsets.append(last_offset)
    return offsets


def split_positions(width, height, tile_size):
    left_list = make_offsets(width, tile_size)
    top_list = make_offsets(height, tile_size)
    positions = []

    for top in top_list:
        for left in left_list:
            right = min(left + tile_size, width) - 1
            bottom = min(top + tile_size, height) - 1
            positions.append((left, top, right, bottom))

    return positions


def squeeze_image_tensor(tensor):
    if tensor.dim() == 4:
        if tensor.shape[0] != 1:
            raise ValueError("only support batch size 1")
        tensor = tensor.squeeze(0)
    return tensor.detach().cpu()


def normalize_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        normalized[new_key] = value
    return normalized


def build_model(config, device):
    raw_model_name = config["run"]["model"]
    model_name = MODEL_NAME_ALIASES.get(raw_model_name, raw_model_name)

    if raw_model_name != model_name:
        print(f"[WARN] `{raw_model_name}` 在当前仓库中没有独立定义，按 `{model_name}` 结构加载。")

    if model_name == "Release":
        model = Release.ResNet_UNet()
    elif model_name == "Release_lightly":
        model = Release_lightly.ResNet_UNet()
    elif model_name == "Release_lightly_extra":
        model = Release_lightly_extra.ResNet_UNet()
    elif model_name == "Custom":
        custom_cfg = config["custom"]
        model = Custom.ResNet_UNet(
            base=custom_cfg["base"],
            refinement=custom_cfg["refinement"],
            ffp=custom_cfg["ffp"],
            ppm=custom_cfg["ppm"],
            down_sample=custom_cfg["down_sample"],
            am=custom_cfg["am"],
        )
    elif model_name == "Custom3Tags":
        custom_cfg = config["custom"]
        model = Custom3Tags.ResNet_UNet(
            base=custom_cfg["base"],
            refinement=custom_cfg["refinement"],
            ffp=custom_cfg["ffp"],
            ppm=custom_cfg["ppm"],
            down_sample=custom_cfg["down_sample"],
            am=custom_cfg["am"],
        )
    elif model_name == "ResNet":
        model = ResNet.ResNet_UNet()
    elif model_name == "IEResNet":
        model = IEResNet.ResNet_UNet()
    elif model_name == "IEXResNet":
        model = IEXResNet.ResNet_UNet()
    else:
        raise RuntimeError(f"unsupported model: {raw_model_name}")

    return model.to(device), model_name


def load_model_weights(model, model_path, device):
    print(f"[INFO] loading weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
            print("[INFO] weights dict: checkpoint['state_dict']")
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            state_dict = checkpoint["model_state_dict"]
            print("[INFO] weights dict: checkpoint['model_state_dict']")
        else:
            print("[INFO] weights dict: checkpoint")
    else:
        print("[INFO] weights dict: checkpoint")

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"unsupported checkpoint format: {type(checkpoint)}")

    state_dict = normalize_state_dict(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys:
        print(f"[WARN] missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[WARN] unexpected keys: {len(incompatible.unexpected_keys)}")


@torch.no_grad()
def forward_outputs(model, image_tensor, device):
    with contextlib.redirect_stdout(io.StringIO()):
        outputs = ensure_model_output(model(image_tensor))

    mask = torch.argmax(outputs["mask_logits"], dim=1, keepdim=True)
    mask = maskToTensor(mask, device)

    return {
        "x1": squeeze_image_tensor(outputs["x1"]),
        "x2": squeeze_image_tensor(outputs["x2"]),
        "x3": squeeze_image_tensor(outputs["x3"]),
        "output": squeeze_image_tensor(outputs["output"]),
        "mask": squeeze_image_tensor(mask),
    }


def init_canvases(full_size, tile_size, predictions):
    full_width, full_height = full_size
    tile_width, tile_height = tile_size

    canvases = {}
    scales = {}

    for name, tensor in predictions.items():
        channels, pred_h, pred_w = tensor.shape
        scale_x = pred_w / tile_width
        scale_y = pred_h / tile_height
        canvas_w = max(1, int(round(full_width * scale_x)))
        canvas_h = max(1, int(round(full_height * scale_y)))

        canvases[name] = TensorCanvas(channels, canvas_h, canvas_w)
        scales[name] = (scale_x, scale_y)

    return canvases, scales


@torch.no_grad()
def predict_one_image(model, image, transform, device, use_crop, tile_size):
    if not use_crop:
        image_tensor = transform(image).unsqueeze(0).to(device)
        return forward_outputs(model, image_tensor, device)

    positions = split_positions(image.width, image.height, tile_size)
    canvases = None
    scales = None

    for left, top, right, bottom in positions:
        tile = image.crop((left, top, right + 1, bottom + 1))
        tile_tensor = transform(tile).unsqueeze(0).to(device)
        tile_predictions = forward_outputs(model, tile_tensor, device)

        if canvases is None:
            canvases, scales = init_canvases(
                full_size=(image.width, image.height),
                tile_size=(tile.width, tile.height),
                predictions=tile_predictions,
            )

        for name, tensor in tile_predictions.items():
            scale_x, scale_y = scales[name]
            scaled_left = int(round(left * scale_x))
            scaled_top = int(round(top * scale_y))
            canvases[name].add(scaled_left, scaled_top, tensor)

    return {name: canvas.merge() for name, canvas in canvases.items()}


def save_prediction_outputs(output_dir, image, predictions):
    output_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_INPUT_IMAGE:
        image.save(output_dir / "input.png")

    save_tensor_as_image(predictions["x1"], output_dir / "x1.png")
    save_tensor_as_image(predictions["x2"], output_dir / "x2.png")
    save_tensor_as_image(predictions["x3"], output_dir / "x3.png")
    save_tensor_as_image(predictions["output"], output_dir / "output.png")
    save_tensor_as_image(predictions["mask"], output_dir / "mask.png", imagenet_denorm=False)


def main():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dir = resolve_batch_input_dir(config)
    output_root = BATCH_OUTPUT_DIR
    model_path = resolve_config_path(config["predict"]["model_path"])

    print(f"[INFO] config_path: {CONFIG_PATH}")
    print(f"[INFO] run.model from config: {config['run']['model']}")
    print(f"[INFO] predict.model_path from config: {config['predict']['model_path']}")
    print(f"[INFO] resolved model_path: {model_path}")
    print(f"[INFO] predict.input_file_path from config: {config['predict']['input_file_path']}")
    print(f"[INFO] resolved input_dir: {input_dir}")

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"batch image dir not found: {input_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"model path not found: {model_path}")

    image_paths = list_flat_images(input_dir)
    if not image_paths:
        raise RuntimeError(f"no images found in: {input_dir}")

    model, model_name = build_model(config, device)
    print(f"[INFO] built model: {model_name}")
    load_model_weights(model, model_path, device)
    model.eval()

    use_crop = config["run"]["data_crop"]["use"]
    tile_size = config["run"]["data_crop"]["size"]
    transform = get_transformer()

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] device: {device}")
    print(f"[INFO] model: {model_name}")
    print(f"[INFO] model_path: {model_path}")
    print(f"[INFO] input_dir: {input_dir}")
    print(f"[INFO] output_dir: {output_root}")
    print(f"[INFO] image_count: {len(image_paths)}")
    print(f"[INFO] data_crop.use: {use_crop}")
    if use_crop:
        print(f"[INFO] data_crop.size: {tile_size}")

    failed = []
    name_counter = {}

    for index, image_path in enumerate(image_paths, start=1):
        base_name = image_path.stem
        name_counter[base_name] = name_counter.get(base_name, 0) + 1
        if name_counter[base_name] > 1:
            output_name = f"{base_name}_{name_counter[base_name]}"
        else:
            output_name = base_name

        output_dir = output_root / output_name

        if output_dir.exists() and not OVERWRITE_OUTPUT:
            print(f"[SKIP] [{index}/{len(image_paths)}] {image_path.name} -> {output_dir}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            predictions = predict_one_image(
                model=model,
                image=image,
                transform=transform,
                device=device,
                use_crop=use_crop,
                tile_size=tile_size,
            )
            save_prediction_outputs(output_dir, image, predictions)
            print(f"[OK]   [{index}/{len(image_paths)}] {image_path.name} -> {output_dir}")
        except Exception as exc:
            failed.append((image_path, exc))
            print(f"[FAIL] [{index}/{len(image_paths)}] {image_path.name}: {exc}")

    print("-" * 60)
    print(f"[SUMMARY] total: {len(image_paths)}")
    print(f"[SUMMARY] success: {len(image_paths) - len(failed)}")
    print(f"[SUMMARY] failed: {len(failed)}")

    if failed:
        print("[SUMMARY] failed files:")
        for image_path, exc in failed:
            print(f"  - {image_path}: {exc}")


if __name__ == "__main__":
    main()
