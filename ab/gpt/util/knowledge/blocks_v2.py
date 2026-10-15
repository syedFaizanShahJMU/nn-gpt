"""Block inventory v2: which building blocks a network ALREADY has (input of the web question and of the filter).

Why (E51): the v1 signature (blocks.ast_blocks / retrieve.block_signature) counts only direct nn.X(...) calls and
local classes that are called. It misses blocks the network clearly has, so the web question told the gist service
"already uses: ..." with them missing, and 10 of 11 networks got suggestions for blocks they already had:
  - layers passed as arguments or via partial():   norm_layer=nn.BatchNorm2d, partial(nn.LayerNorm), activation_layer=nn.SiLU
  - functional calls (removed on purpose in v1):    F.dropout, F.adaptive_avg_pool2d
  - blocks imported from torchvision:               SqueezeExcitation, Conv2dNormActivation
  - local classes that are passed, not called:      inception_block = Inception
  - structural features without a layer class:     grouped/depthwise conv (groups=), channel shuffle, residual add
v2 collects all of these. The old functions stay unchanged (Experiments 1-3 used them).

block_inventory(code) -> sorted tuple of names; custom classes keep the trailing '*' (as in v1).
already_present(item, inventory) -> the inventory entries an item of a suggestion refers to, or None.
v2b (E52, checked on the 74 classified items of E50): SE no longer matches inside "_denseblock"; an item with
alternatives ("X or Y") is present only if all alternatives are present.
"""
import ast
import re

NOT_BLOCKS = {"Sequential", "ModuleList", "ModuleDict", "Identity", "Parameter", "Module", "Flatten", "Unflatten",
              "CrossEntropyLoss", "MSELoss", "NLLLoss", "Tensor"}

F_TO_LAYER = {"dropout": "Dropout", "dropout2d": "Dropout2d", "relu": "ReLU", "relu6": "ReLU6", "gelu": "GELU",
              "silu": "SiLU", "hardswish": "Hardswish", "hardsigmoid": "Hardsigmoid", "sigmoid": "Sigmoid",
              "tanh": "Tanh", "leaky_relu": "LeakyReLU", "elu": "ELU", "mish": "Mish", "max_pool2d": "MaxPool2d",
              "avg_pool2d": "AvgPool2d", "adaptive_avg_pool2d": "AdaptiveAvgPool2d",
              "adaptive_max_pool2d": "AdaptiveMaxPool2d", "layer_norm": "LayerNorm", "batch_norm": "BatchNorm2d",
              "group_norm": "GroupNorm", "instance_norm": "InstanceNorm2d", "softmax": "Softmax",
              "interpolate": "Upsample", "scaled_dot_product_attention": "Attention"}

STRUCT_GROUPED = "grouped or depthwise convolution"
STRUCT_SHUFFLE = "channel shuffle"
STRUCT_RESIDUAL = "residual connection"


def _owner(node):
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover
        return ""


def block_inventory(code: str) -> tuple:
    tree = ast.parse(code)
    local = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)} - {"Net"}
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] in ("torch", "torchvision", "timm"):
            imported |= {a.asname or a.name for a in n.names if (a.asname or a.name)[:1].isupper()}
    found = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            owner = _owner(n.value)
            if owner in ("nn", "torch.nn") and n.attr[:1].isupper():
                found.add(n.attr)                                    # nn.X anywhere: call, argument, partial(...)
            elif owner.split(".")[-1] in ("ops", "misc") and n.attr[:1].isupper():
                found.add(n.attr)                                    # torchvision.ops.SqueezeExcitation
            elif owner in ("F", "torch.nn.functional") and n.attr in F_TO_LAYER:
                found.add(F_TO_LAYER[n.attr])                        # F.dropout -> Dropout
            elif owner == "torch" and n.attr in F_TO_LAYER:
                found.add(F_TO_LAYER[n.attr])                        # torch.relu -> ReLU
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id in local:
                found.add(n.id + "*")                                # local class, called or passed
            elif n.id in imported:
                found.add(n.id)                                      # imported class that is used
        elif isinstance(n, ast.keyword) and n.arg == "groups":
            if not (isinstance(n.value, ast.Constant) and n.value.value == 1):
                found.add(STRUCT_GROUPED)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)) and "shuffle" in n.name.lower():
            found.add(STRUCT_SHUFFLE)
        elif isinstance(n, ast.Call) and "shuffle" in _owner(n.func).lower():
            found.add(STRUCT_SHUFFLE)
    for fn in ast.walk(tree):  # residual add inside a forward(): out + x / x + identity / += shortcut
        if isinstance(fn, ast.FunctionDef) and fn.name == "forward":
            args = {a.arg for a in fn.args.args[1:]}
            names = args | {"identity", "shortcut", "residual", "skip"}
            for n in ast.walk(fn):
                ops = []
                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
                    ops = [n.left, n.right]
                elif isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add):
                    ops = [n.value]
                if any(isinstance(o, ast.Name) and o.id in names for o in ops) or \
                        any("shortcut" in _owner(o) or "downsample" in _owner(o) for o in ops):
                    found.add(STRUCT_RESIDUAL)
    return tuple(sorted(b for b in found if b.rstrip("*") not in NOT_BLOCKS))


# words used in suggestions -> pattern that must match an inventory entry for the item to count as already present
ALIASES = [
    (r"squeeze.?(and.?)?excitation|\bse.?(block|layer|module|net)?s?\b|\bsenet", r"^(se|selayer|seblock|semodule)$|squeeze|excitation"),
    (r"swish|silu", r"silu|swish"),
    (r"batch.?norm", r"batchnorm"),
    (r"layer.?norm", r"layernorm"),
    (r"group.?norm", r"groupnorm"),
    (r"instance.?norm", r"instancenorm"),
    (r"dropout", r"dropout"),
    (r"global.?av(era)?g.?pool|adaptive.?avg", r"adaptiveavgpool"),
    (r"max.?pool", r"maxpool"),
    (r"\bgelu", r"gelu"),
    (r"depthwise|separable|grouped.?conv", STRUCT_GROUPED),
    (r"channel.?shuffle", STRUCT_SHUFFLE),
    (r"residual|skip.?connection|shortcut", STRUCT_RESIDUAL + r"|residual|bottleneck"),
    (r"inception", r"inception"),
    (r"multi.?head.?attention|self.?attention", r"multiheadattention|attention"),
    (r"grid.?attention", r"partitionattention|gridattention|grid"),
    (r"dense.?(layer|block)", r"denselayer|denseblock"),
]


def _present_one(text, inv):
    for say, have in ALIASES:
        if re.search(say, text):
            hit = next((b for b in inv if re.search(have, b)), None)
            if hit:
                return hit
    return None


def already_present(item: str, inventory) -> str | None:
    """An item counts as present only if EVERY alternative it names is present ("NonLocal or SENet" with only SE
    present is still a new suggestion, E52)."""
    inv = [b.lower().rstrip("*") for b in inventory]
    hits = [_present_one(part.strip(), inv) for part in re.split(r"\bor\b|/", item.lower()) if part.strip()]
    return " + ".join(hits) if hits and all(hits) else None
