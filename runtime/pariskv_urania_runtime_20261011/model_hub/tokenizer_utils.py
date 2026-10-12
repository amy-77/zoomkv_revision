"""健壮的 tokenizer 加载器。

背景：在 transformers 5.x（实测 5.9.0）下，部分模型（如 DeepSeek-R1-Distill-Llama-8B，
其 tokenizer_class=LlamaTokenizerFast 但仓库内只有 tokenizer.json）经由
`AutoTokenizer.from_pretrained(..., trust_remote_code=True)` 会回退到一个损坏的实现：
encode/decode 往返会把所有空格丢失（"I have this" -> "Ihavethis"），既污染喂给模型的
input_ids，也污染保存的输出文本，导致评测结果接近随机且出现 Ġ/Ċ 等 byte-level artifact。

本模块提供 `load_tokenizer`：先用常规方式加载，做一次"含空格往返自检"，若发现空格被破坏，
则回退到 `PreTrainedTokenizerFast.from_pretrained`（实测能正确往返且携带 chat_template）。
"""

from transformers import AutoTokenizer, PreTrainedTokenizerFast

# 含空格 / 换行 / 符号的探针；正常 tokenizer 应能无损往返
_PROBE = "Okay, so I have this problem.\n\nThe correct answer is (B)."


def _roundtrip_ok(tokenizer) -> bool:
    try:
        ids = tokenizer.encode(_PROBE, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
    except Exception:
        return False
    # 关键判据：空格必须被保留（损坏实现会把空格全部吞掉）
    return decoded == _PROBE


def load_tokenizer(model_name: str, trust_remote_code: bool = True, **kwargs):
    """加载 tokenizer，并在检测到空格被破坏时回退到 PreTrainedTokenizerFast。

    Args:
        model_name: 模型路径或 HF repo id。
        trust_remote_code: 透传给 AutoTokenizer。
        **kwargs: 透传给 from_pretrained（如 padding_side 等）。
    """
    tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, **kwargs
    )
    if _roundtrip_ok(tok):
        return tok

    # 回退：直接走 tokenizers 后端的 fast tokenizer，from_pretrained 会带上
    # tokenizer_config.json 里的 chat_template / 特殊 token 配置。
    try:
        fast = PreTrainedTokenizerFast.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, **kwargs
        )
    except Exception as exc:  # 回退失败时退回原 tokenizer，不阻断流程
        print(
            f"[tokenizer_utils] WARNING: AutoTokenizer round-trip broken for "
            f"{model_name}, and PreTrainedTokenizerFast fallback failed: {exc!r}. "
            f"Proceeding with the (possibly broken) AutoTokenizer."
        )
        return tok

    if _roundtrip_ok(fast):
        print(
            f"[tokenizer_utils] AutoTokenizer for {model_name} corrupted whitespace; "
            f"falling back to PreTrainedTokenizerFast ({type(fast).__name__})."
        )
        return fast

    print(
        f"[tokenizer_utils] WARNING: both AutoTokenizer and PreTrainedTokenizerFast "
        f"fail the whitespace round-trip for {model_name}. Outputs may be corrupted."
    )
    return tok
