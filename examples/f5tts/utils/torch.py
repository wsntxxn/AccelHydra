import torch
from torch.nn.utils.rnn import pad_sequence


def mask_from_start_end_indices(
    seq_len: "int['b']", start: "int['b']", end: "int['b']"
):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: "int['b']", frac_lengths: "float['b']"):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> "int['b nt']":
    list_tensors = [
        torch.tensor([*bytes(t, "UTF-8")]) for t in text
    ]  # ByT5 style
    text = pad_sequence(
        list_tensors, padding_value=padding_value, batch_first=True
    )
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> "int['b nt']":
    list_idx_tensors = [
        torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text
    ]  # pinyin or char style
    text = pad_sequence(
        list_idx_tensors, padding_value=padding_value, batch_first=True
    )
    return text
