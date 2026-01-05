from functools import lru_cache
from abc import abstractmethod, ABC
import json

import jieba
from pypinyin import Style, lazy_pinyin
from g2p_en import G2p
import torch
from torch.nn.utils.rnn import pad_sequence


class TokenizerBase(ABC):
    def __init__(self, padding_value: int = -1) -> None:
        self.padding_value = padding_value

    @abstractmethod
    def tokenize_sample(self, text: str) -> list[int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    def __call__(self, text: str | list[int]) -> torch.Tensor:
        batch_tokens = []

        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = text

        for text_ in text_list:
            tokens = self.tokenize_sample(text_)
            batch_tokens.append(tokens)

        if isinstance(text, str):
            return torch.as_tensor(batch_tokens[0])
        else:
            return self.to_batch(batch_tokens)

    def to_batch(self, token_list: list[int]) -> torch.Tensor:
        token_batch = pad_sequence(
            [torch.as_tensor(tokens) for tokens in token_list],
            padding_value=self.padding_value,
            batch_first=True,
        )
        return token_batch


class PhonemeTokenizer(TokenizerBase):
    def __init__(self, phoneme_set: str, padding_value: int = -1) -> None:
        super().__init__(padding_value)
        if phoneme_set.endswith(".json"):
            self.phonemes = json.load(open(phoneme_set, "r"))
        else:
            with open(phoneme_set, "r", encoding="utf-8") as f:
                self.phonemes = [phn.strip() for phn in f]
        self.phoneme2id = {phn: i for i, phn in enumerate(self.phonemes)}
        self.g2p = G2p()

    def tokenize_sample(self, text: str) -> list[int]:
        phonemes = self.g2p(text)
        phonemes = [item.replace(' ', '<BLK>') for item in phonemes]
        phonemes = [item for item in phonemes if item in self.phonemes]
        tokens = [self.phoneme2id.get(p, 0) for p in phonemes]
        return tokens

    @property
    def vocab_size(self) -> int:
        return len(self.phonemes)


class CharacterTokenizer(TokenizerBase):
    def __init__(self, character_set: str, padding_value: int = -1) -> None:
        super().__init__(padding_value)
        if character_set.endswith(".json"):
            self.characters = json.load(open(character_set, "r"))
        else:
            with open(character_set, "r", encoding="utf-8") as f:
                self.characters = [char.strip() for char in f]
        self.character2id = {char: i for i, char in enumerate(self.characters)}

    def tokenize_sample(self, text: str) -> list[int]:
        tokens = [self.character2id.get(c, 0) for c in text]
        return tokens

    @property
    def vocab_size(self) -> int:
        return len(self.characters)


class ByteTokenizer(TokenizerBase):
    def tokenize_sample(self, text: str) -> list[int]:
        tokens = [*bytes(text, "UTF-8")]
        return tokens

    @property
    def vocab_size(self) -> int:
        return 256


@lru_cache(maxsize=32)
def get_tokenizer(tokenizer_path: str, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[
            " "
        ] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


def get_vocab_size(tokenizer_path: str, tokenizer: str = "pinyin"):
    return get_tokenizer(tokenizer_path, tokenizer)[1]


def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans({
        ";": ",",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'"
    })  # add custom trans here, to address oov

    def is_chinese(c):
        return ("\u3100" <= c <= "\u9fff"  # common chinese characters
               )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[
                    -1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(
                seg
            ):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(
                            lazy_pinyin(
                                c, style=Style.TONE3, tone_sandhi=True
                            )
                        )
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list
