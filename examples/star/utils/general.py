import re

MAX_FILE_NAME_LENGTH = 100


def sanitize_filename(name: str, max_len: int = MAX_FILE_NAME_LENGTH) -> str:
    """
    Clean and truncate a string to make it a valid and safe filename.
    """
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    name = name.replace('/', '_')
    max_len = min(len(name), max_len)
    return name[:max_len]
