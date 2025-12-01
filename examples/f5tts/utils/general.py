def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d
