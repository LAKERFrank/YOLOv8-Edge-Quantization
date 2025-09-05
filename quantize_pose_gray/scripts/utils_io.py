import os


def ensure_exists(path, kind="file"):
    if kind == "file":
        assert os.path.isfile(path), f"Missing file: {path}"
    else:
        assert os.path.isdir(path), f"Missing dir: {path}"
