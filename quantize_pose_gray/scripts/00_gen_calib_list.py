import argparse
from pathlib import Path

EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def main():
    ap = argparse.ArgumentParser(description="Populate calib_list.txt from val_images directory")
    ap.add_argument('--src', default='data/val_images', help='directory containing images')
    ap.add_argument('--out', default='data/calib_list.txt', help='output list file')
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"WARNING: {src} not found; writing empty list")
        paths = []
    else:
        paths = sorted(p.relative_to(Path.cwd()).as_posix()
                       for p in src.rglob('*')
                       if p.suffix.lower() in EXTS)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for p in paths:
            f.write(p + '\n')
    print(f"wrote {len(paths)} paths to {out_path}")


if __name__ == '__main__':
    main()
