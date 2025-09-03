import argparse
import onnx
import re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--nodes-file')
    ap.add_argument('--regex')
    args = ap.parse_args()

    model = onnx.load(args.input)
    names = set()
    if args.nodes_file:
        with open(args.nodes_file, 'r', encoding='utf-8') as f:
            names.update(ln.strip() for ln in f if ln.strip())
    if args.regex:
        pat = re.compile(args.regex)
        for n in model.graph.node:
            for o in n.output:
                if pat.search(o):
                    names.add(o)
    for name in names:
        if name not in [o.name for o in model.graph.output]:
            model.graph.output.extend([onnx.ValueInfoProto(name=name)])
    onnx.save(model, args.output)
    print('added outputs:', names)

if __name__ == '__main__':
    main()
