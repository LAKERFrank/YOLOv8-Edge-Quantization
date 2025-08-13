import onnx, sys
m = onnx.load(sys.argv[1])
print(f"IR version={m.ir_version}, opset={m.opset_import[0].version}")
print("GRAPH OUTPUTS:")
for o in m.graph.output:
    print("  -", o.name)
print("\nNODES (first 200):")
for i, n in enumerate(m.graph.node[:200]):
    print(f"{i:03d}: op={n.op_type} name={n.name} inputs={list(n.input)} outputs={list(n.output)}")
