import os
from pathlib import Path

ROOT = "data0608_2_aug"

fixed = 0

for txt in Path(ROOT).rglob("*.txt"):

    lines_out = []

    with open(txt, "r") as f:
        for line in f:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls = str(int(float(parts[0])))

            lines_out.append(
                " ".join([cls] + parts[1:])
            )

    with open(txt, "w") as f:
        f.write("\n".join(lines_out) + "\n")

    fixed += 1

print("수정한 파일:", fixed)