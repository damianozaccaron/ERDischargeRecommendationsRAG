"""
Deterministic check for "did dosages arrive": counts dose mentions (number + unit) in each generated
output and in the physician-written original it is compared against. No LLM involved.

Run from the project root:  python code/evaluation/dose_check.py ["Examples/results/output Claude"]
"""
import re
from pathlib import Path

DOSE = re.compile(r'(?<!\w)\d+(?:[.,]\d+)?\s?(?:mg|g|mcg|µg|ug|IU|UI|U|mL|ml)(?:/kg)?(?!\w)', re.IGNORECASE)


def doses(text: str) -> list[str]:
    return sorted(set(m.group(0).lower().replace(" ", "") for m in DOSE.finditer(text)))


def main(output_dir=Path("Examples/results/output"), rec_dir=Path("Examples/recommendations")):
    print(f"{'case':18} {'original':>8} {'generated':>9}  generated doses")
    for out in sorted(output_dir.glob("*_output.txt")):
        stem = out.stem.removesuffix("_output")
        generated = out.read_text(encoding="utf-8").split("ORIGINAL RECOMMENDATIONS")[0]  # main.py appends the original
        rec = rec_dir / f"{stem}_rec.txt"
        original = rec.read_text(encoding="utf-8") if rec.exists() else ""
        g, o = doses(generated), doses(original)
        print(f"{stem:18} {len(o):>8} {len(g):>9}  {', '.join(g)}")


if __name__ == "__main__":
    import sys
    assert doses("enoxaparina 8.000 UI s.c. ogni 12 ore, 2,5 mg e 100 UI/kg") == ["100ui/kg", "2,5mg", "8.000ui"]
    main(Path(sys.argv[1])) if len(sys.argv) > 1 else main()
