"""One check: fit_prompt trims evidence, never instructions. PYTHONPATH=code python -m generation.test_fit_prompt"""
from generation.prompting import build_prompt, fit_prompt


def test():
    chunks = [{"doc_id": "d", "page_start": i, "page_end": i, "text": f"C{i} " + "x" * 1000} for i in range(9)]
    tables = [{"doc_id": "d", "page_start": i, "page_end": i, "caption_raw": f"cap{i}", "markdown": "y" * 1000} for i in range(3)]
    full = build_prompt("note", chunks, "", tables)

    prompt, dc, dt = fit_prompt(len(full) - 2500, "note", chunks, "", tables)
    assert len(prompt) <= len(full) - 2500 and prompt.endswith(full[-300:]), "instructions must survive whole"
    # 9 excerpts and 3 tables at an equal share: the drops alternate excerpt, table, excerpt from the tails
    assert (dc, dt) == (2, 1), (dc, dt)
    assert "C6 " in prompt and "C7 " not in prompt and "cap1" in prompt and "cap2" not in prompt

    assert fit_prompt(len(full), "note", chunks, "", tables) == (full, 0, 0)
    assert fit_prompt(0, "note", chunks, "", tables)[1] == 8, "one excerpt always stays"
    print("ok")


if __name__ == "__main__":
    test()
