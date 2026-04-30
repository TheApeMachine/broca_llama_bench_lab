
from __future__ import annotations

from asi_broca_core.benchmarks.hf_datasets_eval import (
    BenchmarkExample,
    build_arc,
    build_boolq,
    build_gsm8k,
    build_hellaswag,
    build_piqa,
    build_winogrande,
    evaluate_example,
    resolve_task_names,
)


class FakeBackend:
    def score_choices(self, prompt, choices, *, normalize=True, chat_template=False):
        # Prefer choices containing yes, B, or the first HellaSwag ending cue.
        scores = []
        for c in choices:
            s = 0.0
            if "yes" in c.lower() or c.strip() in {"B", "2"} or "correct" in c.lower():
                s = 2.0
            scores.append(s)
        return scores, scores[:], [1 for _ in choices]

    def generate(self, prompt, *, max_new_tokens=128, chat_template=True):
        return "The answer is 42."


def test_hf_dataset_builders_normalize_real_rows():
    b = build_boolq({"passage": "Ada wrote code.", "question": "Did Ada write code?", "answer": True}, 0)
    assert b.task == "boolq"
    assert b.choices == (" no", " yes")
    assert b.gold_index == 1

    p = build_piqa({"goal": "open a jar", "sol1": "smash it", "sol2": "twist the lid", "label": 1}, 1)
    assert p.prompt.startswith("Goal:")
    assert p.choices == (" A", " B")
    assert p.gold_index == 1

    arc = build_arc("arc_easy")(
        {
            "question": "Which is correct?",
            "choices": {"label": ["A", "B"], "text": ["wrong", "right"]},
            "answerKey": "B",
        },
        2,
    )
    assert arc is not None
    assert arc.choices == (" A", " B")
    assert arc.gold_index == 1


def test_cloze_and_generation_builders():
    w = build_winogrande(
        {
            "sentence": "The trophy does not fit because _ is too large.",
            "option1": "the suitcase",
            "option2": "the trophy",
            "answer": "2",
        },
        3,
    )
    assert w.prompt.endswith("because ")
    assert w.choices[1].startswith("the trophy")
    assert w.gold_index == 1

    h = build_hellaswag({"ctx": "A person picks up a ball.", "endings": [" wrong", " correct"], "label": "1"}, 4)
    assert h.gold_index == 1

    g = build_gsm8k({"question": "What is 40+2?", "answer": "40+2=42\n#### 42"}, 5)
    assert g.mode == "generate"
    assert g.expected_text == "42"


def test_evaluate_example_with_fake_backend():
    ex = BenchmarkExample("boolq", "0", "Question?", (" no", " yes"), 1)
    row = evaluate_example(FakeBackend(), ex)
    assert row["pred_index"] == 1
    assert row["correct"] is True

    gen = BenchmarkExample("gsm8k", "0", "Problem?", mode="generate", expected_text="42")
    row2 = evaluate_example(FakeBackend(), gen)
    assert row2["correct"] is True


def test_task_resolution_presets_and_errors():
    assert resolve_task_names(None, preset="smoke") == ["boolq", "piqa"]
    assert resolve_task_names("boolq,piqa") == ["boolq", "piqa"]
