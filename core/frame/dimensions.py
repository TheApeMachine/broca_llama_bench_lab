"""Fixed dimensions of the substrate's continuous frame representations.

These widths are not tunables. They are derived from the subword sketch width,
the number of frame fields that get sketched, the size of the numeric faculty
tail, and the VSA injection width. Changing one without changing the rest
breaks the host's projector and every persisted feature vector.
"""

from __future__ import annotations


class FrameDimensions:
    """Single source of truth for every continuous-frame width in the system.

    ``SKETCH_DIM`` is the width of one subword sketch.
    ``SKETCH_SEEDS`` is the number of independent hash seeds the sketch sums.
    ``NGRAM_MIN`` / ``NGRAM_MAX`` bound the character n-gram window.
    ``NUMERIC_FEATURE_FIELDS`` lists the scalars appended after the three
    sketches (intent, subject, answer); their order is part of the wire format.
    ``COGNITIVE_FRAME_DIM`` = three sketches + the numeric tail.
    ``VSA_INJECTION_DIM`` is the width of the sparse-projected hypervector tail.
    ``BROCA_FEATURE_DIM`` = cognitive frame + VSA injection.
    """

    SKETCH_DIM: int = 128
    SKETCH_SEEDS: int = 8
    NGRAM_MIN: int = 3
    NGRAM_MAX: int = 5

    NUMERIC_FEATURE_FIELDS: tuple[str, ...] = (
        "confidence",
        "p_do_positive",
        "p_do_negative",
        "ate",
        "policy_listen",
        "policy_open_left",
        "policy_open_right",
        "delta_ce",
        "bias",
    )

    @classmethod
    def numeric_tail_len(cls) -> int:
        return len(cls.NUMERIC_FEATURE_FIELDS)

    @classmethod
    def cognitive_frame_dim(cls) -> int:
        return cls.SKETCH_DIM * 3 + cls.numeric_tail_len()

    VSA_INJECTION_DIM: int = 64

    @classmethod
    def broca_feature_dim(cls) -> int:
        return cls.cognitive_frame_dim() + cls.VSA_INJECTION_DIM
