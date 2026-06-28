from pathlib import Path

from argentine_deputies_discursive_distance.topic_preprocessing import (
    P1_ADDITIONS,
    PROTECTED_SUBSTANTIVE_TERMS,
    clean_natural_text,
    lexical_tokens,
    load_stopwords,
)

STOPWORDS_PATH = Path("config/topic_modeling/stopwords_es_p0_v1.txt")


def test_soft_hyphen_join_and_deletion() -> None:
    result = clean_natural_text("provin\u00adcia consti\u00ad tucional resto\u00ad")

    assert result.cleaned_text == "provincia constitucional resto"
    assert result.soft_hyphen_removed_count == 3
    assert result.soft_hyphen_join_count == 2
    assert result.changed_by_soft_hyphen_repair is True


def test_explicit_linebreak_hyphenation_repeated_repairs() -> None:
    result = clean_natural_text("banca- rio fundamen- tal- mente extra- ordin- aria")

    assert result.cleaned_text == "bancario fundamentalmente extraordinaria"
    assert result.explicit_hyphenation_join_count == 5
    assert result.changed_by_explicit_hyphenation_repair is True


def test_explicit_hyphenation_preserves_other_hyphen_uses() -> None:
    text = "2024-2025 político-social texto - separado norte -sur"

    assert clean_natural_text(text).cleaned_text == text


def test_nfkc_and_casefold_behavior() -> None:
    assert clean_natural_text("CAFÉ ﬁnal").cleaned_text == "café final"


def test_lexical_tokenizer_preserves_accents_and_excludes_nonlexical_tokens() -> None:
    tokens = lexical_tokens("café nación ñandú 2024 abc123 ab sol x9 salud")

    assert tokens == ["café", "nación", "ñandú", "sol", "salud"]


def test_stopword_variants_are_frozen_and_protect_substantive_terms() -> None:
    p0 = load_stopwords(STOPWORDS_PATH, variant="P0")
    p1 = load_stopwords(STOPWORDS_PATH, variant="P1")

    assert p0.p0_count == 310
    assert p0.p0_sha256 == "b4d338c3aed3e225105bc0b9cdaf3ae775f131e68cfbc6d9a0e61a8152179c3a"
    assert p1.words == p0.words | P1_ADDITIONS
    assert p1.p1_count == p0.p0_count + len(P1_ADDITIONS)
    assert not (PROTECTED_SUBSTANTIVE_TERMS & p0.words)
