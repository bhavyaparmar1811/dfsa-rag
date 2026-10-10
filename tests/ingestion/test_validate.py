from src.ingestion.validate import sequence_gaps


def test_sequence_gaps_within_a_section():
    assert sequence_gaps(["3.3.1", "3.3.2", "3.3.4", "3.4.1"]) == ["3.3.3"]


def test_lettered_insertions_and_new_sections_are_not_gaps():
    assert sequence_gaps(["3.3.2", "3.3.2A", "3.3.3", "3.4.1", "4.1.1"]) == []
