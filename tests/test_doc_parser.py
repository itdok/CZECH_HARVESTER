import unittest
from typing import Any, Dict, List

from src.doc_parser import (
    SENTENCE_DELIMITERS,
    _iter_paragraphs,
    extract_harvest_phrases,
    find_sentence_context,
    generate_clear_highlight_requests,
    is_yellow_highlight,
)


def make_color_dict(red: float = 0.0, green: float = 0.0, blue: float = 0.0, omit_blue: bool = False) -> Dict[str, Any]:
    """Helper to construct Google Docs API backgroundColor dictionary."""
    rgb: Dict[str, Any] = {"red": red, "green": green}
    if not omit_blue:
        rgb["blue"] = blue
    return {"color": {"rgbColor": rgb}}


def make_text_run(
    content: str,
    is_yellow: bool = False,
    fg_color: Dict[str, float] = None,
    start_index: int = None,
    end_index: int = None,
) -> Dict[str, Any]:
    """Helper to construct a textRun element for Google Docs AST."""
    text_style: Dict[str, Any] = {}
    if is_yellow:
        text_style["backgroundColor"] = make_color_dict(1.0, 1.0, 0.0)
    if fg_color:
        text_style["foregroundColor"] = {"color": {"rgbColor": fg_color}}

    run: Dict[str, Any] = {
        "content": content,
        "textStyle": text_style,
    }
    elem: Dict[str, Any] = {"textRun": run}
    if start_index is not None:
        elem["startIndex"] = start_index
    if end_index is not None:
        elem["endIndex"] = end_index
    return elem


def make_doc_ast(paragraphs: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Helper to wrap paragraphs into a Google Docs Document AST."""
    content = []
    for p_elements in paragraphs:
        content.append({
            "paragraph": {
                "elements": p_elements
            }
        })
    return {"body": {"content": content}}


class TestIsYellowHighlight(unittest.TestCase):
    def test_exact_yellow(self):
        bg = make_color_dict(1.0, 1.0, 0.0)
        self.assertTrue(is_yellow_highlight(bg))

    def test_protobuf_omitted_blue(self):
        bg = make_color_dict(1.0, 1.0, omit_blue=True)
        self.assertTrue(is_yellow_highlight(bg))

    def test_yellow_variations(self):
        # Slightly softer yellow
        bg = make_color_dict(0.9, 0.95, 0.1)
        self.assertTrue(is_yellow_highlight(bg))

    def test_non_yellow_colors(self):
        # Red
        self.assertFalse(is_yellow_highlight(make_color_dict(1.0, 0.0, 0.0)))
        # Green
        self.assertFalse(is_yellow_highlight(make_color_dict(0.0, 1.0, 0.0)))
        # Blue
        self.assertFalse(is_yellow_highlight(make_color_dict(0.0, 0.0, 1.0)))
        # White
        self.assertFalse(is_yellow_highlight(make_color_dict(1.0, 1.0, 1.0)))
        # Black
        self.assertFalse(is_yellow_highlight(make_color_dict(0.0, 0.0, 0.0)))
        # Gray
        self.assertFalse(is_yellow_highlight(make_color_dict(0.5, 0.5, 0.5)))
        # Orange (high red, moderate green, no blue) -> green < 0.75
        self.assertFalse(is_yellow_highlight(make_color_dict(1.0, 0.5, 0.0)))

    def test_direct_and_flexible_color_dicts(self):
        # Flattened rgbColor
        self.assertTrue(is_yellow_highlight({"rgbColor": {"red": 1.0, "green": 1.0}}))
        # Direct red/green keys
        self.assertTrue(is_yellow_highlight({"red": 1.0, "green": 1.0, "blue": 0.0}))
        # String representations of numbers
        self.assertTrue(is_yellow_highlight({"color": {"rgbColor": {"red": "1.0", "green": "1.0", "blue": "0.0"}}}))

    def test_invalid_and_empty_inputs(self):
        self.assertFalse(is_yellow_highlight(None))
        self.assertFalse(is_yellow_highlight({}))
        self.assertFalse(is_yellow_highlight({"color": {}}))
        self.assertFalse(is_yellow_highlight({"color": {"rgbColor": {}}}))
        self.assertFalse(is_yellow_highlight({"color": {"rgbColor": {"red": "not_a_number"}}}))
        self.assertFalse(is_yellow_highlight("yellow"))  # type: ignore


class TestFindSentenceContext(unittest.TestCase):
    def test_sentence_in_middle_of_paragraph(self):
        text = "První věta. Druhá věta obsahuje slovo. Třetí věta."
        # "obsahuje slovo" starts at 23, ends at 37
        phrase_start = text.index("obsahuje slovo")
        phrase_end = phrase_start + len("obsahuje slovo")
        context = find_sentence_context(text, phrase_start, phrase_end)
        self.assertEqual(context, "Druhá věta obsahuje slovo.")

    def test_sentence_at_start_of_paragraph(self):
        text = "První věta začíná zde. Druhá věta pokračuje."
        phrase_start = text.index("začíná")
        phrase_end = phrase_start + len("začíná")
        context = find_sentence_context(text, phrase_start, phrase_end)
        self.assertEqual(context, "První věta začíná zde.")

    def test_sentence_at_end_of_paragraph(self):
        text = "První věta. Druhá věta. Třetí věta končí zde."
        phrase_start = text.index("končí")
        phrase_end = phrase_start + len("končí")
        context = find_sentence_context(text, phrase_start, phrase_end)
        self.assertEqual(context, "Třetí věta končí zde.")

    def test_sentence_with_exclamation_mark(self):
        text = "Pozor na psa! Je to nebezpečné."
        phrase_start = text.index("psa")
        phrase_end = phrase_start + len("psa")
        context = find_sentence_context(text, phrase_start, phrase_end)
        self.assertEqual(context, "Pozor na psa!")

    def test_sentence_with_question_mark(self):
        text = "Jak se dnes máš? Doufám, že dobře."
        phrase_start = text.index("dnes")
        phrase_end = phrase_start + len("dnes")
        context = find_sentence_context(text, phrase_start, phrase_end)
        self.assertEqual(context, "Jak se dnes máš?")

    def test_sentence_with_ellipsis_and_multiple_delimiters(self):
        text = "Čekal jsem velmi dlouho... a nic nepřišlo! Opravdu?!"
        phrase_start = text.index("dlouho")
        phrase_end = phrase_start + len("dlouho")
        context = find_sentence_context(text, phrase_start, phrase_end)
        self.assertEqual(context, "Čekal jsem velmi dlouho...")

        phrase2_start = text.index("Opravdu")
        phrase2_end = phrase2_start + len("Opravdu")
        context2 = find_sentence_context(text, phrase2_start, phrase2_end)
        self.assertEqual(context2, "Opravdu?!")

    def test_sentence_with_semicolon(self):
        text = "Všechno bylo připraveno; vyrazili jsme na cestu."
        phrase_start = text.index("připraveno")
        phrase_end = phrase_start + len("připraveno")
        context = find_sentence_context(text, phrase_start, phrase_end)
        self.assertEqual(context, "Všechno bylo připraveno;")

    def test_sentence_with_closing_quotes_and_brackets(self):
        text = 'Učitel řekl: "To je správná odpověď!" Žáci zatleskali.'
        phrase_start = text.index("správná")
        phrase_end = phrase_start + len("správná")
        context = find_sentence_context(text, phrase_start, phrase_end)
        self.assertEqual(context, 'Učitel řekl: "To je správná odpověď!"')

        text2 = "Musíme to udělat (ihned a bez odkladu). Další krok následuje."
        phrase2_start = text2.index("odkladu")
        phrase2_end = phrase2_start + len("odkladu")
        context2 = find_sentence_context(text2, phrase2_start, phrase2_end)
        self.assertEqual(context2, "Musíme to udělat (ihned a bez odkladu).")

    def test_sentence_with_unicode_ellipsis(self):
        text = "Čekám na odpověď… ale nikdo neodpovídá."
        phrase_start = text.index("odpověď")
        phrase_end = phrase_start + len("odpověď")
        context = find_sentence_context(text, phrase_start, phrase_end)
        self.assertEqual(context, "Čekám na odpověď…")

    def test_phrase_containing_internal_semicolon(self):
        text = "První věta. Slovo jedna; slovo dva. Třetí věta."
        phrase_start = text.index("Slovo jedna; slovo dva")
        phrase_end = phrase_start + len("Slovo jedna; slovo dva")
        context = find_sentence_context(text, phrase_start, phrase_end)
        self.assertEqual(context, "Slovo jedna; slovo dva.")

    def test_single_sentence_without_period(self):
        text = "Pouze jedna věta bez jakékoliv tečky"
        phrase_start = text.index("jedna")
        phrase_end = phrase_start + len("jedna")
        context = find_sentence_context(text, phrase_start, phrase_end)
        self.assertEqual(context, "Pouze jedna věta bez jakékoliv tečky")

    def test_empty_and_out_of_bounds_inputs(self):
        self.assertEqual(find_sentence_context("", 0, 0), "")
        self.assertEqual(find_sentence_context("Krátký text.", 100, 200), "")
        # start >= end
        text = "Ahoj světe."
        self.assertEqual(find_sentence_context(text, 5, 5), "Ahoj světe.")


class TestExtractHarvestPhrases(unittest.TestCase):
    def test_multi_colored_vowels_inside_single_word(self):
        """
        Tests stitching adjacent textRuns that form a single Czech word
        where vowels have different font colors (foreground) for pronunciation training,
        but all have yellow background highlight.
        Word: "drahá" -> "dr" (black), "a" (red), "h" (black), "á" (blue).
        """
        red_fg = {"red": 1.0, "green": 0.0, "blue": 0.0}
        blue_fg = {"red": 0.0, "green": 0.0, "blue": 1.0}
        black_fg = {"red": 0.0, "green": 0.0, "blue": 0.0}

        doc = make_doc_ast([[
            make_text_run("To je ", is_yellow=False, start_index=1, end_index=7),
            # Yellow word "drahá" with colored vowels
            make_text_run("dr", is_yellow=True, fg_color=black_fg, start_index=7, end_index=9),
            make_text_run("a", is_yellow=True, fg_color=red_fg, start_index=9, end_index=10),
            make_text_run("h", is_yellow=True, fg_color=black_fg, start_index=10, end_index=11),
            make_text_run("á", is_yellow=True, fg_color=blue_fg, start_index=11, end_index=12),
            make_text_run(" kniha.\n", is_yellow=False, start_index=12, end_index=20),
        ]])

        results = extract_harvest_phrases(doc)
        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item["phrase"], "drahá")
        self.assertEqual(item["context"], "To je drahá kniha.")
        self.assertEqual(item["start_index"], 7)
        self.assertEqual(item["end_index"], 12)

    def test_multi_colored_phrase_across_words(self):
        """
        Tests a multi-word phrase "dobrý den" highlighted yellow,
        where vowels in both words are colored differently from consonants.
        """
        green_fg = {"red": 0.0, "green": 0.8, "blue": 0.0}
        orange_fg = {"red": 1.0, "green": 0.5, "blue": 0.0}

        doc = make_doc_ast([[
            make_text_run("Řekl: ", is_yellow=False, start_index=1, end_index=7),
            # "d", "o" (colored), "br", "ý" (colored), " ", "d", "e" (colored), "n"
            make_text_run("d", is_yellow=True, start_index=7, end_index=8),
            make_text_run("o", is_yellow=True, fg_color=green_fg, start_index=8, end_index=9),
            make_text_run("br", is_yellow=True, start_index=9, end_index=11),
            make_text_run("ý", is_yellow=True, fg_color=orange_fg, start_index=11, end_index=12),
            make_text_run(" ", is_yellow=True, start_index=12, end_index=13),
            make_text_run("d", is_yellow=True, start_index=13, end_index=14),
            make_text_run("e", is_yellow=True, fg_color=green_fg, start_index=14, end_index=15),
            make_text_run("n", is_yellow=True, start_index=15, end_index=16),
            make_text_run(". A odešel.\n", is_yellow=False, start_index=16, end_index=28),
        ]])

        results = extract_harvest_phrases(doc)
        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item["phrase"], "dobrý den")
        self.assertEqual(item["context"], "Řekl: dobrý den.")
        self.assertEqual(item["start_index"], 7)
        self.assertEqual(item["end_index"], 16)

    def test_multiple_separate_highlights_in_same_paragraph(self):
        """
        Tests that two separate yellow phrases separated by a non-yellow run
        are extracted as two distinct items, both sharing the sentence context.
        """
        doc = make_doc_ast([[
            make_text_run("Včera byla ", is_yellow=False, start_index=1, end_index=12),
            make_text_run("zima", is_yellow=True, start_index=12, end_index=16),
            make_text_run(" a dnes je ", is_yellow=False, start_index=16, end_index=27),
            make_text_run("teplo", is_yellow=True, start_index=27, end_index=32),
            make_text_run(".\n", is_yellow=False, start_index=32, end_index=34),
        ]])

        results = extract_harvest_phrases(doc)
        self.assertEqual(len(results), 2)

        self.assertEqual(results[0]["phrase"], "zima")
        self.assertEqual(results[0]["context"], "Včera byla zima a dnes je teplo.")
        self.assertEqual(results[0]["start_index"], 12)
        self.assertEqual(results[0]["end_index"], 16)

        self.assertEqual(results[1]["phrase"], "teplo")
        self.assertEqual(results[1]["context"], "Včera byla zima a dnes je teplo.")
        self.assertEqual(results[1]["start_index"], 27)
        self.assertEqual(results[1]["end_index"], 32)

    def test_multiple_paragraphs_and_sentence_boundaries(self):
        """
        Tests extraction across multiple paragraphs with distinct sentence boundaries.
        """
        doc = make_doc_ast([
            # Paragraph 1
            [
                make_text_run("První věta. ", is_yellow=False, start_index=1, end_index=13),
                make_text_run("Žlutá fráze", is_yellow=True, start_index=13, end_index=24),
                make_text_run(" v první větě.\n", is_yellow=False, start_index=24, end_index=39),
            ],
            # Paragraph 2
            [
                make_text_run("Druhá věta má také ", is_yellow=False, start_index=39, end_index=58),
                make_text_run("zvláštní", is_yellow=True, start_index=58, end_index=66),
                make_text_run(" slovo!\n", is_yellow=False, start_index=66, end_index=75),
            ],
        ])

        results = extract_harvest_phrases(doc)
        self.assertEqual(len(results), 2)

        self.assertEqual(results[0]["phrase"], "Žlutá fráze")
        self.assertEqual(results[0]["context"], "Žlutá fráze v první větě.")
        self.assertEqual(results[0]["start_index"], 13)
        self.assertEqual(results[0]["end_index"], 24)

        self.assertEqual(results[1]["phrase"], "zvláštní")
        self.assertEqual(results[1]["context"], "Druhá věta má také zvláštní slovo!")
        self.assertEqual(results[1]["start_index"], 58)
        self.assertEqual(results[1]["end_index"], 66)

    def test_nested_table_ast(self):
        """
        Tests that paragraphs nested inside Google Docs table structures are properly extracted.
        """
        table_ast = {
            "body": {
                "content": [
                    {
                        "table": {
                            "tableRows": [
                                {
                                    "tableCells": [
                                        {
                                            "content": [
                                                {
                                                    "paragraph": {
                                                        "elements": [
                                                            make_text_run("Slovo v tabulce je ", is_yellow=False, start_index=1, end_index=20),
                                                            make_text_run("důležité", is_yellow=True, start_index=20, end_index=28),
                                                            make_text_run(".\n", is_yellow=False, start_index=28, end_index=30),
                                                        ]
                                                    }
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }
        }

        results = extract_harvest_phrases(table_ast)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["phrase"], "důležité")
        self.assertEqual(results[0]["context"], "Slovo v tabulce je důležité.")
        self.assertEqual(results[0]["start_index"], 20)
        self.assertEqual(results[0]["end_index"], 28)

    def test_yellow_whitespace_only_ignored(self):
        """
        Tests that runs that are highlighted yellow but contain only whitespace are skipped.
        """
        doc = make_doc_ast([[
            make_text_run("Před tímto ", is_yellow=False, start_index=1, end_index=12),
            make_text_run("   ", is_yellow=True, start_index=12, end_index=15),
            make_text_run("je text.\n", is_yellow=False, start_index=15, end_index=24),
        ]])

        results = extract_harvest_phrases(doc)
        self.assertEqual(len(results), 0)

    def test_direct_paragraph_ast_without_body_wrapper(self):
        """
        Tests parsing when input is directly a paragraph object containing elements.
        """
        paragraph = {
            "elements": [
                make_text_run("Příklad ", is_yellow=False, start_index=0, end_index=8),
                make_text_run("přímého", is_yellow=True, start_index=8, end_index=15),
                make_text_run(" objektu.\n", is_yellow=False, start_index=15, end_index=25),
            ]
        }
        results = extract_harvest_phrases(paragraph)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["phrase"], "přímého")
        self.assertEqual(results[0]["context"], "Příklad přímého objektu.")

    def test_phrase_with_surrounding_whitespace_inside_highlight(self):
        """
        Tests that when a yellow highlight includes surrounding spaces,
        the phrase is stripped cleanly while preserving proper sentence context.
        """
        doc = make_doc_ast([[
            make_text_run("Začátek věty. ", is_yellow=False, start_index=0, end_index=14),
            make_text_run("  zvýrazněné slovo  ", is_yellow=True, start_index=14, end_index=34),
            make_text_run(" v kontextu.\n", is_yellow=False, start_index=34, end_index=47),
        ]])
        results = extract_harvest_phrases(doc)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["phrase"], "zvýrazněné slovo")
        self.assertEqual(results[0]["context"], "zvýrazněné slovo   v kontextu.")

    def test_automatic_offset_calculation_when_indices_omitted(self):
        """
        Tests that if startIndex and endIndex are not present in the AST,
        they are inferred sequentially starting from 0.
        """
        doc = make_doc_ast([[
            make_text_run("Začátek ", is_yellow=False),  # len 8 -> [0, 8]
            make_text_run("klíčové ", is_yellow=True),   # len 8 -> [8, 16]
            make_text_run("slovo", is_yellow=True),      # len 5 -> [16, 21]
            make_text_run(" konec.\n", is_yellow=False), # len 8 -> [21, 29]
        ]])

        results = extract_harvest_phrases(doc)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["phrase"], "klíčové slovo")
        self.assertEqual(results[0]["start_index"], 8)
        self.assertEqual(results[0]["end_index"], 21)


class TestGenerateClearHighlightRequests(unittest.TestCase):
    def test_basic_request_generation(self):
        phrases = [
            {"phrase": "ahoj", "context": "Ahoj světe.", "start_index": 5, "end_index": 9},
            {"phrase": "světe", "context": "Ahoj světe.", "start_index": 10, "end_index": 15},
        ]
        requests = generate_clear_highlight_requests(phrases)
        self.assertEqual(len(requests), 2)

        self.assertEqual(requests[0], {
            "updateTextStyle": {
                "range": {
                    "startIndex": 5,
                    "endIndex": 9,
                },
                "textStyle": {},
                "fields": "backgroundColor",
            }
        })
        self.assertEqual(requests[1], {
            "updateTextStyle": {
                "range": {
                    "startIndex": 10,
                    "endIndex": 15,
                },
                "textStyle": {},
                "fields": "backgroundColor",
            }
        })

    def test_deduplication_of_identical_ranges(self):
        phrases = [
            {"phrase": "test", "context": "Test text.", "start_index": 0, "end_index": 4},
            {"phrase": "test", "context": "Test text.", "start_index": 0, "end_index": 4},
        ]
        requests = generate_clear_highlight_requests(phrases)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["updateTextStyle"]["range"]["startIndex"], 0)
        self.assertEqual(requests[0]["updateTextStyle"]["range"]["endIndex"], 4)

    def test_skips_invalid_ranges(self):
        phrases = [
            # Inverted range
            {"phrase": "bad", "context": "bad", "start_index": 10, "end_index": 5},
            # Equal range
            {"phrase": "empty", "context": "empty", "start_index": 5, "end_index": 5},
            # Missing start_index
            {"phrase": "no_start", "context": "no_start", "end_index": 5},
            # Missing end_index
            {"phrase": "no_end", "context": "no_end", "start_index": 5},
            # Non-integer
            {"phrase": "str_index", "context": "str_index", "start_index": "abc", "end_index": 10},
            # Valid item
            {"phrase": "good", "context": "good", "start_index": 1, "end_index": 5},
        ]
        requests = generate_clear_highlight_requests(phrases)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["updateTextStyle"]["range"]["startIndex"], 1)
        self.assertEqual(requests[0]["updateTextStyle"]["range"]["endIndex"], 5)

    def test_empty_list(self):
        self.assertEqual(generate_clear_highlight_requests([]), [])


if __name__ == "__main__":
    unittest.main()
