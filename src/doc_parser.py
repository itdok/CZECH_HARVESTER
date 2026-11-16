from __future__ import annotations

import re
from typing import Any, Dict, Generator, List, Optional, Set


SENTENCE_DELIMITERS: Set[str] = {".", "!", "?", "…", "\n", ";"}


def is_yellow_highlight(bg_color_dict: Optional[Dict[str, Any]]) -> bool:
    """
    Checks if a backgroundColor dict matches the yellow highlight target.
    Target yellow: RGB ~ (1.0, 1.0, 0.0).
    Handles protobuf JSON serialization where 0.0 values (like blue=0) may be omitted.
    """
    if not isinstance(bg_color_dict, dict):
        return False

    color = bg_color_dict.get("color", bg_color_dict)
    if not isinstance(color, dict):
        return False

    rgb = color.get("rgbColor", color if ("red" in color or "green" in color) else {})
    if not isinstance(rgb, dict) or not rgb:
        return False

    try:
        r = float(rgb.get("red", 0.0))
        g = float(rgb.get("green", 0.0))
        b = float(rgb.get("blue", 0.0))
    except (ValueError, TypeError):
        return False

    # Yellow highlight: high red, high green, significantly low blue
    return r >= 0.75 and g >= 0.75 and b <= 0.4 and (r + g) >= 1.6


def find_sentence_context(paragraph_text: str, start_in_para: int, end_in_para: int) -> str:
    """
    Finds the boundary of the sentence containing [start_in_para:end_in_para]
    within paragraph_text, searching from the nearest preceding delimiter to the next delimiter.
    """
    if not paragraph_text:
        return ""

    total_len = len(paragraph_text)
    start_in_para = max(0, min(start_in_para, total_len))
    end_in_para = max(start_in_para, min(end_in_para, total_len))

    # 1. Search backwards for the preceding sentence delimiter
    sent_start = 0
    for i in range(start_in_para - 1, -1, -1):
        if paragraph_text[i] in SENTENCE_DELIMITERS:
            sent_start = i + 1
            break

    # 2. Search forwards for the next sentence delimiter starting from phrase end
    # (or phrase end - 1 if the phrase itself ends with a delimiter)
    scan_start = max(start_in_para, end_in_para - 1)
    sent_end = total_len
    for j in range(scan_start, total_len):
        if paragraph_text[j] in SENTENCE_DELIMITERS:
            sent_end = j + 1
            # Include consecutive delimiters (e.g. '...', '?!')
            while sent_end < total_len and paragraph_text[sent_end] in SENTENCE_DELIMITERS:
                sent_end += 1
            # Include trailing closing quotes/brackets if present
            while sent_end < total_len and paragraph_text[sent_end] in ')]"\'»”’':
                sent_end += 1
            break

    extracted = paragraph_text[sent_start:sent_end].strip()
    return extracted


def _iter_paragraphs(node: Any) -> Generator[Dict[str, Any], None, None]:
    """
    Recursively traverses a Google Docs AST structure and yields all paragraph dictionaries.
    Handles body content, tables, nested structures, and direct paragraph dictionaries.
    """
    if isinstance(node, dict):
        if "paragraph" in node and isinstance(node["paragraph"], dict):
            yield node["paragraph"]
        elif "elements" in node and isinstance(node["elements"], list):
            yield node
            return
        for key, val in node.items():
            if key != "paragraph" and key != "elements":
                yield from _iter_paragraphs(val)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_paragraphs(item)


def extract_harvest_phrases(doc_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Accepts an AST document from Google Docs API and extracts yellow-highlighted phrases.
    
    Rules:
    - Glues adjacent TextRun elements if they share a yellow background, even if font colors
      differ (e.g. colored vowels).
    - Detects sentence boundaries (from nearest punctuation to next) within the full paragraph text.
    - Returns a list of:
      {"phrase": "...", "context": "...", "start_index": int, "end_index": int}
    """
    results: List[Dict[str, Any]] = []
    running_doc_offset = 0

    for paragraph in _iter_paragraphs(doc_json):
        elements = paragraph.get("elements", [])
        if not elements:
            continue

        # 1. Flatten runs and reconstruct paragraph text
        parsed_runs = []
        para_text_parts = []
        current_para_offset = 0

        for elem in elements:
            text_run = elem.get("textRun")
            if not text_run:
                continue

            content = text_run.get("content", "")
            if not content:
                continue

            start_idx = elem.get("startIndex")
            if start_idx is None:
                start_idx = text_run.get("startIndex", running_doc_offset)

            end_idx = elem.get("endIndex")
            if end_idx is None:
                end_idx = text_run.get("endIndex", start_idx + len(content))

            running_doc_offset = max(running_doc_offset, end_idx)

            style = text_run.get("textStyle", {})
            bg_color = style.get("backgroundColor") if isinstance(style, dict) else None
            is_yellow = is_yellow_highlight(bg_color)

            run_para_start = current_para_offset
            run_para_end = current_para_offset + len(content)
            current_para_offset = run_para_end
            para_text_parts.append(content)

            parsed_runs.append({
                "content": content,
                "is_yellow": is_yellow,
                "start_index": start_idx,
                "end_index": end_idx,
                "para_start": run_para_start,
                "para_end": run_para_end,
            })

        full_para_text = "".join(para_text_parts)

        # 2. Group adjacent yellow runs
        current_yellow_group: List[Dict[str, Any]] = []

        def flush_yellow_group(group: List[Dict[str, Any]]) -> None:
            if not group:
                return

            raw_phrase = "".join(r["content"] for r in group)
            cleaned_phrase = raw_phrase.strip()
            if not cleaned_phrase:
                return

            group_start_idx = group[0]["start_index"]
            group_end_idx = group[-1]["end_index"]
            group_para_start = group[0]["para_start"]
            group_para_end = group[-1]["para_end"]

            context = find_sentence_context(full_para_text, group_para_start, group_para_end)
            if not context:
                context = cleaned_phrase

            results.append({
                "phrase": cleaned_phrase,
                "context": context,
                "start_index": group_start_idx,
                "end_index": group_end_idx,
            })

        for run in parsed_runs:
            if run["is_yellow"]:
                current_yellow_group.append(run)
            else:
                flush_yellow_group(current_yellow_group)
                current_yellow_group = []

        flush_yellow_group(current_yellow_group)

    return results


def generate_clear_highlight_requests(extracted_phrases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates a list of UpdateTextStyleRequest dicts for Google Docs batchUpdate,
    clearing the yellow background on the extracted phrase ranges.
    """
    requests: List[Dict[str, Any]] = []
    seen_ranges = set()

    for item in extracted_phrases:
        start = item.get("start_index")
        end = item.get("end_index")

        if start is None or end is None:
            continue

        try:
            start_val = int(start)
            end_val = int(end)
        except (ValueError, TypeError):
            continue

        if start_val >= end_val:
            continue

        range_key = (start_val, end_val)
        if range_key in seen_ranges:
            continue
        seen_ranges.add(range_key)

        request = {
            "updateTextStyle": {
                "range": {
                    "startIndex": start_val,
                    "endIndex": end_val,
                },
                "textStyle": {},
                "fields": "backgroundColor",
            }
        }
        requests.append(request)

    return requests
