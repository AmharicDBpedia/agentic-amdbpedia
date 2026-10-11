from __future__ import annotations

from mcp_server.pipeline import (
    TemplateField,
    _parse_with_fallback,
    _parse_with_mwparser,
    extract_first_infobox,
    extract_infobox,
    parse_templates,
)
from rag.ontology import AmharicMappingIndex

# A real Amharic Wikipedia-shaped infobox: Dedessa Bridge ("ደደሳ ድልድይ"),
# its length field ("ርዝመት") the stated 16.1 acceptance criterion.
BRIDGE_WIKITEXT = """{{መረጃሳጥን ድልድይ
| ስም = ደደሳ ድልድይ
| ርዝመት = 1,700 ሜትር
| የተከፈተበት_ቀን = 1985
}}"""


def test_extract_infobox_parses_the_bridge_example_length_field() -> None:
    fields = extract_infobox(BRIDGE_WIKITEXT)

    length_field = next(f for f in fields if f.name == "ርዝመት")
    assert length_field.value == "1,700 ሜትር"


def test_extract_infobox_returns_all_fields_in_order() -> None:
    fields = extract_infobox(BRIDGE_WIKITEXT)

    assert fields == [
        TemplateField(name="ስም", value="ደደሳ ድልድይ"),
        TemplateField(name="ርዝመት", value="1,700 ሜትር"),
        TemplateField(name="የተከፈተበት_ቀን", value="1985"),
    ]


def test_extract_infobox_returns_empty_for_a_non_infobox_template() -> None:
    fields = extract_infobox("{{Cite web | url = http://example.com | title = Something}}")

    assert fields == []


def test_extract_infobox_returns_empty_for_plain_text() -> None:
    assert extract_infobox("just some prose, no templates at all") == []


def test_extract_infobox_recognizes_english_infobox_marker() -> None:
    fields = extract_infobox("{{Infobox airport | name = Bole International Airport}}")

    assert fields == [TemplateField(name="name", value="Bole International Airport")]


def test_extract_infobox_ignores_positional_parameters() -> None:
    fields = extract_infobox("{{Infobox bridge | positional_value | ርዝመት = 500}}")

    assert fields == [TemplateField(name="ርዝመት", value="500")]


def test_extract_infobox_picks_the_first_infobox_when_multiple_templates_present() -> None:
    wikitext = "{{Cite web | url = x}} {{Infobox bridge | ርዝመት = 500}}"

    fields = extract_infobox(wikitext)

    assert fields == [TemplateField(name="ርዝመት", value="500")]


def test_extract_first_infobox_misses_a_real_template_with_no_marker_word_by_itself() -> None:
    # Regression: confirmed live against the real corpus that "የቦታ ስም" (the
    # Place template) contains none of INFOBOX_MARKERS ("infobox"/"info
    # box"/"መረጃ"/"ሳጥን") -- the marker heuristic alone silently drops every
    # Place infobox a user pastes.
    wikitext = "{{የቦታ ስም | ስም = ጎንደር | ክልል = አማራ}}"

    assert extract_first_infobox(wikitext) is None


def test_extract_first_infobox_recognizes_a_known_template_the_marker_heuristic_misses() -> None:
    wikitext = "{{የቦታ ስም | ስም = ጎንደር | ክልል = አማራ}}"
    mapping_index = AmharicMappingIndex({}, frozenset({"የቦታ ስም"}))

    template = extract_first_infobox(wikitext, mapping_index=mapping_index)

    assert template is not None
    assert template.name == "የቦታ ስም"
    assert [f.name for f in template.fields] == ["ስም", "ክልል"]


def test_extract_first_infobox_known_template_check_ignores_whitespace_underscore_and_case() -> (
    None
):
    wikitext = "{{የቦታ_ስም | ስም = ጎንደር}}"
    mapping_index = AmharicMappingIndex({}, frozenset({"የቦታ ስም"}))

    template = extract_first_infobox(wikitext, mapping_index=mapping_index)

    assert template is not None
    assert template.name == "የቦታ_ስም"


def test_parse_templates_handles_nested_templates_without_breaking_on_inner_pipes() -> None:
    wikitext = "{{Infobox bridge | ርዝመት = {{convert|500|m}} }}"

    templates = parse_templates(wikitext)

    assert len(templates) == 1
    assert templates[0].fields[0].name == "ርዝመት"
    assert templates[0].fields[0].value == "{{convert|500|m}}"


def test_mwparser_and_fallback_parsers_agree_on_the_bridge_example() -> None:
    """The fallback parser (used when mwparserfromhell isn't installed)
    must produce identical results to the real parser for the stated
    acceptance-criterion example — not just "close enough"."""

    assert _parse_with_mwparser(BRIDGE_WIKITEXT) == _parse_with_fallback(BRIDGE_WIKITEXT)


def test_an_infobox_wrapped_in_html_is_found() -> None:
    # ደብረ ዳሞ wraps its infobox in <center>...</center>.
    wikitext = (
        "'''ደብረ ዳሞ''' ገዳም ነው።\n<center>{{መረጃሳጥን አለት ፍልፍል\n|ስም = ደብረ ዳሞ\n"
        "|ስዕል = Debre Damo.jpg\n}}</center>"
    )

    template = extract_first_infobox(wikitext)

    assert template is not None
    assert [f.name for f in template.fields] == ["ስም", "ስዕል"]


def test_a_template_nested_in_a_parameter_is_not_a_separate_candidate() -> None:
    wikitext = "{{Infobox bridge\n| ስም = {{lang|am|ድልድይ}}\n| ርዝመት = 10\n}}"

    template = extract_first_infobox(wikitext)

    assert template is not None and template.name == "Infobox bridge"


def test_an_infobox_without_a_marker_word_is_found_by_its_shape() -> None:
    # ሜሪላንድ uses {{ስቴት|...}} -- no "መረጃ"/"infobox"/"ሳጥን" in the name.
    wikitext = (
        "{{ስቴት|\nስቴት_ስም = ሜሪላንድ|\nስቴት_ባንዲራ = Flag.svg|\nስቴት_ዋና_ከተማ = አናፖሊስ|\n"
        "ስቴት_ትልቋ_ከተማ = ባልቲሞር|\nስቴት_መሬት_ስፋት = 32,160|\n}}\nሜሪላንድ ...\n"
    )

    template = extract_first_infobox(wikitext)

    assert template is not None and template.name == "ስቴት"
    assert len(template.fields) == 5


def test_citations_with_many_parameters_are_never_taken_for_an_infobox() -> None:
    wikitext = (
        "{{Cite web | url = u | title = t | author = a | date = d | publisher = p"
        " | access-date = x}}"
    )

    assert extract_first_infobox(wikitext) is None
