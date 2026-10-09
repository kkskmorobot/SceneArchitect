from __future__ import annotations

from scenearchitect.data.caption_curation import _selection_decision, clean_caption, split_tags

SELECTION = {
    "minimum_scene_tags": 2,
    "keep_no_humans": True,
    "scene_terms": ["scenery", "building", "city", "sky", "road"],
    "composition_tags": ["wide shot", "from above"],
    "person_focus_tags": ["close-up", "portrait", "upper body"],
}
CAPTION = {
    "prefix_tags": ["anime scenery", "detailed background"],
    "person_scale_tag": "distant human figure",
    "max_tags": 20,
    "drop_exact": ["solo", "looking at viewer", "dress"],
    "drop_patterns": [
        r"^(?:\d+|multiple)\s*(?:girls?|boys?)$",
        r"(?:^|\s)hair(?:$|\s)",
        r"(?:shirt|dress)",
    ],
}


def test_no_humans_scene_is_kept_and_meta_tags_removed() -> None:
    tags = split_tags("no humans, scenery, city, blue sky, artist name")
    keep, reason, _details = _selection_decision(tags, SELECTION)
    assert keep is True
    assert reason == "no_humans"
    cleaned = clean_caption(tags, {**CAPTION, "drop_exact": ["artist name"]}, has_person=False)
    assert cleaned.startswith("anime scenery, detailed background")
    assert "artist name" not in cleaned


def test_no_humans_wins_over_a_conflicting_person_tag() -> None:
    tags = split_tags("no humans, 1girl, scenery, city")
    keep, reason, details = _selection_decision(tags, SELECTION)
    assert keep is True
    assert reason == "no_humans"
    cleaned = clean_caption(
        tags,
        CAPTION,
        has_person=details["has_person"] and not details["no_humans"],
    )
    assert "distant human figure" not in cleaned


def test_wide_person_scene_is_kept_but_character_traits_are_removed() -> None:
    tags = split_tags("1girl, long hair, white dress, scenery, building, wide shot")
    keep, reason, details = _selection_decision(tags, SELECTION)
    assert keep is True
    assert reason == "person_scene_composition"
    cleaned = clean_caption(tags, CAPTION, has_person=details["has_person"])
    assert "distant human figure" in cleaned
    assert "1girl" not in cleaned
    assert "long hair" not in cleaned
    assert "white dress" not in cleaned
    assert "building" in cleaned
    assert "wide shot" in cleaned


def test_person_closeup_and_person_without_wide_composition_are_rejected() -> None:
    closeup = split_tags("1girl, scenery, building, wide shot, close-up")
    keep, reason, _details = _selection_decision(closeup, SELECTION)
    assert keep is False
    assert reason == "person_focus"

    no_wide = split_tags("1girl, scenery, building, sky")
    keep, reason, _details = _selection_decision(no_wide, SELECTION)
    assert keep is False
    assert reason == "person_without_wide_composition"


def test_character_mode_keeps_safe_people_and_rejects_blocked_or_empty_scenes() -> None:
    character = {
        "mode": "character",
        "blocked_tags": ["nude", "loli"],
        "scene_terms": [],
        "composition_tags": [],
        "person_focus_tags": [],
    }
    keep, reason, _details = _selection_decision(split_tags("1girl, long hair"), character)
    assert keep is True
    assert reason == "person_character"

    keep, reason, _details = _selection_decision(split_tags("1girl, nude"), character)
    assert keep is False
    assert reason == "blocked_content"

    keep, reason, _details = _selection_decision(split_tags("no humans, scenery"), character)
    assert keep is False
    assert reason == "character_no_humans"
