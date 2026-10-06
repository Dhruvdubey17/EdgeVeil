from pathlib import Path
from xml.etree import ElementTree

from edgeveil.spans import EntityType

CONFIG_PATH = Path(__file__).parent.parent / "docs" / "label_studio.xml"


def test_config_offers_exactly_the_entity_types_in_enum_order() -> None:
    root = ElementTree.parse(CONFIG_PATH).getroot()
    labels = [label.get("value") for label in root.iter("Label")]
    assert labels == [member.value for member in EntityType]


def test_labels_apply_to_the_text_field() -> None:
    root = ElementTree.parse(CONFIG_PATH).getroot()
    labels = root.find("Labels")
    text = root.find("Text")
    assert labels is not None
    assert text is not None
    assert labels.get("toName") == text.get("name")
    assert text.get("value") == "$text"
