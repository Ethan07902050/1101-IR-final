from pathlib import Path
import json

import xml.etree.ElementTree as ET


def build_collection_json(src_dir: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)

    for doc_path in src_dir.iterdir():
        doc_id = doc_path.name
        # extract contents
        contents = ""
        root = ET.parse(doc_path).getroot()
        for p in root.findall(".//p"):
            # include all contents under p tag recursively
            for text in p.itertext():
                contents += text.strip()

        # write to json file
        new_item = {"id": doc_id, "contents": contents}
        dest_path = (dest_dir / doc_id).with_suffix(".json")
        with open(dest_path, "w") as fp:
            json.dump(new_item, fp, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    data_path = Path("data")
    doc_dir = data_path / "doc"
    json_dir = data_path / "doc_json"
    build_collection_json(doc_dir, json_dir)
