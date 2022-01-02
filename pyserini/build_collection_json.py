from pathlib import Path
import json
import argparse
import re

import spacy
import xml.etree.ElementTree as ET


class DocParser:
    def __init__(self, src_dir: Path, dest_dir: Path, nlp="en_core_sci_sm"):
        self.nlp = spacy.load(nlp)
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

    def build_collection_json(self, named_entity: bool = False):
        for doc_path in self.src_dir.iterdir():
            self._build_document_json(doc_path, named_entity)

    def _build_document_json(self, doc_path: Path, named_entity: bool = False):
        doc_id = doc_path.name
        contents = ""
        root = ET.parse(doc_path).getroot()
        for p in root.findall(".//p"):
            # include all contents under p tag recursively
            for text in p.itertext():
                contents += text.strip()
        new_item = {"id": doc_id, "contents": contents}

        # extract named entity
        if named_entity:
            new_item["NER"] = self._get_named_entity(contents)

        # write to json file
        dest_path = (self.dest_dir / doc_id).with_suffix(".json")
        with open(dest_path, "w") as fp:
            json.dump(new_item, fp, ensure_ascii=False)

    def _get_named_entity(self, contents: str):
        doc = self.nlp(contents)
        named_entities = {}
        for ent in doc.ents:
            # skip numbers
            replaced_text = re.sub("[.%,: ±]", "0", ent.text)
            if replaced_text.isnumeric():
                continue
            # append named entity
            named_entities.setdefault(ent.label_, [])
            named_entities[ent.label_].append(ent.text)
        return named_entities


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=Path)
    parser.add_argument("--dest-dir", type=Path)
    parser.add_argument("--named-entity", action="store_true")
    args = parser.parse_args()

    # parse documents
    doc_parser = DocParser(args.src_dir, args.dest_dir)
    doc_parser.build_collection_json(args.named_entity)
