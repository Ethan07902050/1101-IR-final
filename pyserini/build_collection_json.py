from pathlib import Path
import json
import argparse

import spacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
import xml.etree.ElementTree as ET


class DocParser:
    def __init__(self, src_dir: Path, dest_dir: Path, nlp="en_core_sci_sm"):
        self.nlp = spacy.load(nlp)
        self.nlp.add_pipe("abbreviation_detector")
        # self.nlp.add_pipe(
        #     "scispacy_linker",
        #     config={"resolve_abbreviations": True, "linker_name": "umls"},
        # )
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

    def build_collection_json(self, named_entity: bool = False):
        transformed_doc_paths = [p.stem for p in self.dest_dir.glob("*.json")]
        for doc_path in self.src_dir.iterdir():
            if doc_path.name not in transformed_doc_paths:
                self._build_document_json(doc_path, named_entity)

    def _build_document_json(self, doc_path: Path, named_entity: bool = False):
        doc_id = doc_path.name
        contents = ""
        root = ET.parse(doc_path).getroot()
        for p in root.findall(".//p"):
            # include all contents under p tag recursively
            for text in p.itertext():
                contents += text.strip()

        if named_entity:
            named_entities = self._get_named_entity(contents)
            contents = " ".join(named_entities)

        new_item = {"id": doc_id, "contents": contents}

        # write to json file
        dest_path = (self.dest_dir / doc_id).with_suffix(".json")
        with open(dest_path, "w") as fp:
            json.dump(new_item, fp, ensure_ascii=False)

    def _get_named_entity(self, contents: str):
        try:
            named_entities = []
            doc = self.nlp(contents)
            abbrev = {abrv: abrv._.long_form for abrv in doc._.abbreviations}
            for ent in doc.ents:
                named_entities.append(abbrev.get(ent.text) or ent.text)
            return named_entities
        except ValueError:
            # spacy throws ValueError when the contents is too long
            # split the contents into halves
            words = contents.split(" ")
            half = int(len(words) / 2)
            return self._get_named_entity(
                " ".join(words[:half])
            ) + self._get_named_entity(" ".join(words[half:]))


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
