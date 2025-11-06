# Annotation Project

This project aims to develop a machine-assisted method for extracting and tracking narratives and actor–action–object structures from strategic documents.

## Project structure
annotation_project/
├── data/
    ├── annotated
        ├── entities
            ├── 1_NS_Concept_1998_ENG_entities.json
            ├── 2_NS_Strategy_2011_ENG_entities.json
            ├── 3_Updated_NS_Strategy_2018_ENG_entities.json
        ├── relations
        ├── scenarios
    ├── clean_texts
        ├── text_1.txt
        ├── text_2.txt
        ├── text_3.txt
    ├── raw_texts
        ├── text_1.txt
        ├── text_2.txt
        ├── text_3.txt
    ├── results
    ├── samples_for_annotation
├── scripts/
    ├── clean_texts.py
├── modules_extractors/
    ├── __init__.py
    ├── entity_extractor.py #rule based
    ├── relations_extractor.py
    ├── schema_loader.py
    ├── entity_frequency_extractor.py #statistic + ML-based
├── schema
    ├── entity_types.json
    ├── relation_types.json
    ├── replacements.json
├── tests
    ├── test_clean_texts.py
├── utils
    ├── file_manager.py
├── main.py
├── config.py
├── README.md
└── requirements.txt


## Usage
Run the main script:
python main.py