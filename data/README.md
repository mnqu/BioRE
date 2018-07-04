The UMLS Dataset

There are three folders and a pdf file in the folder.

1. raw_data

This folder contains the raw UMLS data. 

There are 4 files, including:
MRCONSO.RRF, which has the mapping between the UMLS id of entities and the entity names;
MRCUI.RRF, which has the concept history;
MRREL.RRF, which contains relation instances between entities;
MRSTY.RRF, which describes entity types.

For the details of each file and the format, users can refer to:
(1) introduction-of-umls.pdf in the main folder. 
The main part of the slides gives an overview of the UMLS dataset, and the appendix introduces the format of each file.
(2) https://www.nlm.nih.gov/research/umls/user_education/learning_resources.html
The link lists some learning resources of the UMLS dataset.
(3) https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/index.html
The link introduces the format of the UMLS dataset.

2. processed_data

This folder contains some processed files.

(1) cui2type.txt
This file contains the mapping from cui (UMLS entity id) to entity types.
Each line has a cui and an entity type, and the format is:
<cui>   <type>

(2) cui2syns.txt
This file contains the mapping from cui (UMLS entity id) to its names and synonyms.
Each line has a cui and a list of names, and the format is:
<cui>::<name1>::<name2>::<name3>

(3) name2cui.txt
This file contains the mapping from entity names to cui (UMLS entity id).
Each line has an entity name and a cui, and the format is:
<name>  <cui>

(4) pair2rlt.txt
This file contains all relation instances.
Each line has two cui and a relation, and the format is:
<cui1>  <cui2>  <relation>

3. umls_cotype

This paper contains the data generation pipeline of CoType for the UMLS dataset.

