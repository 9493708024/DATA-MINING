***************
Folder Details*
***************

customer review data :
----- reviews of 5 products collected from amazon

modified_review_files3 :
----- Segregated reviews in a proper order and saved in csv format

nouns :
----- For each review in customer review data :
--------- For each type of rule we saved the dependencies of each review of that particular product.
--------- The dependencies are obtained by using stanford parser.
--------- Noun phrases are not collected here.

noun_phrases :
----- For each review in customer review data :
--------- For each type of rule we saved the dependencies of each review of that particular product.
--------- The dependencies are obtained by using stanford parser.
--------- Noun phrases are collected here.

opinions :
----- Consists a set of positive and negative seed opinions.


*************
Source Files*
*************

extraction.py :
----- raw reviews from customer review data are segregated and using stanford parser dependencies are obtained for each review and saved in modified_review_files3.
----- Each dependency between two words are delimited by '####' ([nsubj , rahul , name] => nsubj####rahul####name) and set of dependencies are delimited by '######' and saved in modified_review_files3.

aspect_extraction.py :
----- Extracting aspects obtained from each rule and saving the details in 'nouns' and 'noun_phrases' folder.

f1_score.py :
----- From the extracted aspects for each rule, calculating precision and recall.
----- Automating to select the rules which extracts better aspects.


*************
System setup*
*************

----- python nltk , pandas , numpy
----- Download stanford dependency parser for python interface and add path to environment variables.
----- Link for stanford parser download : http://projects.csail.mit.edu/spatial/Stanford_Parser
----- Follow Readme instructions to setup Stanford parser.
