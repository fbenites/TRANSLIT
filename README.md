# TRANSLIT: A Large Name Transliteration Resource

**TRANSLIT** is A Large Name Transliteration Resource. If you find this code useful in your research, please consider citing:


    @inproceedings{benitesLREC2020,
	Author = {Fernando Benites, Gilbert François Duivesteijn, Pius von Däniken, Mark Cieliebak}
	Title = {Large Name Transliteration Resource},
	booktitle = {Proceedings of the Thirteenth International Conference on Language Resources and Evaluation (LREC 2020)},
	Year = {2020},
    }

## We merged together sources that now encompasses 3 Millions surfaces (names) of around 1.6 Million entities

We merged four data sources:
1. [JRC named entities](https://ec.europa.eu/jrc/en/language-technologies/jrc-names)
2. [Amazon Wiki-Names](https://github.com/steveash/NETransliteration-COLING2018)
3. [Google En-Ar transliterations](https://github.com/google/transliteration)
4. [Geonames](https://download.geonames.org/export/dump/alternateNamesV2.zip)

We also searched for lang tags of wikipedia for transliterations (wiki-all).

We merged multiple names of an entity and assigned a UUID to it. We saved all the gathered names/entities in the file [TRANSLIT.json](https://github.com/fbenites/TRANSLIT/blob/master/artefacts/TRANSLIT.json), in the artefacts directory.

|Dataset       |# entities|# name variations| mean length of chars per name|
|--------------|----------|-----------------|------------------------------|   
|JRC           |819'209   |1'338'463        |14.3                          |
| Geonames       | 139'549    | 758'274           | 10.6                           |
| SubWikiLang    | 609'420    | 1'376'446         | 10.3                           |
| En-Ar          | 15'858     | 31'716            | 4.4                            |
| Wiki-lang-all  | 122'180    | 144'588           | 17.0                           |
| TRANSLIT (all) | 1'655'972  | 3'008'239         | 11.8                           |
     
     
## Experiments

The experiments of the paper can be retraced with the use of the scripts abalation_study.py, classification_experiments.py and cnn_classification.py in the code directory. For their use, the data in artefact is used. To recreate this data, you need to download the original data (17G zipped) with download_data.sh. Afterward you should run run_preprare_data.sh.

## Troubleshooting

the artefacts are quite large, so git lfs needs to be installed:
$ sudo apt install git-lfs
$ git lfs install --local
$ git lfs fetch


     

