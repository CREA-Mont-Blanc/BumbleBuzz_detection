# Documentation Technique

## Vue d'ensemble

Ce système permet l'analyse et l'évaluation de détections acoustiques automatisées, avec support pour l'optimisation de seuils, l'analyse d'erreurs par classe, et l'évaluation à différentes échelles (locale et globale).

## Configuration de l'environnement

### Création de l'environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Activation de l'environnement virtuel

```bash
source .venv/bin/activate
```


## Taxonomie des classes

### Classes primaires (TaggingCategory.csv)

- **fly_housefly** : Mouches, mouches domestiques
- **bee_wasp** : Abeilles, guêpes
- **water** : Sons d'eau (ruisseau, cascade, pluie)
- **wind** : Sons de vent (feuilles, bruit de micro)
- **motor_vehicle** : Véhicules motorisés (voiture, klaxon, freinage)
- **aircraft** : Aéronefs (avion, hélicoptère, moteur à réaction)
- **human_voice** : Voix humaine (parole, chant, rire, pleurs)

### Meta-classes (configuration)

- **Group_buzz** : Regroupement des sons d'insectes bourdonnants
- **Group_geophony** : Sons naturels non-biologiques (eau, vent)
- **Group_anthropophony** : Sons d'origine humaine (véhicules, voix)

## Exécution des scripts
### Détection par lots (01_batchbuzz_detection.sh)

```bash
./01_batchbuzz_detection.sh
``` 
Ce script traite tous les sous-répertoires dans le répertoire spécifié, effectuant la détection acoustique sur chaque ensemble de données, place les découpages audio dans un sous-dossier `data_cut` et sauvegarde les résultats dans un fichier CSV.

### Évaluation par lots (02_batchbuzz_evaluation.sh)

```bash
./02_batchbuzz_evaluation.sh
```
Ce script évalue les résultats de la détection en comparant les fichiers CSV générés avec les annotations de référence, et sauvegarde les résultats d'évaluation dans un fichier CSV. Les fichiers de prédiction sont déplacés dans un sous-dossier `predictions` et les résultats d'évaluation dans un sous-dossier `evaluation`.


## How to use directly process.py and dash_app.py

First, create a directory with wav audio files named YYMMDD_HHMMSS.wav/flac or What_EVER_YYMMDD_HHMMSS.wav/flac.
Then, use process.py to perform buzz detection (+many other audio tagging).
Finally, use the dash_app.py script to run the web application.

![plot](image.png)

## Audio processing

```
python process.py [-h] [--data_path DATA_PATH] [--save_path SAVE_PATH] [--name NAME] [--audio_format AUDIO_FORMAT] [--l LENGTH_AUDIO_SEGMENT] [--save_audio_flac SAVE_AUDIO_FLAC] 

Script to process sound files recorded by Audiomoth

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to a folder with wav / flac files
  --save_path SAVE_PATH
                        Path to save outputs and audio files. folder will be created if does not exist
  --name NAME           name of measurement - you can put whatever you want, for example the name of the site 
  --audio_format AUDIO_FORMAT
                        wav or flac
  --l LENGTH_AUDIO_SEGMENT
                        Window length in seconds, must be larger than 5
  --save_audio_flac SAVE_AUDIO_FLAC
                        Saving audio in flac format (needed to run visualization tool)

```
There are other options (such as the type of pretrained model to use), please check the source code if necessary.
### Example

```
python3 process.py --save_path example/metadata/ --data_path example/metadata/audio_0002/ --name 0004 --audio_format flac --l 5 
```

Tagging of all flac files in folder example/metadata/audio_002 in non-overlapping chunks of 5 seconds.

Each chunk is converted to flac and saved.

## Dash app

```
python dash_app.py [-h] [--save_path SAVE_PATH]

Script to display sound files recorded by Audiomoth

options:
  -h, --help            show this help message and exit
  --save_path SAVE_PATH
                        Path to folder output of process.py : has to be the one that was entered when launching process.py
  --name NAME           name of measurement : has to be the one that was entered when launching process.py
```

Code for Audioset Tagging CNN from [Qiu Qiang Kong](https://github.com/qiuqiangkong/audioset_tagging_cnn)
