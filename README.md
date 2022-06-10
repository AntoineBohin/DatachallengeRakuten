# DatachallengeRakuten

Ce projet est réalisé dans le cadre de l'enseignement d'intégration `DataChallengeRakuten` de la `ST4 Data@Web` de CentraleSupélec et mené par Antoine Bohin, Logan Renaud, Nicolas des Cognets, Victor Garcia et Clément Ribe. 

## Organisation de repository 

Ce repository est organisé de manière classique en séparant les fichiers qui contiennet les données utilisées par nos algorithmes (entrainement et test) que l'on retrouve dans le dossier `dataset` ainsi que les fichiers python qui contiennent toutes les fonctions utiles à l'exécution du code et la réalisation du projet en lui même dans le dossier `src`. 

**dataset**
Contient l'ensemble des données utilisées par nos algorithmes autant pour le training que pour les test, ces données sont séparés dans 3 sous-dossiers:  
    • `baseData` qui contient les données non traitées comme données dans le cadre du challenge.  
    • `processed` qui contient les textes après passage des scripts de traitement de texte avec certaines des descriptions qui manquent.   
    • `processedWIthDescription` qui contient les données utilisées finalement pour l'entrainement de l'algorithme de machine learning avec des descriptions pour chaque produit.  

**src**
Contient l'ensemble des scripts python ainsi que les modèles de machine learning enregistrés et sont séparés en plusieurs catégories:  
    • `mlModelsText` contient les scripts qui servent à entrainer les modèles de machine learning que l'on a utilisé ou testé (SVM, multinomialNB, réseau de neurones personnalisé).  
    • `preprocessing` contient les scripts qui servent au preprocessing de texte mais aussi d'image dans le sous-dossier `imagePreprocessing`
    • `models` contient les modèles svm entrainés sur les dataset de training.  
    • `utils` contient d'autres scripts de traitement de l'information comme `statistiques.py` qui permet d'accéder à des informations numériques relatives à la composition du dataset.

**app.py**
C'est le fichier principal de ce projet qui permet de lancer, en l'appelant en ligne de commande, dans un premier temps l'évaluation d'un modèle et dans un second temps de faire une prédiction pour la catégorie pour un produit inconnu. 

## Exécuter le Code 

Pour exécuter le code, vous avez le choix entre deux possibilités.

### Evaluation d'un modèle sur une base de données annotées

Si vous voulez prédire les classes d'un ensemble de produits (dont les informations sont réunies dans un fichier .csv dont les colonnes sont "IntegerID", "Titre", "Description", "ProductID", "ImageID"), vous pouvez exécuter la ligne de code suivante dans votre terminal. Le paramètre --csv-to-test correspond au path de votre .csv test, --path-root au dossier dans lequel vous souhaitez stocker les .csv et matrices pré-traitées utilisées par les algorithmes, et --model le modèle que vous souhaitez utiliser (entre 'multinomialnb', 'customnn', 'svm', 'logisticregression').

```bash
python3 main_prediction.py --csv-to-test 'dataset/baseData/X_test_update.csv' --path-root 'dataset/test/' --model 'multinomialnb'
```

Le resultat est stocké dans un fichier .csv nommé prediction_without_labels_{nommodèle} dans le dossier output. Il contient les colonnes "IntegerID" et "ProductTypeCode", pour la classe de produit prédite.

### Prédiction des classes d'un ensemble de produits avec un certain modèle

Si vous voulez évaluer un des modèles implémentés en utilisant un ensemble de produits (dont les informations sont réunies dans un fichier .csv X_test dont les colonnes sont "IntegerID", "Titre", "Description", "ProductID", "ImageID") et leurs classes de produit associées (dont les informations sont réunies dans un fichier .csv Y_test dont les colonnes sont "IntegerID", "ProductTypeCode"), vous pouvez exécuter la ligne de code suivante dans votre terminal. Le paramètre --csv-X-test correspond au path de votre .csv X_test, --csv-Y-test correspond au path de votre .csv Y_test, --path-root au dossier dans lequel vous souhaitez stocker les .csv et matrices pré-traitées utilisées par les algorithmes, et --model le modèle que vous souhaitez utiliser (entre 'multinomialnb', 'customnn', 'svm', 'logisticregression').

```bash
python3 main_evaluation.py --csv-X-test 'dataset/baseData/X_test_update.csv' --csv-Y-test 'dataset/baseData/Y_test_update.csv' --path-root 'dataset/test/' --model 'multinomialnb'
```

e resultat est stocké dans un fichier .csv nommé prediction_{nommodèle} dans le dossier output. Il contient les colonnes "IntegerID","CodePredictions" et " RealProductTypeCodes".

## Le preprocessing des données d'entrée

**Données Textuelles**  

Pour pouvoir utitliser les données textuelles d'entrée (titres, descriptions des produits), il faut s'assurer que toutes les descriptions présente des termes normalisés (lemmes), et que tous les produits présentent une description. 

Pour répondre à ces impératifs, il s'agit alors de procéder à un pré-processing des textes:    
    • suppression des balises html dans les titres et les descriptions.   
    • tokenisation du texte.  
    • retrait des stop words (mots vides de sens car trop utilisés dans la langue).  
    • lemmatisation.  
    • concaténation des lemmes pour reformer les descriptions et les titres.  

Une fois les descriptions existantes traitées, il s'agit alors de donner une description aux produits qui n'en ont pas : pour ce faire on choisit au hasard une description parmi les descriptions des produits de la même catégorie et on lui donne cette description comme si elle était relative au produit considéré.

**Images**

Pour pouvoir espérer traiter des images par ordinateur on doit lui donner des images normalisées et on doit pour ce faire lui appliquer plusieurs scripts pour passer d'une base d'images disparates à une collection utilisable pour entrainer une intelligence artificielle. On doit alors:   
    • resize des images vers 500*500 pixels  
    • passage d'images en couleur à des images en noir et blanc  
    • effaçage du bruit sur les images en appliquant un filtre sur les images

## Les modèles de machine Learning utilisés pour la classification de texte 

## La classification par reconnaissance d'images 

## Requirements 
