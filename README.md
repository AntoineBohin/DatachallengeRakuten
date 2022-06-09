# DatachallengeRakuten

Ce projet est réalisé dans le cadre de l'enseignement d'intégration `DataChallengeRakuten` de la `ST4 Data@Web` de CentraleSupélec et mené par Antoine Bohin, Logan Renaud, Nicolas des Cognets, Victor Garcia et Clément Ribe. 

## Organisation de repository 

Ce repository est organisé de manière classique en séparant les fichiers qui contiennet les données utilisées par nos algorithmes (entrainement et test) que l'on retrouve dans le dossier `dataset` ainsi que les fichiers python qui contiennent toutes les fonctions utiles à l'exécution du code et la réalisation du projet en lui même dans le dossier `src`. 

## Exécuter le Code 

Pour exécuter le code, il suffit de rentrer dans le terminal la commande suivante : 

```bash
python3  -m app.py
```

Cette ligne de commande permet de lancer le fichier `app.py` 

## Le preprocessing des données d'entrée

**Données Textuelles**  

Pour pouvoir utitliser les données textuelles d'entrée (titres, descriptions des produits), il faut s'assurer que toutes les descriptions présente des termes normalisés (lemmes), et que tous les produits présentent une description. 

Pour répondre à ces impératifs, il s'agit alors de procéder à un pré-processing des textes :  
    • suppression des balises html dans les titres et les descriptions. 
    • tokenisation du texte. 
    • retrait des stop words (mots vides de sens car trop utilisés dans la langue).
    • lemmatisation. 
    • concaténation des lemmes pour reformer les descriptions et les titres.  

Une fois les descriptions existantes traitées, il s'agit alors de donner une description aux produits qui n'en ont pas : pour ce faire on choisit au hasard une description parmi les descriptions des produits de la même catégorie et on lui donne cette description comme si elle était relative au produit considéré.

**Images**

Pour pouvoir espérer traiter des images par ordinateur on doit lui donner des images normalisées et on doit pour ce faire lui appliquer plusieurs scripts pour passer d'une base d'images disparates à une collection utilisable pour entrainer une intelligence artificielle. On doit alors : 
    • resize des images vers 500*500 pixels 
    • passage d'images en couleur à des images en noir et blanc
    • effaçage du bruit sur les images en appliquant un filtre sur les images 

## Les modèles de machine Learning utilisés pour la classification de texte 

## La classification par reconnaissance d'images 

## Requirements 