# XEmines_DataScience_2020

Cours de Data Science Septembre 2019 - UM6P - Emines


## Semaine 1 : De la régression linéaire aux modèles linéaires généralisés.

### Jour 1 : Introduction, data science et régression linéaire

* Présentation du programme, du travail attendu
* Mise en place de l'environnement de travail
    * Jupyter notebook
    * Librairies python, scikit-learn, numpy, pandas     * Google Colab
* Conférence introductive : Qu'est-ce que la data science ?

* Data Science, Apprentissage Machine, Intelligence Artificielle, Deep Learning
* Les différents types d’apprentissage :
    * approche supervisée ou non-supervisée,
    * Régression ou classification.
* Rappels de statistique
    * Modélisation statistique, distribution, construction d’estimateurs
    * Corrélation : Spearman et Pearson
* Librairie statsmodel
* Manipulation des données en python avec les dataframes pandas

### Jour 2 : Tests statistiques
* Bases des tests d’hypothèses
* Lois gaussiennes et distributions associées (Student, chi-carré)
* Niveau d’un test, p-valeur
* Quelques tests classiques : test de Student, test de Fisher, test à deux échantillons
* Introduction aux tests non-paramétriques : tests de d’adéquation de loi (Kolmogorov-
Smirnov), tests d’indépendance (test de rang)
* Comment interpréter correctement un test d’hypothèses ?

### Jour 3 : Régression linéaire univariée
* Régression linéaire univariée
* La Méthode des Moindres Carrés Ordinaires (OLS)
* Interprétation géométrique.
* Métriques de résultat : R^2, coefficients, p-valeur,
* Test d’hypothèses pour le modèle de régression.
* Diagnostics graphiques : résiduels, QQ plot,...
* Analyse de la variance

### Jour 4 : Régression linéaire multivariée
* Régression linéaire multivariée : interprétation géométrique
* Loi Gaussienne multivariée
* Interprétation géométrique
* Théorème de Gauss-Markov
* Liens avec la méthode du maximum de vraisemblance dans le cas d’un modèle gaussien
* Quelques difficultés usuelles : multi-colinéarité des régresseurs, heteroscédasticité des erreurs : caractérisation, détection, stratégies de remédiation
* Interprétation des métriques de résultat (suite): log-vraisemblance, critères d’information d’Akaike (AIC), critère d’information Bayésien BIC
* p-hacking
* Variables quantitatives: one-hot encoding, malédiction de la dimension
* Python
    * Visualisation des données, principales librairies: matplotlib, plot.ly, seaborn

### Jour 5 : Classification et régression logistique
* Régression logistique
* Méthode du maximum de vraisemblance.
* Quelques interprétations numériques sur les méthodes de vraisemblance
* Tests d’hypothèses pour le modèle de régression logistique
* Métriques de classification : matrice de confusion, AUC, F1
* Python
* Régression logistique en pratique avec scikit-learn vs statsmodel

## Semaine 2- Arbres de classification, méthodes d’ensembles et Boosting

### Jour 6 : Sous-apprentissage / Sur-apprentissage
* Introduction à Scikit-learn
* Régression polynomiale
* La notion de sous-apprentissage et de sur-apprentissage (sur-apprentissage, sous-
apprentissage)
* Le compromis biais-variance
* Estimation sans biais du risque

### Jour 7 : Validation croisée et régularisation
* Validation croisée et découpage apprentissage, test et validation     * Sur-apprentissage, détection, solutions
    * Courbes d’apprentissage : détecter et corriger le sur-apprentissage     * La régression régularisée
    * Régularisation L1 et L2 : régression Ridge & Lasso
* Identification et traitement des données aberrantes

### Jour 8 : Arbres et Forêts
* Forêts aléatoires
    * Arbre de décisions et forêts aléatoires
    * Approfondissement sur les métriques de classification : AUC, F1, ...
* Comment traiter des données déséquilibrées en classification     * Paradoxe de la précision
    * Boostrapping
    * Sous et sur-échantillonnage
    * SMOTE

### Jour 9 : Méthodes ensemblistes, gradient stochastique
* Agrégation de classificateurs et de régresseurs : Bagging     * Weak learner
    * Régression et classification
* Gradient stochastique
    * Principe et bases mathématiques des algorithmes de descente de gradient     * Du gradient déterministe au gradient stochastique
    * Choix du pas d’apprentissage, algorithmes d’ajustement du pas.
    * Applications, visualisation et diagnostics de convergence

### Jour 10 : Boosting et XGBoost
* Boosting
    * Principe
    * XGBoost et LightGBM: tuning et application
* Biais dans les modèles « boites noires ».
* Interprétabilité des modèles black-box avec SHAP
