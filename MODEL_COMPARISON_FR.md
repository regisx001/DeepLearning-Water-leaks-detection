# Comparaison des modèles : Deep Learning vs ML classique

Ce rapport compare les trois modèles entraînés sur le même ensemble de test :
- Régression logistique
- Random Forest
- Deep Learning (CNN-BiLSTM-Attention)

## Données source
- data/benchmark_metrics_all_models.csv
- data/final_metrics.csv

## Tableau de benchmark

| Modèle | Seuil | AUC-ROC | AUC-PR | Accuracy | Balanced Accuracy | Précision (Fuite) | Rappel (Fuite) | Score F1 (Fuite) | Taux de fausse alerte | Taux de non-détection | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Régression logistique | 0.3752 | 0.9524 | 0.9595 | 0.9231 | 0.9148 | 0.9405 | 0.8681 | 0.9029 | 0.0385 | 0.1319 | 79 | 5 | 125 | 12 |
| Random Forest | 0.3695 | 0.9710 | 0.9652 | 0.9050 | 0.8912 | 0.9487 | 0.8132 | 0.8757 | 0.0308 | 0.1868 | 74 | 4 | 126 | 17 |
| Deep Learning (CNN-BiLSTM-Attention) | 0.0333 | 0.9412 | 0.9583 | 0.9457 | 0.9390 | 0.9647 | 0.9011 | 0.9318 | 0.0231 | 0.0989 | 82 | 3 | 127 | 9 |

## Principaux résultats

1. Meilleures métriques de classement :
- Random Forest obtient la meilleure AUC-ROC (0.9710) et la meilleure AUC-PR (0.9652).

2. Meilleure performance opérationnelle au seuil sélectionné :
- Le modèle Deep Learning atteint la meilleure Accuracy (0.9457), Balanced Accuracy (0.9390), Précision (0.9647), Rappel (0.9011) et score F1 (0.9318).

3. Compromis sur les erreurs :
- Le Deep Learning présente le plus faible taux de fausse alerte (0.0231) et le plus faible taux de non-détection (0.0989), avec le moins de faux positifs (3) et de faux négatifs (9).

4. Comportement des modèles classiques :
- La régression logistique est plus équilibrée que Random Forest au seuil choisi et reste plus proche du Deep Learning en rappel et en score F1.
- Random Forest produit un meilleur classement probabiliste, mais des performances de classification seuillée plus faibles dans la stratégie de seuil actuelle.

## Conclusion générale

Pour un déploiement orienté détection de fuites (maximiser la détection tout en limitant les fausses alertes), le modèle Deep Learning est l'option la plus solide dans ce benchmark.

Si l'objectif devient uniquement la qualité de classement probabiliste, Random Forest est très compétitif et peut être reconsidéré avec une calibration ou une stratégie de seuil différente.

## Recommandations

- Conserver le modèle Deep Learning comme modèle principal pour la décision en production.
- Conserver la régression logistique comme baseline légère.
- Revoir la calibration du seuil de Random Forest si vous souhaitez exploiter ses fortes métriques AUC dans un autre régime opérationnel.
