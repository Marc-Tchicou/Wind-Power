# Wind Power Generation Prediction

## Présentation

Ce projet implémente un pipeline complet de prédiction de la **puissance générée par une éolienne**
à partir de données météorologiques SCADA. Trois modèles sont comparés : une régression linéaire
(baseline), un Random Forest et un réseau de neurones MLP (PyTorch). Une analyse d'interprétabilité
via **SHAP values** est également réalisée.

Projet réalisé dans le cadre du cours **DATASCI 3ML3 — Introduction to Neural Networks**
à McMaster University (Avril 2026).

---

## Données

- **Source** : [Wind Turbine SCADA Dataset — Kaggle](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)
- **Taille** : 50 530 observations à intervalles de 10 minutes sur l'année 2018
- **Variables originales** :

| Variable | Description |
|---|---|
| `LV ActivePower (kW)` | Puissance électrique réelle produite ← **variable cible** |
| `Wind Speed (m/s)` | Vitesse du vent |
| `Wind Direction (°)` | Direction du vent |
| `Theoretical_Power_Curve (KWh)` | Courbe de puissance théorique du fabricant |

- **Split** : 80% train / 20% test — **split temporel** (pas aléatoire)

---

## Méthodologie

### Étape 1 — Analyse exploratoire
- Visualisation de la production et de la vitesse du vent sur l'année 2018
- Courbe de puissance empirique (relation non linéaire en S)
- Matrice de corrélation et distributions des variables

### Étape 2 — Feature Engineering
20 variables explicatives construites à partir des 4 variables originales :

| Type | Features |
|---|---|
| **Physiques** | `wind_speed_sq`, `wind_speed_cube` (loi de Betz) |
| **Temporelles** | `hour`, `month`, `weekday`, `season` |
| **Lag** | vitesse et puissance aux instants t-1, t-2, t-4, t-6 |
| **Rolling** | moyenne et écart-type glissants (6h et 24h) |
| **Cyclique** | `wind_dir_sin`, `wind_dir_cos` (encodage de la direction) |

### Étape 3 — Prétraitement
- Valeurs manquantes : interpolation linéaire + forward/backward fill
- Standardisation via `StandardScaler` (fit sur le train uniquement pour éviter le data leakage)
- Normalisation séparée de la variable cible

### Étape 4 — Modèles

**Baseline 1 : Régression Linéaire**
Modèle linéaire enrichi par les features polynomiales en vitesse du vent.

**Baseline 2 : Random Forest** (100 arbres)
Modèle non linéaire robuste, bien adapté aux données tabulaires.

**Modèle principal : MLP (PyTorch)**

Input(20) → Dense(256) → BatchNorm → ReLU → Dropout(0.2)
→ Dense(128) → BatchNorm → ReLU → Dropout(0.2)
→ Dense(64)  → BatchNorm → ReLU → Dropout(0.1)
→ Dense(32)  → ReLU
→ Dense(1)

- Optimiseur : Adam (lr=3×10⁻⁴, weight_decay=10⁻⁴)
- Scheduler : ReduceLROnPlateau
- Early stopping (patience=20), batch size=256, max 300 epochs

### Étape 5 — Interprétabilité
Analyse des **SHAP values** (KernelExplainer) sur 200 observations de test pour identifier
les features les plus importantes dans les prédictions du MLP.

---

## Résultats

| Modèle | RMSE (kW) | MAE (kW) | R² |
|---|---|---|---|
| Régression Linéaire | 0.510 | 0.325 | 0.986 |
| **Random Forest** | **0.464** | **0.259** | **0.988** |
| MLP (PyTorch) | 0.798 | 0.555 | 0.965 |

Les trois modèles obtiennent un R² supérieur à 0.96. Le **Random Forest est le meilleur modèle**
grâce à sa capacité à capturer les interactions non linéaires sur des données tabulaires structurées.
La Régression Linéaire performe remarquablement bien grâce aux features polynomiales inspirées
de la physique.

**SHAP — Features les plus importantes :**
1. `wind_speed` — driver principal (cohérent avec la physique)
2. `power_lag_1` — dépendance temporelle à court terme
3. `wind_speed_sq` et `wind_speed_cube`
4. `month` et `season` — structure saisonnière

---

## Stack technique

| Outil | Usage |
|---|---|
| `PyTorch` | Architecture MLP, entraînement |
| `scikit-learn` | Régression linéaire, Random Forest, normalisation |
| `shap` | Interprétabilité du modèle |
| `pandas` / `numpy` | Manipulation des données |
| `matplotlib` / `seaborn` | Visualisations |

---

## Auteur

**Marc-Antony TCHICOU**
Étudiant ingénieur @ ENSIIE — Double diplôme Mathématiques Appliquées (Paris-Saclay)
Étudiant en échange @ McMaster University
tchicouantony@gmail.com
