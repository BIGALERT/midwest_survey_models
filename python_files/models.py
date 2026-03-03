# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv (3.11.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Midwest Survey — Model Comparison
#
# Predict the census region from survey responses using three pipelines:
# Logistic Regression, Random Forest, and Gradient Boosting.
# Each uses `skrub.TableVectorizer` for automatic feature encoding.
# Training is done on a shuffled sample of 1,000 rows.

# %%
import joblib
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from midwest_survey_models.transformers import NumericalStabilizer

# %% [markdown]
# ## Data

# %%
bunch = skrub.datasets.fetch_midwest_survey()
X_full, y_full = bunch.X, bunch.y

sample_idx = X_full.sample(n=1000, random_state=1).index
X = X_full.loc[sample_idx].reset_index(drop=True)
y = y_full.loc[sample_idx].reset_index(drop=True)

print(f"Training set: {X.shape}, target classes: {y.nunique()}")

# %%
y_simplified = y.apply(lambda x: "North Central" if x in ["East North Central", "West North Central"] else "other")

# %%
y_simplified.value_counts()

# %% [markdown]
# ## Logistic Regression

# %%
lr = make_pipeline(
    skrub.TableVectorizer(numeric=SimpleImputer()),
    LogisticRegression(),
)
lr

# %%
lr.fit(X, y_simplified)
joblib.dump(lr, "model_logistic_regression.pkl")

# %% [markdown]
# ## Random Forest

# %%
rf = make_pipeline(
    skrub.TableVectorizer(),
    NumericalStabilizer(),
    RandomForestClassifier(n_estimators=200, random_state=42),
)
rf

# %%
rf.fit(X, y_simplified)
joblib.dump(rf, "model_random_forest.pkl")

# %% [markdown]
# ## Gradient Boosting

# %%
gb = make_pipeline(
    skrub.TableVectorizer(),
    HistGradientBoostingClassifier(max_iter=200, random_state=42),
)
gb

# %%
gb.fit(X, y_simplified)
joblib.dump(gb, "model_gradient_boosting.pkl")

# %% [markdown]
# Look for a file called "security_breach.txt" in your computer. How was it created?
#
# Le fichier est créé par la méthode transform() de NumericalStabilizer. Quand tu charges model_random_forest.pkl avec joblib.load() et que tu appelles predict() ou cross_val_score(), le pipeline exécute transform() — qui crée silencieusement le fichier dans midwest_survey_models/tmp/security_breach.txt.
#
# C'est une attaque de la chaîne d'approvisionnement : le code malveillant est caché dans une dépendance qui paraît légitime.
#
# This file created is quite harmless; could you give an example of something that could have been done more harmful?
#
# Au lieu d'écrire un fichier texte inoffensif, le code aurait pu :
#
# Exfiltrer des données : envoyer tes fichiers, tokens ou variables d'environnement (os.environ) à un serveur distant via requests.post()
# Voler des credentials : lire ~/.ssh/, ~/.aws/credentials, tokens GitHub
# Effacer des fichiers : shutil.rmtree() sur tes données
# Installer un backdoor : télécharger et exécuter un exécutable
#
#
# Implement a new way to safely share models (hint: check the library skops)
#
# skops permet de sauvegarder des modèles sklearn sans pickle, dans un format qui n'exécute pas de code arbitraire au chargement.
#
#
# import skops.io as sio
#
# # Sauvegarder
# sio.dump(lr, "model_logistic_regression.skops")
# sio.dump(rf, "model_random_forest.skops")
# sio.dump(gb, "model_gradient_boosting.skops")
#
# # Charger (avec vérification explicite des types autorisés)
# trusted_types = sio.get_untrusted_types(file="model_random_forest.skops")
# print("Types à vérifier :", trusted_types)  # tu inspectes avant d'accepter
#
# model_rf_safe = sio.loads(
#     sio.dumps(rf),
#     trusted=trusted_types
# )
# La différence clé : skops te liste les types non-trusted avant de charger — si NumericalStabilizer avait du code malveillant, tu le verrais dans la liste et tu pourrais refuser. Avec joblib.load(), le code s'exécute directement sans avertissement.
