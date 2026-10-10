"""
Identification de la voix de 3 personnes (njiva, fetra, iantsamitia) + classe "autre"
Réseau de neurones TensorFlow/Keras, caractéristiques MFCC + temporisation.

Structure attendue :
    /home/njiva/
    ├── donnee/
    │   ├── njiva/        (train/ et test/)
    │   ├── fetra/        (train/ et test/)
    │   ├── iantsamitia/  (train/ et test/)
    │   └── autre/        (train/ et test/)
    ├── modele/
    └── script/identification_voix.py

Dépendances : pip install librosa soundfile scikit-learn joblib tensorflow
(et ffmpeg installé sur la machine pour lire/convertir mp3 et m4a)
"""

# %% 1 - Configuration
import os, glob, random, shutil, subprocess, sys
import numpy as np
import librosa

# Dossier du projet : /home/njiva s'il existe (machine locale ou fichiers déjà
# présents dans Colab), sinon Drive (montage possible seulement dans un notebook)
if os.path.isdir("/home/njiva/donnee"):
    BASE_DIR = "/home/njiva"
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        BASE_DIR = "/content/drive/MyDrive/projet_voix"
    except Exception as e:
        sys.exit(f"Dossier /home/njiva/donnee introuvable et Drive indisponible : {e}")

RAW_DIR   = f"{BASE_DIR}/donnee"        # originaux (wav, mp3, m4a...), jamais modifiés
DATA_DIR  = f"{BASE_DIR}/donnee_wav"    # copies converties en wav (créé automatiquement)
MODEL_DIR = f"{BASE_DIR}/modele"

PERSONNES = ["njiva", "fetra", "iantsamitia"]
AUTRE = "autre"
CLASSES = PERSONNES + [AUTRE]

SR = 16000          # fréquence d'échantillonnage
SEG_SEC = 3.0       # durée d'un segment / clip
HOP_SEC = 1.5       # pas entre segments d'entraînement (chevauchement 50 %)
MIN_SEC = 1.5       # durée minimale d'un clip utilisable
N_TRAIN, N_TEST = 15, 5   # utilisés seulement si un dossier n'a pas de sous-dossiers train/test
SEUIL = 0.60        # confiance minimale, sinon "autre"
SEED = 42

# Caractéristiques utilisées (étude de la temporisation) :
#   "mfcc"     -> MFCC + deltas seuls (80 valeurs)
#   "tout"     -> MFCC + caractéristiques temporelles (93 valeurs)
#   "temporel" -> caractéristiques temporelles seules (13 valeurs)
CARACT = "tout"
COLS = {"mfcc": slice(0, 80), "tout": slice(0, 93), "temporel": slice(80, 93)}

# %% 2 - Conversion des fichiers en WAV 16 kHz mono (train/ et test/ conservés)
EXTENSIONS = (".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma")

def convertir_tout():
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg est introuvable : installe-le (ex. sudo apt install ffmpeg).")
    total, convertis, ignores, erreurs = 0, 0, 0, []
    for classe in CLASSES:
        src_classe = os.path.join(RAW_DIR, classe)
        if not os.path.isdir(src_classe):
            print(f"Dossier introuvable : {src_classe}")
            continue
        for racine, _, noms in os.walk(src_classe):
            rel = os.path.relpath(racine, RAW_DIR)          # ex. fetra/train
            dst_dir = os.path.join(DATA_DIR, rel)
            for nom in sorted(noms):
                base, ext = os.path.splitext(nom)
                if ext.lower() not in EXTENSIONS:
                    continue
                total += 1
                os.makedirs(dst_dir, exist_ok=True)
                # l'extension d'origine reste dans le nom pour éviter les collisions
                dst = os.path.join(dst_dir, f"{base}_{ext[1:].lower()}.wav")
                if os.path.exists(dst):
                    ignores += 1
                    continue
                cmd = ["ffmpeg", "-y", "-loglevel", "error",
                       "-i", os.path.join(racine, nom),
                       "-ac", "1", "-ar", str(SR), dst]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode == 0:
                    convertis += 1
                else:
                    erreurs.append((os.path.join(rel, nom), r.stderr.strip()[:120]))
    print(f"{total} fichiers audio trouvés : {convertis} convertis, "
          f"{ignores} déjà convertis, {len(erreurs)} erreurs")
    for f, e in erreurs:
        print("  ERREUR", f, "->", e)

convertir_tout()

for c in CLASSES:
    for s in ("train", "test"):
        n = len(glob.glob(os.path.join(DATA_DIR, c, s, "*.wav")))
        print(f"{c:12s} {s:5s}: {n} fichiers .wav")

# %% 3 - Prétraitement et caractéristiques
def charger_et_pretraiter(chemin):
    y, _ = librosa.load(chemin, sr=SR, mono=True)
    y = y - np.mean(y)                        # supprime la composante continue
    y = librosa.effects.preemphasis(y)        # accentue les hautes fréquences
    y = y / (np.max(np.abs(y)) + 1e-9)        # normalisation d'amplitude
    return y

def decouper(y, hop_sec=HOP_SEC):
    """Découpe en segments de 3 s. Un enregistrement plus court (clip de test)
    est gardé tel quel, sans remplissage de zéros (qui fausserait les pauses)."""
    L, H = int(SEG_SEC * SR), int(hop_sec * SR)
    if len(y) < L:
        ok = len(y) >= MIN_SEC * SR and np.sqrt(np.mean(y ** 2)) > 0.01
        return [y] if ok else []
    segments = []
    for debut in range(0, len(y) - L + 1, H):
        seg = y[debut:debut + L]
        if np.sqrt(np.mean(seg ** 2)) > 0.01:   # écarte les segments silencieux
            segments.append(seg)
    return segments

def caracteristiques(seg):
    hop = 160   # 10 ms
    feats = []

    # Spectral : MFCC + deltas -> colonnes 0 à 79
    mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=20, n_fft=512, hop_length=hop)
    delta = librosa.feature.delta(mfcc)
    for m in (mfcc, delta):
        feats += list(m.mean(axis=1)) + list(m.std(axis=1))

    # Temporel : énergie, passages par zéro, centroïde -> colonnes 80 à 85
    rms = librosa.feature.rms(y=seg, frame_length=512, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(seg, frame_length=512, hop_length=hop)[0]
    cent = librosa.feature.spectral_centroid(y=seg, sr=SR, n_fft=512, hop_length=hop)[0]
    for v in (rms, zcr, cent):
        feats += [v.mean(), v.std()]

    # Hauteur (F0) -> colonnes 86 à 87
    f0 = librosa.yin(seg, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=hop)
    feats += [np.mean(f0), np.std(f0)]

    # Temporisation : pauses et débit -> colonnes 88 à 92
    duree = len(seg) / SR
    inter = librosa.effects.split(seg, top_db=30, frame_length=512, hop_length=hop)
    duree_parole = sum(e - s for s, e in inter) / SR
    ratio_pause = 1 - duree_parole / duree
    pauses = [(inter[i + 1][0] - inter[i][1]) / SR for i in range(len(inter) - 1)]
    pause_moy = np.mean(pauses) if pauses else 0.0
    pause_max = np.max(pauses) if pauses else 0.0
    onsets = librosa.onset.onset_detect(y=seg, sr=SR, hop_length=hop)
    feats += [ratio_pause, len(pauses) / duree, pause_moy, pause_max, len(onsets) / duree]

    return np.array(feats, dtype=np.float32)   # 93 valeurs

# %% 4 - Jeu de données (train/ et test/ existants)
random.seed(SEED)

def fichiers_classe(classe):
    """Utilise les sous-dossiers train/ et test/ s'ils existent,
    sinon tire au hasard N_TRAIN fichiers pour l'entraînement et N_TEST pour le test."""
    tr = sorted(glob.glob(os.path.join(DATA_DIR, classe, "train", "*.wav")))
    te = sorted(glob.glob(os.path.join(DATA_DIR, classe, "test", "*.wav")))
    if tr and te:
        return tr, te
    tous = sorted(glob.glob(os.path.join(DATA_DIR, classe, "**", "*.wav"), recursive=True))
    random.shuffle(tous)
    print(f"{classe}: pas de train/test -> séparation aléatoire {N_TRAIN}/{N_TEST}")
    return tous[:N_TRAIN], tous[N_TRAIN:N_TRAIN + N_TEST]

def construire(fichiers, label, id_offset, hop_sec):
    X, y, g = [], [], []
    for i, f in enumerate(fichiers):
        for seg in decouper(charger_et_pretraiter(f), hop_sec=hop_sec):
            X.append(caracteristiques(seg)); y.append(label); g.append(id_offset + i)
    return X, y, g

X_tr, y_tr, g_tr = [], [], []
X_te, y_te, g_te = [], [], []
gid = 0
for c in CLASSES:
    train_f, test_f = fichiers_classe(c)
    print(f"{c:12s}: {len(train_f)} fichiers train, {len(test_f)} fichiers test")

    # entraînement : segments de 3 s avec chevauchement
    a, b, gg = construire(train_f, c, gid, HOP_SEC); gid += len(train_f)
    X_tr += a; y_tr += b; g_tr += gg
    # test : clips de 3 s sans chevauchement (un clip = une décision)
    a, b, gg = construire(test_f, c, gid, SEG_SEC); gid += len(test_f)
    X_te += a; y_te += b; g_te += gg

X_tr, y_tr, g_tr = np.array(X_tr), np.array(y_tr), np.array(g_tr)
X_te, y_te, g_te = np.array(X_te), np.array(y_te), np.array(g_te)
print("Train:", X_tr.shape, " Test:", X_te.shape)

# %% 5 - Réseau TensorFlow et entraînement
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

tf.random.set_seed(SEED)
np.random.seed(SEED)

cols = COLS[CARACT]
Xtr, Xte = X_tr[:, cols], X_te[:, cols]
print(f"Caractéristiques utilisées : '{CARACT}' ({Xtr.shape[1]} valeurs)")

def entrainer(X, y, g, epochs=200, verbose=0):
    scaler = StandardScaler().fit(X)
    enc = LabelEncoder().fit(y)
    Xs, ys = scaler.transform(X), enc.transform(y)

    # validation séparée par enregistrement
    idx_a, idx_v = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                          random_state=SEED).split(Xs, ys, groups=g))
    tf.keras.backend.clear_session()
    modele = tf.keras.Sequential([
        layers.Input(shape=(Xs.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(enc.classes_), activation="softmax"),
    ])
    modele.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = modele.fit(Xs[idx_a], ys[idx_a],
                      validation_data=(Xs[idx_v], ys[idx_v]),
                      epochs=epochs, batch_size=32, verbose=verbose,
                      callbacks=[callbacks.EarlyStopping(patience=20,
                                                         restore_best_weights=True)])
    return modele, scaler, enc, hist

def predire(modele, scaler, enc, X, seuil):
    proba = modele.predict(scaler.transform(X), verbose=0)
    labels = enc.classes_[proba.argmax(axis=1)]
    return np.where(proba.max(axis=1) < seuil, AUTRE, labels)

modele, scaler, enc, hist = entrainer(Xtr, y_tr, g_tr)
print("Classes :", list(enc.classes_))

# %% 6 - Évaluation sur les clips de test
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

y_pred = predire(modele, scaler, enc, Xte, SEUIL)
print(f"{len(y_te)} clips de test, précision : {np.mean(y_pred == y_te):.3f}")
print(classification_report(y_te, y_pred, zero_division=0))

labels = list(enc.classes_)
ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred, labels=labels),
                       display_labels=labels).plot(cmap="Blues")
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val")
ax[0].set_title("Perte"); ax[0].legend()
ax[1].plot(hist.history["accuracy"], label="train"); ax[1].plot(hist.history["val_accuracy"], label="val")
ax[1].set_title("Précision"); ax[1].legend()
plt.show()

# %% 7 - Validation croisée par enregistrement
from sklearn.model_selection import GroupKFold

X_all = np.vstack([Xtr, Xte]); y_all = np.concatenate([y_tr, y_te])
g_all = np.concatenate([g_tr, g_te])

scores = []
for k, (a, t) in enumerate(GroupKFold(n_splits=5).split(X_all, y_all, g_all)):
    m, sc, en, _ = entrainer(X_all[a], y_all[a], g_all[a])
    p = predire(m, sc, en, X_all[t], SEUIL)
    scores.append(np.mean(p == y_all[t]))
    print(f"Pli {k + 1} : {scores[-1]:.3f}")
print(f"Moyenne ({CARACT}) : {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

# %% 8 - Sauvegarde du modèle (.h5)
import joblib
os.makedirs(MODEL_DIR, exist_ok=True)

modele.save(f"{MODEL_DIR}/modele_voix.h5")
joblib.dump({"scaler": scaler, "encodeur": enc, "seuil": SEUIL, "caract": CARACT},
            f"{MODEL_DIR}/preprocesseur.joblib")
print("Sauvegardé dans", MODEL_DIR)

# %% 9 - Identifier un nouvel enregistrement
# Pour utiliser un modèle déjà entraîné (nouvelle session), décommenter :
# modele = tf.keras.models.load_model(f"{MODEL_DIR}/modele_voix.h5", compile=False)
# p = joblib.load(f"{MODEL_DIR}/preprocesseur.joblib")
# scaler, enc, SEUIL, CARACT = p["scaler"], p["encodeur"], p["seuil"], p["caract"]

def identifier(chemin):
    segs = decouper(charger_et_pretraiter(chemin))   # lit aussi mp3/m4a via ffmpeg
    if not segs:
        print("Enregistrement trop court ou silencieux"); return
    X = np.array([caracteristiques(s) for s in segs])[:, COLS[CARACT]]
    proba = modele.predict(scaler.transform(X), verbose=0).mean(axis=0)  # moyenne par enregistrement
    for nom, pr in zip(enc.classes_, proba):
        print(f"  {nom:12s} {pr:.2f}")
    i = proba.argmax()
    print("Résultat :", AUTRE if proba[i] < SEUIL else enc.classes_[i])

def tester_dossier(dossier):
    """Teste tous les clips (wav/mp3/m4a...) d'un dossier, un résultat par fichier."""
    for nom in sorted(os.listdir(dossier)):
        if not nom.lower().endswith(EXTENSIONS):
            continue
        segs = decouper(charger_et_pretraiter(os.path.join(dossier, nom)))
        if not segs:
            print(f"{nom:30s} -> ignoré (trop court ou silencieux)"); continue
        X = np.array([caracteristiques(s) for s in segs])[:, COLS[CARACT]]
        proba = modele.predict(scaler.transform(X), verbose=0).mean(axis=0)
        i = proba.argmax()
        res = AUTRE if proba[i] < SEUIL else enc.classes_[i]
        print(f"{nom:30s} -> {res:12s} (confiance {proba[i]:.2f})")

# identifier(f"{BASE_DIR}/test.m4a")
# tester_dossier(f"{BASE_DIR}/test")   # dossier de clips de 3 s