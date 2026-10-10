"""
Test du modèle d'identification de voix (njiva, fetra, iantsamitia, autre).

Charge /home/njiva/modele/modele_voix.h5 et preprocesseur.joblib (créés par
identification_voix.py) puis classe des enregistrements (wav, mp3, m4a...).

Exemples :
  # un seul fichier
  python tester_voix.py /home/njiva/test/clip1.m4a

  # un dossier de clips dont on connaît le locuteur (calcule la précision)
  python tester_voix.py /home/njiva/donnee/fetra/test --classe fetra

  # un dossier organisé en sous-dossiers njiva/ fetra/ iantsamitia/ autre/
  python tester_voix.py /home/njiva/test

  # options : --seuil 0.7 (change le seuil)  --detail (résultat clip par clip)
"""
import os, sys, argparse, shutil, subprocess, tempfile
import numpy as np
import librosa
import joblib
import tensorflow as tf

BASE_DIR  = "/home/njiva"
MODEL_DIR = f"{BASE_DIR}/modele"

PERSONNES = ["njiva", "fetra", "iantsamitia"]
AUTRE = "autre"
CLASSES = PERSONNES + [AUTRE]

SR = 16000
SEG_SEC = 3.0
MIN_SEC = 1.5
EXTENSIONS = (".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma")
COLS = {"mfcc": slice(0, 80), "tout": slice(0, 93), "temporel": slice(80, 93)}


# --- Mêmes fonctions que dans identification_voix.py (doivent rester identiques) ---
def charger_et_pretraiter(chemin):
    y, _ = librosa.load(chemin, sr=SR, mono=True)
    y = y - np.mean(y)
    y = librosa.effects.preemphasis(y)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y

def decouper(y, hop_sec=SEG_SEC):
    L, H = int(SEG_SEC * SR), int(hop_sec * SR)
    if len(y) < L:
        ok = len(y) >= MIN_SEC * SR and np.sqrt(np.mean(y ** 2)) > 0.01
        return [y] if ok else []
    segments = []
    for debut in range(0, len(y) - L + 1, H):
        seg = y[debut:debut + L]
        if np.sqrt(np.mean(seg ** 2)) > 0.01:
            segments.append(seg)
    return segments

def caracteristiques(seg):
    hop = 160
    feats = []
    mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=20, n_fft=512, hop_length=hop)
    delta = librosa.feature.delta(mfcc)
    for m in (mfcc, delta):
        feats += list(m.mean(axis=1)) + list(m.std(axis=1))
    rms = librosa.feature.rms(y=seg, frame_length=512, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(seg, frame_length=512, hop_length=hop)[0]
    cent = librosa.feature.spectral_centroid(y=seg, sr=SR, n_fft=512, hop_length=hop)[0]
    for v in (rms, zcr, cent):
        feats += [v.mean(), v.std()]
    f0 = librosa.yin(seg, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=hop)
    feats += [np.mean(f0), np.std(f0)]
    duree = len(seg) / SR
    inter = librosa.effects.split(seg, top_db=30, frame_length=512, hop_length=hop)
    duree_parole = sum(e - s for s, e in inter) / SR
    ratio_pause = 1 - duree_parole / duree
    pauses = [(inter[i + 1][0] - inter[i][1]) / SR for i in range(len(inter) - 1)]
    pause_moy = np.mean(pauses) if pauses else 0.0
    pause_max = np.max(pauses) if pauses else 0.0
    onsets = librosa.onset.onset_detect(y=seg, sr=SR, hop_length=hop)
    feats += [ratio_pause, len(pauses) / duree, pause_moy, pause_max, len(onsets) / duree]
    return np.array(feats, dtype=np.float32)
# -----------------------------------------------------------------------------------


def charger_audio(chemin):
    """Charge un fichier ; les formats autres que wav passent par ffmpeg."""
    if chemin.lower().endswith(".wav"):
        return charger_et_pretraiter(chemin)
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg introuvable : installe-le pour lire les mp3/m4a.")
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "clip.wav")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", chemin,
                            "-ac", "1", "-ar", str(SR), wav],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:150])
        return charger_et_pretraiter(wav)

def lister_audio(racine):
    if os.path.isfile(racine):
        return [racine]
    fichiers = []
    for dossier, _, noms in os.walk(racine):
        fichiers += [os.path.join(dossier, n) for n in sorted(noms)
                     if n.lower().endswith(EXTENSIONS)]
    return sorted(fichiers)

def vraie_classe(fichier, racine, classe_forcee):
    """Classe connue : --classe, ou nom d'un sous-dossier de la racine."""
    if classe_forcee:
        return classe_forcee
    if os.path.isdir(racine):
        for partie in os.path.relpath(fichier, racine).split(os.sep)[:-1]:
            if partie in CLASSES:
                return partie
    return None


def main():
    ap = argparse.ArgumentParser(description="Test du modèle de voix")
    ap.add_argument("chemin", help="fichier audio ou dossier")
    ap.add_argument("--classe", choices=CLASSES, help="locuteur réel de tous les fichiers")
    ap.add_argument("--seuil", type=float, help="remplace le seuil enregistré")
    ap.add_argument("--detail", action="store_true", help="affiche chaque clip de 3 s")
    args = ap.parse_args()

    modele = tf.keras.models.load_model(f"{MODEL_DIR}/modele_voix.h5", compile=False)
    p = joblib.load(f"{MODEL_DIR}/preprocesseur.joblib")
    scaler, enc, caract = p["scaler"], p["encodeur"], p["caract"]
    seuil = args.seuil if args.seuil is not None else p["seuil"]
    print(f"Modèle chargé | caractéristiques : {caract} | seuil : {seuil:.2f} "
          f"| classes : {list(enc.classes_)}\n")

    fichiers = lister_audio(args.chemin)
    if not fichiers:
        sys.exit("Aucun fichier audio trouvé.")

    y_true, y_pred = [], []
    print(f"{'fichier':38s} {'résultat':12s} {'confiance':>9s}  probabilités")
    for f in fichiers:
        nom = os.path.relpath(f, args.chemin) if os.path.isdir(args.chemin) else os.path.basename(f)
        try:
            segs = decouper(charger_audio(f))
        except Exception as e:
            print(f"{nom[:38]:38s} ERREUR : {e}"); continue
        if not segs:
            print(f"{nom[:38]:38s} ignoré (trop court ou silencieux)"); continue

        X = np.array([caracteristiques(s) for s in segs])[:, COLS[caract]]
        probas = modele.predict(scaler.transform(X), verbose=0)
        moy = probas.mean(axis=0)                      # moyenne sur les clips du fichier
        i = moy.argmax()
        res = AUTRE if moy[i] < seuil else enc.classes_[i]
        detail = " ".join(f"{c[:4]}={v:.2f}" for c, v in zip(enc.classes_, moy))
        print(f"{nom[:38]:38s} {res:12s} {moy[i]:9.2f}  {detail}")

        if args.detail and len(segs) > 1:
            for k, pr in enumerate(probas):
                j = pr.argmax()
                r = AUTRE if pr[j] < seuil else enc.classes_[j]
                print(f"    clip {k + 1:2d}: {r:12s} {pr[j]:.2f}")

        vrai = vraie_classe(f, args.chemin, args.classe)
        if vrai:
            y_true.append(vrai); y_pred.append(res)

    if y_true:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        print(f"\nPrécision : {np.mean(y_true == y_pred):.3f} "
              f"({int(np.sum(y_true == y_pred))}/{len(y_true)} fichiers)")
        labels = [c for c in CLASSES if c in set(y_true) | set(y_pred)]
        print("\nMatrice de confusion (lignes = vrai, colonnes = prédit) :")
        print(" " * 14 + "".join(f"{l[:11]:>12s}" for l in labels))
        for a in labels:
            ligne = "".join(f"{int(np.sum((y_true == a) & (y_pred == b))):12d}" for b in labels)
            print(f"{a:14s}{ligne}")


if __name__ == "__main__":
    main()
