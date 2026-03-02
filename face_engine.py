import insightface
import numpy as np
import pickle
import cv2

DB_FILE = "face_db.pkl"
THRESHOLD = 0.50

# ---------------- LOAD DB ----------------
with open(DB_FILE, "rb") as f:
    FACE_DB = pickle.load(f)

print("Face DB loaded:", list(FACE_DB.keys()))

# ---------------- LOAD MODEL ----------------
app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=['CPUExecutionProvider']
)
app.prepare(ctx_id=-1)

# ---------------- BLUR FILTER ----------------
def blur_score(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

# ---------------- MATCH ----------------
def match_embedding(emb):
    best_name = "Unknown"
    best_score = -1

    for name, ref in FACE_DB.items():
        score = float(np.dot(emb, ref))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score

# ---------------- MAIN API ----------------
def recognize_faces(frame):

    results = []
    H, W = frame.shape[:2]

    faces = app.get(frame)

    for face in faces:
        x1,y1,x2,y2 = map(int, face.bbox)
        x1=max(0,x1); y1=max(0,y1)
        x2=min(W,x2); y2=min(H,y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if blur_score(crop) < 70:
            continue

        emb = face.embedding
        emb = emb / np.linalg.norm(emb)

        name, score = match_embedding(emb)

        results.append({
            "name": name,
            "score": float(score),
            "bbox": [x1,y1,x2,y2]
        })

    return results
