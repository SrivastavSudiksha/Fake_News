
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import numpy as np

TRAINING_DATA = [
    ("Scientists discover miracle cure for cancer hidden by government!", 0),
    ("5G towers are secretly spreading virus through the air!", 0),
    ("Doctors hate him! This trick cures diabetes in 3 days!", 0),
    ("Bill Gates planning to microchip everyone through vaccines!", 0),
    ("NASA admits Moon landing was faked in a Hollywood studio!", 0),
    ("Share this before it gets deleted: COVID vaccine contains tracking device!", 0),
    ("World leaders are secretly lizard people controlling your mind!", 0),
    ("One fruit destroys cancer 100x better than chemo!", 0),
    ("Fluoride in water is making people stupid on purpose!", 0),
    ("Eating chocolate every day burns belly fat, big pharma hides it!", 0),
    ("Secret society controls all world governments!", 0),
    ("Election was rigged using alien technology, sources confirm!", 0),
    ("Ancient remedy cures all diseases in 24 hours!", 0),
    ("Politician confesses crimes in viral video media is hiding!", 0),
    ("Earth is actually flat, NASA is covering up the truth!", 0),
    ("Drinking bleach kills coronavirus, doctor reveals!", 0),
    ("Government putting mind control drugs in tap water!", 0),
    ("Celebrity secretly dead, replaced by clone for years!", 0),
    ("Scientists prove mobile phones cause instant brain cancer!", 0),
    ("Aliens already living among us, government confirms secretly!", 0),
    ("Reserve Bank of India raised repo rate by 25 basis points to control inflation.", 1),
    ("ISRO successfully launched Chandrayaan-3 mission to the lunar south pole.", 1),
    ("Indian government announced new policy to boost manufacturing under Make in India.", 1),
    ("WHO releases updated vaccination guidelines for 2024.", 1),
    ("Apple reported quarterly revenue of 89 billion dollars beating expectations.", 1),
    ("India GDP grew at 7.2 percent in the last fiscal quarter according to government data.", 1),
    ("Scientists at MIT developed a battery that charges in under five minutes.", 1),
    ("Supreme Court of India delivered verdict on the electoral bonds case.", 1),
    ("Global temperatures in 2023 were highest recorded in over a century says NASA.", 1),
    ("Union Budget 2024 allocated 11 lakh crore rupees for capital expenditure.", 1),
    ("IIT Delhi developed low cost water purification using nanotechnology.", 1),
    ("India signed trade agreement with UAE to boost bilateral trade.", 1),
    ("Electric vehicle sales in India grew by 45 percent year on year.", 1),
    ("Parliament passed new education bill after three days of debate.", 1),
    ("Sensex crossed 75000 points for the first time in stock market history.", 1),
    ("India won gold medal in wrestling at Commonwealth Games.", 1),
    ("Government launched free health insurance scheme for below poverty line families.", 1),
    ("New metro line inaugurated in Delhi connecting airport to city centre.", 1),
    ("Scientists develop new malaria vaccine with 90 percent effectiveness.", 1),
    ("RBI announced new rules for digital payment security.", 1),
]

class FakeNewsModel:
    def __init__(self):
        self.nb_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("clf",   MultinomialNB()),
        ])
        self.lr_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("clf",   LogisticRegression(max_iter=1000)),
        ])
        self.metrics = {}
        self._train()

    def _train(self):
        texts  = [d[0] for d in TRAINING_DATA]
        labels = [d[1] for d in TRAINING_DATA]
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        self.nb_pipeline.fit(X_train, y_train)
        self.lr_pipeline.fit(X_train, y_train)
        nb_pred = self.nb_pipeline.predict(X_test)
        lr_pred = self.lr_pipeline.predict(X_test)
        self.metrics = {
            "train_size": len(X_train),
            "test_size":  len(X_test),
            "naive_bayes": {
                "accuracy":         round(accuracy_score(y_test, nb_pred) * 100, 1),
                "report":           classification_report(y_test, nb_pred, target_names=["FAKE", "REAL"], output_dict=True),
                "confusion_matrix": confusion_matrix(y_test, nb_pred).tolist(),
            },
            "logistic_regression": {
                "accuracy":         round(accuracy_score(y_test, lr_pred) * 100, 1),
                "report":           classification_report(y_test, lr_pred, target_names=["FAKE", "REAL"], output_dict=True),
                "confusion_matrix": confusion_matrix(y_test, lr_pred).tolist(),
            },
        }

    def predict(self, text: str) -> dict:
        nb_proba = self.nb_pipeline.predict_proba([text])[0]
        nb_label = int(np.argmax(nb_proba))
        lr_proba = self.lr_pipeline.predict_proba([text])[0]
        lr_label = int(np.argmax(lr_proba))
        avg_fake = round((nb_proba[0] + lr_proba[0]) / 2 * 100, 1)
        avg_real = round((nb_proba[1] + lr_proba[1]) / 2 * 100, 1)
        verdict  = "REAL" if avg_real > avg_fake else "FAKE"
        return {
            "verdict":    verdict,
            "confidence": max(avg_fake, avg_real),
            "prob_fake":  avg_fake,
            "prob_real":  avg_real,
            "naive_bayes": {
                "label":      "REAL" if nb_label == 1 else "FAKE",
                "confidence": round(float(max(nb_proba)) * 100, 1),
            },
            "logistic_regression": {
                "label":      "REAL" if lr_label == 1 else "FAKE",
                "confidence": round(float(max(lr_proba)) * 100, 1),
            },
        }


app = FastAPI(
    title="Fake News Detector",
    description="ML-based fake news detection — TF-IDF + Naive Bayes + Logistic Regression",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = FakeNewsModel()

app.mount("/static", StaticFiles(directory="static"), name="static")


class NewsRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=2000,
                      example="Scientists discover miracle cure hidden by government!")


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/health", tags=["Info"])
def health():
    return {"status": "ok", "model": "loaded"}


@app.get("/metrics", tags=["ML"])
def get_metrics():
    return model.metrics


@app.get("/examples", tags=["Info"])
def get_examples():
    return {
        "examples": [
            {"id": 1, "expected": "FAKE", "text": "5G towers are spreading virus through the air!"},
            {"id": 2, "expected": "REAL", "text": "ISRO successfully launched Chandrayaan-3 to the lunar south pole."},
            {"id": 3, "expected": "FAKE", "text": "Drinking bleach cures coronavirus says leaked government document!"},
            {"id": 4, "expected": "REAL", "text": "India GDP grew at 7.2 percent in the last fiscal quarter."},
        ]
    }


@app.post("/predict", tags=["Predict"])
def predict(req: NewsRequest):
    try:
        result = model.predict(req.text)
        return {"input": req.text, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))