import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from langdetect import detect, detect_langs
from transformers import pipeline
import google.generativeai as genai

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "sentiment_app")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PORT = int(os.getenv("PORT", "5000"))

# Initialize Flask and CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# MongoDB client
mongo_client = MongoClient(MONGODB_URI) if MONGODB_URI else None
db = mongo_client[MONGODB_DB_NAME] if mongo_client else None
results_col = db["results"] if db else None
summaries_col = db["summaries"] if db else None
if results_col:
  results_col.create_index([("timestamp", DESCENDING)])
  results_col.create_index([("batch_id", ASCENDING)])

# Hugging Face sentiment pipeline (multilingual)
sentiment_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# Gemini
if GEMINI_API_KEY:
  genai.configure(api_key=GEMINI_API_KEY)
  gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
  gemini_model = None

def safe_lang_detect(text: str) -> Tuple[str, float]:
  try:
    langs = detect_langs(text)
    if not langs:
      return "unknown", 0.0
    top = langs[0]
    return top.lang, float(top.prob)
  except Exception:
    try:
      return detect(text), 0.0
    except Exception:
      return "unknown", 0.0

def translate_to_english(text: str, source_lang: str) -> str:
  if source_lang.lower().startswith("en"):
    return text
  if not gemini_model:
    # Fallback: return original if Gemini not configured
    return text
  prompt = (
    "Translate the following text to English. Return only the translation with no additional commentary.\n\n"
    f"Source language (may be approximate): {source_lang}\n\n"
    f"Text:\n{text}"
  )
  try:
    resp = gemini_model.generate_content(prompt)
    translated = (resp.text or "").strip()
    return translated if translated else text
  except Exception:
    return text

def analyze_sentiment_en(text_en: str) -> Tuple[str, float]:
  # Returns label in {positive, neutral, negative} and confidence [0..1]
  out = sentiment_pipe(text_en, truncation=True)[0]
  label = out.get("label", "").lower()
  score = float(out.get("score", 0.0))
  # Standardize labels
  if "pos" in label:
    std = "positive"
  elif "neg" in label:
    std = "negative"
  else:
    std = "neutral"
  return std, score

def generate_summary(items: List[Dict[str, Any]]) -> str:
  if not gemini_model or not items:
    return ""
  # Build a concise, multilingual-aware summary prompt
  sample_lines = []
  for it in items[:50]:  # cap context size
    sample_lines.append(
      f"- [{it['detected_language']}] Sentiment: {it['sentiment_label']} ({round(it['confidence']*100)}%) | EN: {it['translated_text'][:220]}"
    )
  prompt = (
    "You are an analyst. Summarize the overall sentiment trends across these multilingual reviews. "
    "Highlight patterns, key positives/negatives, and concrete themes (like shipping delays, product quality). "
    "Be concise (3-6 sentences)."
    "\n\nReviews:\n" + "\n".join(sample_lines)
  )
  try:
    resp = gemini_model.generate_content(prompt)
    return (resp.text or "").strip()
  except Exception:
    return ""

def clamp_confidence(score: float) -> int:
  try:
    return int(max(0, min(100, round(score * 100))))
  except Exception:
    return 0

def save_results(docs: List[Dict[str, Any]]):
  if results_col and docs:
    results_col.insert_many(docs)

def save_summary(batch_id: str, summary: str):
  if summaries_col and summary:
    summaries_col.insert_one({
      "batch_id": batch_id,
      "summary": summary,
      "timestamp": datetime.utcnow()
    })

@app.route("/analyze", methods=["POST"])
def analyze():
  if not request.is_json:
    return jsonify({"error": "JSON body required"}), 400
  data = request.get_json() or {}
  texts: List[str] = []
  if "texts" in data and isinstance(data["texts"], list):
    texts = [str(t) for t in data["texts"] if isinstance(t, (str, int, float))]
  elif "text" in data and isinstance(data["text"], (str, int, float)):
    texts = [str(data["text"])]

  # Validate
  texts = [t.strip()[:2000] for t in texts if str(t).strip()]
  if not texts:
    return jsonify({"error": "No texts provided"}), 400
  if len(texts) > 100:
    texts = texts[:100]

  batch_id = str(uuid.uuid4()) if len(texts) > 1 else None
  ts = datetime.utcnow()

  items: List[Dict[str, Any]] = []
  for t in texts:
    lang, _ = safe_lang_detect(t)
    en_text = translate_to_english(t, lang)
    label, conf = analyze_sentiment_en(en_text)
    item = {
      "input_text": t,
      "detected_language": lang,
      "translated_text": en_text if en_text != t else (None if lang.startswith("en") else t),
      "sentiment_label": label,
      "confidence": clamp_confidence(conf),
      "timestamp": ts,
      "batch_id": batch_id,
    }
    items.append(item)

  # Optional summary for batch
  summary = ""
  if batch_id:
    summary = generate_summary(items)

  # Save to DB
  try:
    save_results(items)
    if summary:
      save_summary(batch_id, summary)
  except Exception as e:
    # Non-fatal for response
    pass

  resp = {
    "results": [
      {
        **it,
        # Ensure serializable timestamp
        "timestamp": it["timestamp"].isoformat() + "Z",
      } for it in items
    ]
  }
  if batch_id:
    resp["batch_id"] = batch_id
  if summary:
    resp["summary"] = summary

  return jsonify(resp), 200

@app.route("/history", methods=["GET"])
def history():
  limit = int(request.args.get("limit", "50"))
  if not results_col:
    return jsonify({"results": []}), 200
  cur = results_col.find({}).sort("timestamp", DESCENDING).limit(limit)
  out = []
  for doc in cur:
    doc["_id"] = str(doc["_id"])
    if isinstance(doc.get("timestamp"), datetime):
      doc["timestamp"] = doc["timestamp"].isoformat() + "Z"
    out.append(doc)
  return jsonify({"results": out}), 200

@app.route("/", methods=["GET"])
def root():
  return jsonify({"ok": True, "service": "Multilingual Sentiment Analyzer API"}), 200

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=PORT)
