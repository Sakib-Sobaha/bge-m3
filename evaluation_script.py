from loguru import logger
import requests
import pandas as pd
from pathlib import Path
from time import sleep

# ---------- Terminal Colors ----------
RED = "\033[31m"
GREEN = "\033[32m"
CYAN = "\033[36m"
RESET = "\033[0m"

# ---------- Logging ----------
logger.remove()

logger.add(
    "bge-m3/bge-m3-story-updated2-evaluation_threshold_0.6.log",
    format="<yellow>[{time:YYYY-MM-DD HH:mm:ss}]</yellow> <level>{level}</level> <cyan>{message}</cyan>",
    level="INFO",
    encoding="utf-8"
)

logger.add(
    "bge-m3/bge-m3-story-updated2-evaluation_mismatches_threshold_0.6.log",
    format="<yellow>[{time:YYYY-MM-DD HH:mm:ss}]</yellow> <level>{level}</level> <cyan>{message}</cyan>",
    level="INFO",
    encoding="utf-8",
    filter=lambda record: record["extra"].get("mismatch_only", False)
)

logger.add(
    sink=lambda msg: print(msg, end=""),
    format="<yellow>[{time:HH:mm:ss}]</yellow> <level>{level}</level> <cyan>{message}</cyan>",
    level="INFO"
)

# ---------- Data ----------
csv_path = Path("ec_story_test.csv")
tag_answer_path = Path("tag_answer.csv")

df = pd.read_csv(csv_path)
if "question" not in df.columns or "tag" not in df.columns:
    raise ValueError("Evaluation dataset must contain 'question' and 'tag' columns.")

df["question"] = df["question"].astype(str).str.strip()
df["tag"] = df["tag"].astype(str).str.strip()
if "answer" not in df.columns:
    df["answer"] = ""
else:
    df["answer"] = df["answer"].fillna("").astype(str).str.strip()

# ---------- Load Tag->Answer Mapping ----------
tag_answer_lookup = {}
if tag_answer_path.exists():
    tag_answer_df = pd.read_csv(tag_answer_path)
    if {"tag", "answer"}.issubset(tag_answer_df.columns):
        tag_answer_df = tag_answer_df[["tag", "answer"]].fillna("")
        tag_answer_df["tag"] = tag_answer_df["tag"].astype(str).str.strip()
        tag_answer_df["answer"] = tag_answer_df["answer"].astype(str).str.strip()
        tag_answer_lookup = dict(zip(tag_answer_df["tag"], tag_answer_df["answer"]))
    else:
        logger.warning("Tag/answer mapping missing 'tag' or 'answer' columns.")
else:
    logger.warning(f"Tag/answer file not found at {tag_answer_path}")

# Fill missing answers
if tag_answer_lookup:
    missing_mask = df["answer"] == ""
    if missing_mask.any():
        df.loc[missing_mask, "answer"] = df.loc[missing_mask, "tag"].map(tag_answer_lookup).fillna("")

# ---------- Evaluation ----------
url = "http://localhost:8000/search/"
correct_tag = 0
correct_answer = 0
total = len(df)

for idx, row in df.iterrows():
    question = row["question"]
    expected_tag = row["tag"]
    expected_answer = row["answer"]

    payload = {
        "query": question,
        "k": 5
    }

    try:
        response = requests.post(url, json=payload, timeout=20)
        result = response.json()

        results_list = result.get("results", [])
        if not results_list:
            logger.warning(f"[{idx+1}] Empty 'results' for question: {question}")
            predicted_tag, predicted_answer, score = "", "", 0.0
        else:
            top_result = results_list[0]
            predicted_tag = str(top_result.get("tag", "")).strip()
            predicted_answer = str(top_result.get("answer", "")).strip()
            score = top_result.get("score", 0.0)

        tag_match = predicted_tag == expected_tag
        answer_match = predicted_answer == expected_answer

        if tag_match:
            correct_tag += 1
        if answer_match:
            correct_answer += 1

        logger.info(
            f"\n[ID: {idx+1}/{total}]"
            f"\nQuestion: {question}"
            f"\nExpected Tag: {expected_tag}"
            f"\nPredicted Tag: {predicted_tag} | Tag Match: {tag_match}"
            f"\nExpected Answer: {expected_answer}"
            f"\nPredicted Answer: {predicted_answer} | Answer Match: {answer_match}"
            f"\nScore: {score:.3f}\n" + "=" * 60
        )

        # Log mismatches separately
        if not tag_match or not answer_match:
            logger.bind(mismatch_only=True).info(
                f"\n[MISMATCH] [ID: {idx+1}/{total}]"
                f"\nQuestion: {question}"
                f"\nExpected Tag: {expected_tag} | Predicted: {predicted_tag}"
                f"\nExpected Answer: {expected_answer}"
                f"\nPredicted Answer: {predicted_answer}"
                f"\nScore: {score:.3f}\n" + "=" * 60
            )

        print(f"[{idx+1}] Question: {question}")
        print(f"Expected Tag: {expected_tag} | Predicted: {predicted_tag} | Match: {tag_match}")
        print(f"Expected Ans: {expected_answer}\nPredicted Ans: {predicted_answer} | Match: {answer_match}")
        print(f"Score: {score:.3f}\n" + "=" * 60)

    except Exception as e:
        print(f"[{idx+1}] Request failed for question: {question}")
        print("Error:", e)
        logger.bind(mismatch_only=True).info(
            f"[ERROR] [{idx+1}] Question: {question}\nError: {e}\n" + "=" * 60
        )

    sleep(0.2)

# ---------- Summary ----------
tag_acc = (correct_tag / total * 100.0) if total else 0.0
ans_acc = (correct_answer / total * 100.0) if total else 0.0

summary = (
    f"\nTotal: {total}"
    f"\nTag Accuracy: {correct_tag}/{total} = {tag_acc:.2f}%"
    f"\nAnswer Accuracy: {correct_answer}/{total} = {ans_acc:.2f}%"
)
print(summary)
logger.info(summary)
