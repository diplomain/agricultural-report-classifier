
"""Simple keyword baseline for the agricultural report classifier."""

from app.core import CATEGORIES, get_eval_rows

KEYWORDS = {
    "pest_alert": [
        "aphid", "armyworm", "locust", "whitefly", "rat", "borer",
        "planthopper", "bird", "caterpillar", "termite", "thrips",
        "mite", "fruit fly", "pest", "insect"
    ],
    "soil_condition": [
        "soil", "ph", "acidic", "salinity", "erosion", "compacted",
        "nitrogen", "phosphorus", "potassium", "organic matter",
        "humus", "crusting", "topsoil", "nutrient"
    ],
    "irrigation_issue": [
        "irrigation", "water", "drip", "pump", "sprinkler", "canal",
        "borewell", "pipe", "leak", "valve", "flow", "emitter"
    ],
    "harvest_update": [
        "harvest", "picking", "yield", "crop ready", "cutting",
        "threshing", "collected", "collect", "tonne", "crates",
        "moisture", "packing", "picked", "shelled"
    ],
}


def classify(report: str) -> str:
    """Predict a category using keyword matches."""
    text = report.lower()

    scores = {
        category: sum(keyword in text for keyword in words)
        for category, words in KEYWORDS.items()
    }

    # Stable tie-breaking follows CATEGORIES order.
    return max(CATEGORIES, key=lambda category: scores[category])


def evaluate():
    """Evaluate the baseline on the same held-out examples as GPT-2."""
    rows = get_eval_rows()
    correct = 0
    results = []

    for row in rows:
        predicted = classify(row["text"])
        actual = row["label"]
        is_correct = predicted == actual
        correct += is_correct

        results.append({
            "report": row["text"],
            "actual": actual,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(rows) if rows else 0.0
    return accuracy, results


if __name__ == "__main__":
    accuracy, results = evaluate()

    print("Keyword Baseline Evaluation")
    print("-" * 30)
    print(f"Correct: {sum(r['correct'] for r in results)}/{len(results)}")
    print(f"Accuracy: {accuracy:.2%}")

    for row in results:
        status = "PASS" if row["correct"] else "MISS"
        print(f"[{status}] Actual: {row['actual']} | Predicted: {row['predicted']}")
