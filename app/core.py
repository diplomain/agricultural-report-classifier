"""Agricultural report classifier: fine-tune GPT-2 so it answers a field report with a label token.

How it works
------------
Every training example is one short text:

    Report: <the farmer's report>
    Category:<label token>

The four label tokens (for example ``<pest_alert>``) are NEW special tokens that we add to GPT-2's
tokenizer and embedding matrix. The model is trained (with the Hugging Face ``Trainer``) so that the
token it produces right after ``Category:`` is the correct label token. At inference time we read the
model's next-token scores after ``Category:`` and pick the best of the four label tokens.

The heavy libraries (torch, transformers) are imported lazily inside the functions that need them,
so ``python -m app.cli hello`` and the data / file helpers stay fast and work without them.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Milestone 1: categories + sample text
# ---------------------------------------------------------------------------
CATEGORIES = ["pest_alert", "soil_condition", "irrigation_issue", "harvest_update"]

# One new special token per category, e.g. "<pest_alert>".
LABEL_TOKENS = {category: f"<{category}>" for category in CATEGORIES}

# Short plain-English hints. Used only to give each NEW token embedding a sensible starting point
# (the average of the embeddings of these words) so that training converges quickly.
LABEL_HINTS = {
    "pest_alert": "pest insects infestation",
    "soil_condition": "soil nutrients acidity",
    "irrigation_issue": "irrigation water pump",
    "harvest_update": "harvest crop yield",
}

# 14 different reports per category (balanced). The last EVAL_PER_CATEGORY of each category are
# held out and never used for training, so we can measure accuracy on text the model has not seen.
SAMPLES: dict[str, list[str]] = {
    "pest_alert": [
        "Spotted a heavy aphid infestation on the underside of the cotton leaves in the north field.",
        "Fall armyworm larvae are chewing through the maize whorls in block C, about one in four plants affected.",
        "Locust swarm sighted moving in from the east, they have already stripped the grazing paddock.",
        "Whiteflies are clustering on the tomato seedlings and leaving sticky honeydew everywhere.",
        "Rat burrows and gnaw marks found along the rice bunds, grain loss expected this week.",
        "Stem borer holes visible on sugarcane stalks, frass is piling up at the base of the plants.",
        "Brown planthopper numbers have crossed the threshold in the paddy, spraying needed urgently.",
        "Birds are raiding the sunflower heads each morning and wiping out whole rows of seed.",
        "Caterpillars have defoliated nearly half of the cabbage rows since Monday.",
        "Termites are tunnelling into the base of the young mango saplings near the boundary wall.",
        "Thrips damage with silvery streaks on the onion leaves is spreading across the plot.",
        "Wild boar broke through the fence last night and trampled the sweet potato ridges.",
        "Red spider mites are webbing the undersides of bean leaves in the greenhouse after the hot dry spell.",
        "Fruit fly traps caught over fifty adults overnight in the guava orchard.",
    ],
    "soil_condition": [
        "Soil in the lower field is badly compacted, roots cannot penetrate beyond ten centimetres.",
        "Soil test shows pH at 5.2, the plot is too acidic and needs lime before planting.",
        "Topsoil has been washed away by heavy rain, leaving deep erosion gullies along the slope.",
        "White salt crust is forming on the surface of the east plot, salinity looks very high.",
        "Organic matter is low and the sandy soil in block B drains too quickly to hold any nutrients.",
        "Heavy clay soil is cracking wide open now that it has dried out after the rains.",
        "Nitrogen deficiency is likely, the soil sample came back very low in nitrogen and phosphorus.",
        "Mulching with compost has darkened the soil and earthworm activity has clearly increased.",
        "Soil sample from the north plot shows organic carbon at only 0.4 percent, well below target.",
        "Soil near the stream has turned grey and smells sour, a sign of poor aeration.",
        "Potassium levels in the loam are adequate but zinc is low according to the lab results.",
        "Crusting after the storm has sealed the seedbed surface and the soil feels hard and powdery.",
        "Sandy patches in the west field show poor structure and almost no humus.",
        "Soil temperature at planting depth is only nine degrees, still too cold for sowing.",
    ],
    "irrigation_issue": [
        "The drip line in row twelve is blocked and half the tomato bed is getting no water.",
        "Main canal gate is jammed shut so the lower fields have had no water for three days.",
        "Diesel pump at the borewell failed this morning and the sprinklers are completely down.",
        "Water pressure in the sprinkler system has dropped, the far end of the pivot barely reaches the crop.",
        "Irrigation schedule needs changing, the east block is being watered twice as often as it needs.",
        "Pipe joint burst near the tank and a large amount of water is leaking onto the track.",
        "The solar pump controller keeps tripping at midday and the reservoir is running low.",
        "Farmers at the tail end of the canal say the water allocation never reaches their plots.",
        "Emitters on the drip tape are clogged with algae and need flushing with acid.",
        "Overhead sprinklers are creating puddles by the gate while the middle of the field stays dry.",
        "The irrigation timer was reset after the power cut and the valves did not open at dawn.",
        "Borewell water level has fallen by four metres since last month, so pumping hours must be cut.",
        "Furrow irrigation is uneven because the field levelling is poor and water pools at one end.",
        "Filter on the fertigation unit is clogged and flow to zone three has stopped.",
    ],
    "harvest_update": [
        "First picking of strawberries is done, we gathered nearly two tonnes from the main tunnel.",
        "Wheat harvest in block A is complete, the combine finished the last pass at dusk.",
        "Grain moisture is down to fourteen percent so we can start threshing the sorghum tomorrow.",
        "Mango pickers collected about four hundred crates today, grading and packing begin in the morning.",
        "Cotton harvest is about sixty percent finished and the second picking is scheduled for next week.",
        "Yield from the soybean plot came in at three tonnes per hectare, above last season.",
        "We need extra labour on Friday to bring in the remaining tomatoes before the rain arrives.",
        "Rice is ready for cutting, the panicles have turned golden and most grains are hard.",
        "The potato lifting crew has finished the lower field and the tubers are curing in the shed.",
        "Sugarcane cutting begins Monday and the mill has confirmed trucks for the first three days.",
        "Coffee cherries are ripening unevenly, so the harvest will be done in three passes.",
        "Maize cobs have been dried and shelled, and the bags are stacked in the store awaiting transport.",
        "Apple picking wrapped up today with a total of nine hundred bins sent to the cold store.",
        "Onion bulbs have been lifted and left to dry in the field for five days before storage.",
    ],
}

EVAL_PER_CATEGORY = 4


def get_train_rows() -> list[dict]:
    """Training rows: [{"text": ..., "label": category}, ...] (10 per category)."""
    return [
        {"text": text, "label": category}
        for category in CATEGORIES
        for text in SAMPLES[category][:-EVAL_PER_CATEGORY]
    ]


def get_eval_rows() -> list[dict]:
    """Held-out rows never used for training (4 per category)."""
    return [
        {"text": text, "label": category}
        for category in CATEGORIES
        for text in SAMPLES[category][-EVAL_PER_CATEGORY:]
    ]


# ---------------------------------------------------------------------------
# Milestone 2: extend GPT-2's tokenizer and model with the label tokens
# ---------------------------------------------------------------------------
PROMPT_PREFIX = "Report:"
PROMPT_SUFFIX = "\nCategory:"
MAX_REPORT_TOKENS = 200  # long reports are cut to this many tokens (GPT-2 can read 1024)


def extend_tokenizer_and_model(tokenizer, model) -> int:
    """Add the label tokens to the tokenizer and resize the model's embeddings to match.

    Returns how many tokens were newly added (4 the first time, 0 if already present).
    """
    import torch

    added = tokenizer.add_special_tokens({"additional_special_tokens": list(LABEL_TOKENS.values())})
    if tokenizer.pad_token is None:  # GPT-2 has no pad token; reuse end-of-text for padding
        tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    if added:
        # Start each new embedding at the average of related words instead of random noise.
        embeddings = model.get_input_embeddings().weight
        with torch.no_grad():
            for category, token in LABEL_TOKENS.items():
                new_id = tokenizer.convert_tokens_to_ids(token)
                hint_ids = tokenizer(" " + LABEL_HINTS[category], add_special_tokens=False)["input_ids"]
                embeddings[new_id] = embeddings[hint_ids].mean(dim=0)
    return added


def label_token_ids(tokenizer) -> list[int]:
    """Token ids of the label tokens, in CATEGORIES order. Fails if the tokenizer was not extended."""
    ids = tokenizer.convert_tokens_to_ids(list(LABEL_TOKENS.values()))
    if any(i is None or i == tokenizer.unk_token_id for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("This tokenizer does not have the report-type tokens. Train a model first: "
                         "python -m app.cli train")
    return list(ids)


def prompt_ids(tokenizer, text: str) -> list[int]:
    """Token ids for 'Report: <text>\\nCategory:' (identical for training and inference)."""
    body = tokenizer(" " + text.strip(), add_special_tokens=False)["input_ids"][:MAX_REPORT_TOKENS]
    return (tokenizer(PROMPT_PREFIX, add_special_tokens=False)["input_ids"]
            + body
            + tokenizer(PROMPT_SUFFIX, add_special_tokens=False)["input_ids"])


def load_base(base_model: str = "gpt2"):
    """Download/load pretrained GPT-2 and extend it with the label tokens. Returns (tokenizer, model)."""
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = GPT2LMHeadModel.from_pretrained(base_model)
    extend_tokenizer_and_model(tokenizer, model)
    return tokenizer, model


# ---------------------------------------------------------------------------
# Milestone 3: train with the Hugging Face Trainer
# ---------------------------------------------------------------------------
def encode_example(tokenizer, text: str, label: str) -> dict:
    """One training example. Only the label token counts towards the loss (the rest is -100)."""
    ids = prompt_ids(tokenizer, text)
    label_id = tokenizer.convert_tokens_to_ids(LABEL_TOKENS[label])
    return {
        "input_ids": ids + [label_id],
        "attention_mask": [1] * (len(ids) + 1),
        "labels": [-100] * len(ids) + [label_id],
    }


def make_collator(pad_id: int):
    """Pad a batch to its longest example (labels are padded with -100 so padding is ignored)."""
    def collate(batch):
        import torch

        longest = max(len(item["input_ids"]) for item in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in batch:
            gap = longest - len(item["input_ids"])
            out["input_ids"].append(item["input_ids"] + [pad_id] * gap)
            out["attention_mask"].append(item["attention_mask"] + [0] * gap)
            out["labels"].append(item["labels"] + [-100] * gap)
        return {key: torch.tensor(value) for key, value in out.items()}

    return collate


def train_model(tokenizer, model, rows: list[dict], output_dir: str, epochs: int = 15,
                batch_size: int = 8, learning_rate: float = 1e-4, seed: int = 42):
    """Fine-tune `model` in place on `rows` with the Hugging Face Trainer. Returns the TrainOutput."""
    from transformers import Trainer, TrainingArguments

    dataset = [encode_example(tokenizer, row["text"], row["label"]) for row in rows]
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset,
                      data_collator=make_collator(tokenizer.pad_token_id))
    return trainer.train()


def save_classifier(model, tokenizer, model_dir: str) -> None:
    """Write the fine-tuned model and the extended tokenizer to `model_dir`."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def evaluate(classifier, rows: list[dict]) -> float:
    """Fraction of `rows` the classifier labels correctly (0.0 to 1.0)."""
    if not rows:
        return 0.0
    correct = sum(1 for row in rows if classifier.classify(row["text"]) == row["label"])
    return correct / len(rows)


def train_and_save(base_model: str = "gpt2", model_dir: str = "model", epochs: int = 15,
                   batch_size: int = 8, learning_rate: float = 1e-4) -> dict:
    """Whole pipeline: load GPT-2, add tokens, train, save, and report accuracy."""
    tokenizer, model = load_base(base_model)
    train_rows, eval_rows = get_train_rows(), get_eval_rows()
    train_model(tokenizer, model, train_rows, output_dir=str(Path(model_dir) / "checkpoints"),
                epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    save_classifier(model, tokenizer, model_dir)
    classifier = Classifier(model, tokenizer)
    return {
        "train_accuracy": evaluate(classifier, train_rows),
        "eval_accuracy": evaluate(classifier, eval_rows),
        "model_dir": model_dir,
    }


# ---------------------------------------------------------------------------
# Milestone 4: classify one report string
# ---------------------------------------------------------------------------
class Classifier:
    """Wraps a fine-tuned model + tokenizer. `classify(text)` returns a category name."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.label_ids = label_token_ids(tokenizer)
        self.model.eval()

    def classify(self, text: str) -> str:
        import torch

        ids = prompt_ids(self.tokenizer, text)
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            next_token_scores = self.model(input_ids=input_ids).logits[0, -1]
        best = int(torch.argmax(next_token_scores[self.label_ids]))  # best of the label tokens only
        return CATEGORIES[best]


def load_classifier(model_dir: str = "model") -> Classifier:
    """Load a model saved by `train_and_save`."""
    if not Path(model_dir).is_dir():
        raise FileNotFoundError(f"No trained model found in '{model_dir}'. "
                                "Train one first: python -m app.cli train")
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    return Classifier(model, tokenizer)


# ---------------------------------------------------------------------------
# Milestone 5: classify many reports from a file (+ stretch: directory, JSON/CSV)
# ---------------------------------------------------------------------------
def read_reports(path: str):
    """Yield the non-empty lines of a text file, one report per line."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Report file not found: {path}")
    with file_path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def classify_file(classifier, path: str) -> list[dict]:
    """Classify every report in a file. Returns [{"report": ..., "category": ...}, ...]."""
    return [{"report": report, "category": classifier.classify(report)} for report in read_reports(path)]


def classify_directory(classifier, directory: str) -> list[dict]:
    """Classify every report in every *.txt file directly inside `directory`."""
    folder = Path(directory)
    if not folder.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")
    results = []
    for file_path in sorted(folder.glob("*.txt")):
        for row in classify_file(classifier, str(file_path)):
            results.append({"file": file_path.name, **row})
    return results


def format_results(results: list[dict], fmt: str = "text") -> str:
    """Render results as 'text' (tab separated), 'json' or 'csv'."""
    if fmt == "json":
        return json.dumps(results, indent=2, ensure_ascii=False)
    if fmt == "csv":
        fields = ["file", "report", "category"] if results and "file" in results[0] else ["report", "category"]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
        return buffer.getvalue().rstrip("\n")
    if fmt == "text":
        lines = []
        for row in results:
            parts = [row["file"]] if "file" in row else []
            lines.append("\t".join(parts + [row["category"], row["report"]]))
        return "\n".join(lines)
    raise ValueError(f"Unknown format: {fmt}")
