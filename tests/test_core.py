"""One group of tests per milestone in README.md.

Milestones 1 and 5 (and the CLI) need nothing but Python. Milestones 2-4 use the real GPT-2 tokenizer
(a ~1 MB download, cached) with a tiny randomly initialised GPT-2 *model*, so they run in seconds and never
download the 500 MB pretrained weights. They are skipped automatically if torch/transformers are missing
or the tokenizer cannot be downloaded.
"""
import json
import math

import pytest

from app import cli, core


# ---------------------------------------------------------------- helpers
class FakeClassifier:
    """Stands in for the real model in tests that only check plumbing."""

    def classify(self, text):
        return "pest_alert" if "aphid" in text.lower() else "soil_condition"


@pytest.fixture
def tokenizer():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("gpt2")
    except Exception as error:  # offline, hub down, ...
        pytest.skip(f"cannot load the GPT-2 tokenizer: {error}")


@pytest.fixture
def tiny_model():
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(vocab_size=50257, n_positions=256, n_embd=32, n_layer=2, n_head=2)
    return transformers.GPT2LMHeadModel(config)


@pytest.fixture
def extended(tokenizer, tiny_model):
    core.extend_tokenizer_and_model(tokenizer, tiny_model)
    return tokenizer, tiny_model


# ---------------------------------------------------------------- milestone 1
def test_milestone_1_categories_and_balanced_samples():
    assert 3 <= len(core.CATEGORIES) <= 4
    assert set(core.SAMPLES) == set(core.CATEGORIES)
    sizes = {len(texts) for texts in core.SAMPLES.values()}
    assert len(sizes) == 1, "every category must have the same number of samples"
    everything = [text for texts in core.SAMPLES.values() for text in texts]
    assert all(text.strip() for text in everything)
    assert len(set(everything)) == len(everything), "samples must be unique"


def test_milestone_1_train_and_eval_rows_do_not_overlap():
    train, held_out = core.get_train_rows(), core.get_eval_rows()
    assert {row["label"] for row in train} == set(core.CATEGORIES)
    assert {row["label"] for row in held_out} == set(core.CATEGORIES)
    assert not {r["text"] for r in train} & {r["text"] for r in held_out}
    for category in core.CATEGORIES:
        assert sum(r["label"] == category for r in train) == sum(r["label"] == core.CATEGORIES[0] for r in train)


# ---------------------------------------------------------------- milestone 2
def test_milestone_2_tokens_added_and_embeddings_resized(tokenizer, tiny_model):
    before = len(tokenizer)
    added = core.extend_tokenizer_and_model(tokenizer, tiny_model)
    assert added == len(core.CATEGORIES)
    assert len(tokenizer) == before + len(core.CATEGORIES)
    assert tiny_model.get_input_embeddings().weight.shape[0] == len(tokenizer)
    ids = core.label_token_ids(tokenizer)
    assert len(set(ids)) == len(core.CATEGORIES)
    assert all(i >= before for i in ids)
    # a label token is a single token, never split into pieces
    assert tokenizer(core.LABEL_TOKENS["pest_alert"], add_special_tokens=False)["input_ids"] == [ids[0]]


def test_milestone_2_extending_twice_adds_nothing(extended):
    tokenizer, model = extended
    assert core.extend_tokenizer_and_model(tokenizer, model) == 0


def test_milestone_2_training_example_only_scores_the_label(extended):
    tokenizer, _ = extended
    example = core.encode_example(tokenizer, "Aphids on the cotton.", "pest_alert")
    label_id = tokenizer.convert_tokens_to_ids(core.LABEL_TOKENS["pest_alert"])
    assert example["input_ids"][-1] == label_id
    assert example["labels"][-1] == label_id
    assert all(value == -100 for value in example["labels"][:-1])
    assert len(example["input_ids"]) == len(example["attention_mask"]) == len(example["labels"])


# ---------------------------------------------------------------- milestone 3
def test_milestone_3_trainer_runs_and_updates_the_model(extended, tmp_path):
    tokenizer, model = extended
    before = model.get_input_embeddings().weight.detach().clone()
    rows = core.get_train_rows()[:8] + core.get_train_rows()[10:18]
    result = core.train_model(tokenizer, model, rows, output_dir=str(tmp_path / "out"),
                              epochs=1, batch_size=4)
    assert result.global_step > 0
    assert math.isfinite(result.training_loss)
    assert not (model.get_input_embeddings().weight.detach() == before).all()


def test_milestone_3_collator_pads_with_ignored_labels(extended):
    tokenizer, _ = extended
    collate = core.make_collator(tokenizer.pad_token_id)
    short = core.encode_example(tokenizer, "Aphids.", "pest_alert")
    long = core.encode_example(tokenizer, "Aphids are spreading quickly across the whole cotton field.", "pest_alert")
    batch = collate([short, long])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    gap = len(long["input_ids"]) - len(short["input_ids"])
    assert gap > 0
    assert (batch["labels"][0, -gap:] == -100).all()
    assert (batch["attention_mask"][0, -gap:] == 0).all()


# ---------------------------------------------------------------- milestone 4
def test_milestone_4_classifier_returns_one_of_the_categories(extended):
    tokenizer, model = extended
    classifier = core.Classifier(model, tokenizer)
    label = classifier.classify("Aphids are eating the cotton.")
    assert label in core.CATEGORIES
    assert classifier.classify("Aphids are eating the cotton.") == label  # deterministic


def test_milestone_4_saved_model_loads_and_agrees(extended, tmp_path):
    tokenizer, model = extended
    core.save_classifier(model, tokenizer, str(tmp_path / "m"))
    loaded = core.load_classifier(str(tmp_path / "m"))
    text = "The drip line is blocked and the tomato bed is dry."
    assert loaded.classify(text) == core.Classifier(model, tokenizer).classify(text)


def test_milestone_4_missing_model_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        core.load_classifier(str(tmp_path / "nope"))


def test_milestone_4_cli_classifies_a_report_string(monkeypatch, capsys):
    monkeypatch.setattr(core, "load_classifier", lambda model_dir="model": FakeClassifier())
    cli.main(["classify", "Aphids on the cotton leaves"])
    assert capsys.readouterr().out.strip() == "pest_alert"


def test_milestone_4_cli_needs_exactly_one_input(monkeypatch):
    monkeypatch.setattr(core, "load_classifier", lambda model_dir="model": FakeClassifier())
    with pytest.raises(SystemExit):
        cli.main(["classify"])
    with pytest.raises(SystemExit):
        cli.main(["classify", "some text", "--file", "x.txt"])
    with pytest.raises(SystemExit):
        cli.main(["classify", "   "])


# ---------------------------------------------------------------- milestone 5
def test_milestone_5_reads_a_file_line_by_line_and_skips_blanks(tmp_path):
    path = tmp_path / "reports.txt"
    path.write_text("Aphids on cotton\n\n   \nSoil is too acidic\n", encoding="utf-8")
    assert list(core.read_reports(str(path))) == ["Aphids on cotton", "Soil is too acidic"]


def test_milestone_5_classifies_every_report_in_a_file(tmp_path):
    path = tmp_path / "reports.txt"
    path.write_text("Aphids on cotton\nSoil is too acidic\n", encoding="utf-8")
    assert core.classify_file(FakeClassifier(), str(path)) == [
        {"report": "Aphids on cotton", "category": "pest_alert"},
        {"report": "Soil is too acidic", "category": "soil_condition"},
    ]


def test_milestone_5_cli_file_json_and_csv(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(core, "load_classifier", lambda model_dir="model": FakeClassifier())
    path = tmp_path / "reports.txt"
    path.write_text("Aphids on cotton\nSoil is too acidic, needs lime\n", encoding="utf-8")

    cli.main(["classify", "--file", str(path)])
    assert capsys.readouterr().out.splitlines() == [
        "pest_alert\tAphids on cotton",
        "soil_condition\tSoil is too acidic, needs lime",
    ]

    cli.main(["classify", "--file", str(path), "--format", "json"])
    assert json.loads(capsys.readouterr().out)[1] == {
        "report": "Soil is too acidic, needs lime", "category": "soil_condition"}

    cli.main(["classify", "--file", str(path), "--format", "csv"])
    assert capsys.readouterr().out.splitlines() == [
        "report,category",
        "Aphids on cotton,pest_alert",
        '"Soil is too acidic, needs lime",soil_condition',
    ]


def test_milestone_5_missing_file_is_a_clean_error(monkeypatch, capsys):
    monkeypatch.setattr(core, "load_classifier", lambda model_dir="model": FakeClassifier())
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["classify", "--file", "does-not-exist.txt"])
    assert exit_info.value.code == 1
    assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------- stretch ideas
def test_stretch_directory_of_files(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(core, "load_classifier", lambda model_dir="model": FakeClassifier())
    (tmp_path / "b.txt").write_text("Soil is too acidic\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("Aphids on cotton\n", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("Aphids\n", encoding="utf-8")
    cli.main(["classify", "--dir", str(tmp_path), "--format", "csv"])
    assert capsys.readouterr().out.splitlines() == [
        "file,report,category",
        "a.txt,Aphids on cotton,pest_alert",
        "b.txt,Soil is too acidic,soil_condition",
    ]


def test_stretch_interactive_loop(monkeypatch, capsys):
    monkeypatch.setattr(core, "load_classifier", lambda model_dir="model": FakeClassifier())
    lines = iter(["Aphids on cotton", "Soil is too acidic", "quit", "never reached"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    cli.main(["classify", "--interactive"])
    out = capsys.readouterr().out.splitlines()
    assert out[-2:] == ["pest_alert", "soil_condition"]
