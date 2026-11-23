"""Command-line entry point: `python -m app.cli --help`."""
import argparse

from app import core


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agricultural-report-classifier",
        description="Agronomists use this tool to automatically classify short farmer reports or field "
                    "observations (e.g., \"pest alert\", \"soil condition\") to monitor farm health and "
                    "respond proactively.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hello", help="check that everything is wired up")

    train = sub.add_parser("train", help="fine-tune GPT-2 on the sample reports and save the model")
    train.add_argument("--base-model", default="gpt2", help="pretrained model to start from (default: gpt2)")
    train.add_argument("--model-dir", default="model", help="where to save the model (default: model)")
    train.add_argument("--epochs", type=int, default=15, help="training epochs (default: 15)")
    train.add_argument("--batch-size", type=int, default=8, help="batch size (default: 8)")
    train.add_argument("--learning-rate", type=float, default=1e-4, help="learning rate (default: 1e-4)")

    classify = sub.add_parser("classify", help="classify a report, a file of reports, or a folder of files")
    classify.add_argument("report", nargs="?", help="one report as text, e.g. \"Aphids on the cotton\"")
    classify.add_argument("-f", "--file", help="text file with one report per line")
    classify.add_argument("-d", "--dir", help="folder; every *.txt file inside is classified")
    classify.add_argument("-i", "--interactive", action="store_true",
                          help="keep classifying reports you type, until you enter an empty line")
    classify.add_argument("--format", choices=["text", "json", "csv"], default="text",
                          help="output format (default: text)")
    classify.add_argument("--model-dir", default="model", help="folder with the trained model (default: model)")
    return parser


def _run_interactive(classifier) -> None:
    print("Type a report and press Enter. Empty line or 'quit' to stop.")
    while True:
        try:
            line = input("report> ")
        except EOFError:
            break
        line = line.strip()
        if not line or line.lower() in {"quit", "exit"}:
            break
        print(classifier.classify(line))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Agricultural Report Classifier is ready. Open README.md for your milestones.")
        return

    if args.command == "train":
        result = core.train_and_save(args.base_model, args.model_dir, args.epochs,
                                     args.batch_size, args.learning_rate)
        print(f"Saved model to {result['model_dir']}")
        print(f"Accuracy on training reports: {result['train_accuracy']:.0%}")
        print(f"Accuracy on held-out reports: {result['eval_accuracy']:.0%}")
        return

    # classify
    chosen = [args.report is not None, args.file is not None, args.dir is not None, args.interactive]
    if sum(chosen) != 1:
        parser.error("give exactly one of: a report text, --file, --dir, or --interactive")
    if args.report is not None and not args.report.strip():
        parser.error("the report text is empty")

    try:
        classifier = core.load_classifier(args.model_dir)
        if args.interactive:
            _run_interactive(classifier)
            return
        if args.report is not None:
            category = classifier.classify(args.report)
            if args.format == "text":
                print(category)
            else:
                print(core.format_results([{"report": args.report, "category": category}], args.format))
            return
        if args.file is not None:
            results = core.classify_file(classifier, args.file)
        else:
            results = core.classify_directory(classifier, args.dir)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

    print(core.format_results(results, args.format))


if __name__ == "__main__":
    main()
