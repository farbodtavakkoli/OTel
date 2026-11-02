"""Create a synthetic fixture, fetch the original smoke CSV, or audit choice JSONL."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from decider_recipe_utils import load_rows, require

OPTIONS = [
    ["doc_direct", "Original document text or a direct document excerpt."],
    ["key_facts", "A list of extracted key facts, possibly with explanations."],
    [
        "extractive_summary",
        "A summary composed of quotations or extracts from a passage.",
    ],
    ["detailed_summary", "A detailed synthesized or conceptual summary of a document."],
]
OTEL_URL = (
    "https://raw.githubusercontent.com/farbodtavakkoli/OTel/main/"
    "training/classification/deepspeed/data/classification_sample.csv"
)
OTEL_SHA = "94ad9da34e2d11f34e65d1513c1e962c871d50b105bb3b994e933b77d49c4202"


def example(text, label):
    return dict(
        kind="choice",
        state=text,
        instructions="Which document representation best describes this text?",
        options=OPTIONS,
        label=label,
        task="document_representation",
    )


def download_otel():
    """Fetch a bounded body using one retry budget for headers and body reads."""
    try:
        import urllib3
        from urllib3.exceptions import HTTPError, ProtocolError, ReadTimeoutError
        from urllib3.util import Retry, Timeout
    except ImportError:
        raise SystemExit(
            "OTel download requires urllib3. Run prepare_decider_dataset.py with the application "
            "venv created by setup_trainium_environment.py."
        ) from None

    max_attempts = 4
    retries = Retry(
        total=max_attempts - 1,
        connect=max_attempts - 1,
        read=max_attempts - 1,
        status=max_attempts - 1,
        redirect=0,
        other=0,
        allowed_methods=frozenset({"GET"}),
        status_forcelist=(429, *range(500, 600)),
        backoff_factor=1,
        backoff_jitter=0.5,
        backoff_max=8,
        # Keep delays bounded and avoid implicit retries of HTTP 413 when it
        # carries Retry-After. Only the explicit status list is retried.
        respect_retry_after_header=False,
    )
    try:
        with urllib3.PoolManager() as http:
            while True:
                try:
                    response = http.request(
                        "GET",
                        OTEL_URL,
                        retries=retries,
                        timeout=Timeout(connect=10, read=60),
                        redirect=False,
                        preload_content=False,
                        decode_content=False,
                        headers={"Accept-Encoding": "identity"},
                    )
                    # Carry forward retries already consumed by the adapter.
                    retries = response.retries
                    try:
                        if response.status != 200:
                            raise SystemExit(
                                f"Failed to download OTel CSV from {OTEL_URL}: "
                                f"HTTP {response.status}; not retried. "
                                "Check source availability and access."
                            )
                        return response.read(1024 * 1024 + 1, decode_content=False)
                    finally:
                        response.close()
                        response.release_conn()
                except (ReadTimeoutError, ProtocolError) as exc:
                    # Streaming reads happen outside the adapter's retry loop.
                    # Its updated Retry object prevents a second, larger budget.
                    retries = retries.increment(method="GET", url=OTEL_URL, error=exc)
                    retries.sleep()
    except (HTTPError, OSError) as exc:
        detail = str(exc) or type(exc).__name__
        raise SystemExit(
            f"Failed to download OTel CSV from {OTEL_URL} "
            f"(maximum {max_attempts} GET attempts): {detail}. "
            "Check network connectivity, proxy/TLS settings and source "
            "availability, then retry."
        ) from None


def main(args):
    if args.mode == "audit":
        rows = load_rows(args.source)
        print(
            json.dumps(
                dict(
                    rows=len(rows),
                    classes=sorted({r["options"][r["label"]][0] for r in rows}),
                )
            )
        )
        return
    require(args.output is not None, "--output is required")
    if args.mode == "synthetic":
        texts = [
            (
                "The garden gate opens at sunrise. Visitors follow the marked path to the greenhouse.",
                0,
            ),
            (
                "Facts: the garden opens at sunrise; visitors use a marked path; the greenhouse is accessible.",
                1,
            ),
            ('Selected passage: "The garden gate opens at sunrise."', 2),
            (
                "The passage describes a predictable morning opening routine and a clear route for visitors, "
                "emphasizing orderly access to the greenhouse.",
                3,
            ),
            ("The library loans three books to each member for fourteen days.", 0),
            ("Key facts: three books per member; loan duration is fourteen days.", 1),
            ('Excerpt: "The library loans three books" and "for fourteen days."', 2),
            (
                "The library policy combines a modest borrowing limit with a two-week period, "
                "balancing reader access and circulation.",
                3,
            ),
            ("A blue sign marks the bicycle parking area beside the station.", 0),
            ("Facts: bicycle parking is beside the station; its sign is blue.", 1),
        ]
        rows = [example(*x) for x in texts]
    else:
        content = download_otel()
        require(
            len(content) <= 1024 * 1024
            and hashlib.sha256(content).hexdigest() == OTEL_SHA,
            "OTel dataset checksum mismatch; refusing changed source",
        )
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
        require(reader.fieldnames == ["text", "label"], "Unexpected CSV columns")
        labels = [o[0] for o in OPTIONS]
        rows = [example(r["text"], labels.index(r["label"])) for r in reader]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        out.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    load_rows(args.output)
    print(json.dumps(dict(rows=len(rows), source=args.mode, quality_evaluation=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["synthetic", "otel", "audit"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.mode == "audit" and args.source is None:
        parser.error("audit requires --source")
    main(args)
