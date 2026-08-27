from pathlib import Path

import polars as pl

from naturalv2.sources.reddit.stages.curate import (
    RedditCurateStage,
    _aggregate_reports_by_author,
    _build_report_expr,
)


def test_build_report_distinguishes_author_text_from_context():
    records = pl.DataFrame(
        {
            "subreddit": ["health", "health"],
            "title": ["My treatment", "Another person's question"],
            "initial_post": ["", "Context author's experience"],
            "report_text": ["My submission body", "My comment"],
            "report_type": ["submission", "comment"],
            "date_created": ["January 01, 2024", "January 02, 2024"],
            "permalink": ["/r/health/post", "/r/health/post/_/comment"],
        }
    )

    reports = records.with_columns(_build_report_expr(records.columns).alias("report"))[
        "report"
    ].to_list()

    assert "**Author's own text**\nTitle: My treatment" in reports[0]
    assert "My submission body" in reports[0]
    assert "Context written by another Reddit user" not in reports[0]

    assert "**Context written by another Reddit user**" in reports[1]
    assert "Original post title: Another person's question" in reports[1]
    assert "Context author's experience" in reports[1]
    assert "**Author's own text**\nComment:\nMy comment" in reports[1]
    assert reports[1].index("Context author's experience") < reports[1].index(
        "My comment"
    )


def test_aggregate_reports_by_author_preserves_order_and_unkeyed_records():
    records = pl.DataFrame(
        {
            "author_key": ["author-a", "author-a", "author-a", None, None],
            "date_created": [
                "January 02, 2024",
                "January 01, 2024",
                "January 01, 2024",
                "January 03, 2024",
                "January 04, 2024",
            ],
            "permalink": ["/later", "/earlier", "/earlier", "/one", "/two"],
            "subreddit": ["b", "a", "a", "c", "d"],
            "report": [
                "later author text",
                "earlier author text",
                "duplicate author text",
                "first unkeyed text",
                "second unkeyed text",
            ],
            "author_replies": [[], ["reply"], ["reply"], [], []],
            "treatments_mentioned": [
                ["treatment-b"],
                ["treatment-a"],
                ["treatment-a"],
                ["treatment-c"],
                ["treatment-d"],
            ],
        }
    )

    aggregated = _aggregate_reports_by_author(records)

    assert len(aggregated) == 3
    author = aggregated.filter(pl.col("author_key") == "author-a").row(0, named=True)
    assert author["source_record_count"] == 2
    assert author["source_permalinks"] == ["/earlier", "/later"]
    assert author["source_dates"] == ["January 01, 2024", "January 02, 2024"]
    assert author["source_subreddits"] == ["a", "b"]
    assert author["author_replies"] == ["reply"]
    assert author["treatments_mentioned"] == ["treatment-a", "treatment-b"]
    assert "Combined Reddit records from one pseudonymous author" in author["report"]
    assert author["report"].index("earlier author text") < author["report"].index(
        "later author text"
    )
    assert "duplicate author text" not in author["report"]

    unkeyed = aggregated.filter(pl.col("author_key").is_null())
    assert len(unkeyed) == 2
    assert unkeyed["source_record_count"].to_list() == [1, 1]
    assert all(
        "Combined Reddit records from one pseudonymous author" not in report
        for report in unkeyed["report"]
    )


def test_consolidation_combines_author_records_across_worker_files(tmp_path: Path):
    nct_id = "NCT00000001"
    temp_root = tmp_path / "temp"
    first_worker = temp_root / "worker-a"
    second_worker = temp_root / "worker-b"
    first_worker.mkdir(parents=True)
    second_worker.mkdir(parents=True)

    pl.DataFrame(
        {
            "author_key": ["author-a"],
            "date_created": ["January 02, 2024"],
            "permalink": ["/later"],
            "subreddit": ["health"],
            "report": ["later author text"],
        }
    ).write_parquet(first_worker / f"{nct_id}_first.parquet")
    pl.DataFrame(
        {
            "author_key": ["author-a"],
            "date_created": ["January 01, 2024"],
            "permalink": ["/earlier"],
            "subreddit": ["health"],
            "report": ["earlier author text"],
        }
    ).write_parquet(second_worker / f"{nct_id}_second.parquet")

    target_path = tmp_path / "curated"
    target_path.mkdir()
    (target_path / "stale.parquet").write_bytes(b"stale")

    count = RedditCurateStage._consolidate_parquet_chunks(
        str(temp_root), nct_id, str(target_path)
    )

    consolidated = pl.read_parquet(target_path)
    assert count == 1
    assert len(consolidated) == 1
    assert consolidated["source_record_count"].item() == 2
    assert consolidated["source_permalinks"].to_list()[0] == ["/earlier", "/later"]
    assert not (target_path / "stale.parquet").exists()
