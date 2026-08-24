import polars as pl
import pyarrow as pa
import pytest

from naturalv2.sources.reddit.processing import _utils as utils
from naturalv2.sources.reddit.processing import contextualize as ctx
from naturalv2.sources.reddit.processing import filter as pfilter
from naturalv2.sources.reddit.processing._utils import bucket_from_subreddit
from naturalv2.sources.reddit.stages.curate import RedditCurateStage


def test_apply_rule_based_filter_flags_common_cases():
    table = pa.table(
        {
            "body": pa.array(
                [
                    "Fish &amp; chips with sauce",  # valid text, HTML unescapes
                    "[deleted]",  # sentinel
                    "Real content",  # bot author blocked
                    "www.example.com",  # URL-only
                    None,  # null text
                ],
                type=pa.string(),
            ),
            "author": pa.array(
                ["user", "user", "bot_123", "user", "AutoModerator"], type=pa.string()
            ),
        }
    )

    mask = pfilter.apply_rule_based_filter(table, "body").combine_chunks()

    values = mask.to_pylist()

    assert values[0] is True
    assert values[1] is False  # sentinel
    assert values[2] is True  # bot-like author
    assert not bool(values[3])  # URL-only text should be filtered
    assert values[4] is False  # AutoModerator blocked


def test_scan_reddit_chunks_filters_and_limits_columns(tmp_path):
    test_subreddit = "TestSub"
    bucket = bucket_from_subreddit(pa.array([test_subreddit])).to_pylist()[0]
    parquet_dir = tmp_path / "content_type=submissions" / f"bucket={bucket}"
    parquet_dir.mkdir(parents=True)
    file_path = parquet_dir / "sample.parquet"
    pl.DataFrame(
        {
            "subreddit": ["TestSub", "Other"],
            "title": ["keep", "drop"],
            "report_text": ["body", "ignored"],
            "score": [1, 2],
        }
    ).write_parquet(file_path)

    batches = list(
        pfilter.scan_reddit_dataset(
            [file_path.as_posix()],
            columns=["subreddit", "title", "report_text", "score", "missing"],
            subreddit=[test_subreddit],
            batch_size=1,
        )
    )

    assert len(batches) == 1
    batch = batches[0]
    assert batch.shape == (1, 4)
    assert batch["subreddit"].to_list() == ["TestSub"]
    assert batch["title"].to_list() == ["keep"]
    assert set(batch.columns) == {"subreddit", "title", "report_text", "score"}


def test_write_to_parquet_partitions_creates_hive_layout(tmp_path):
    schema = ctx.CONTEXTUALIZED_RECORD_SCHEMA
    bucket = utils.bucket_from_subreddit(pa.array(["testsub"])).to_pylist()[0]
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(["testsub"]),
            pa.array(["title"]),
            pa.array(["body"]),
            pa.array(["report"]),
            pa.array(["submission"]),
            pa.array([1], type=pa.int64()),
            pa.array(["2024-01-01T00:00:00Z"]),
            pa.array([""]),
            pa.array([ctx._pseudonymize_author("test-author")]),
            pa.array([["reply"]], type=pa.list_(pa.string())),
            pa.array(["submissions"]),
            pa.array([bucket]),
        ],
        names=[field.name for field in schema],
    )

    written = ctx.write_to_parquet_partitions(
        data_stream=[batch],
        output_dir=tmp_path.as_posix(),
        schema=schema,
        run_tag="unit",
        max_partitions=8,
        min_rows_per_group=1,
        max_rows_per_group=2,
        max_open_files=8,
    )

    assert len(written) == 1
    assert "content_type=submissions" in written[0]
    assert f"bucket={bucket}" in written[0]
    assert tmp_path.joinpath("content_type=submissions", f"bucket={bucket}").exists()
    assert tmp_path.joinpath(
        "content_type=submissions", f"bucket={bucket}", "unit-part-0.parquet"
    ).exists()


def test_write_to_parquet_partitions_validates_args(tmp_path):
    schema = ctx.CONTEXTUALIZED_RECORD_SCHEMA
    with pytest.raises(ValueError):
        ctx.write_to_parquet_partitions(
            data_stream=[],
            output_dir=tmp_path.as_posix(),
            schema=schema,
            parquet_compression_level=0,
        )


def test_contextualization_preserves_pseudonymous_author_keys(tmp_path):
    submissions_path = tmp_path / "submissions.parquet"
    comments_path = tmp_path / "comments.parquet"
    output_path = tmp_path / "contextualized"

    pl.DataFrame(
        {
            "id": ["post-1", "post-2"],
            "created_utc": [1_700_000_000, 1_700_000_100],
            "subreddit": ["testsub", "testsub"],
            "title": ["First post", "Second post"],
            "selftext": ["First body", "Second body"],
            "author": ["OriginalPoster", "AnotherPoster"],
            "score": [1.0, 2.0],
        }
    ).write_parquet(submissions_path)
    pl.DataFrame(
        {
            "id": ["reply-1", "comment-1", "comment-2", "comment-3"],
            "link_id": ["t3_post-1", "t3_post-1", "t3_post-2", "t3_post-2"],
            "created_utc": [
                1_700_000_010,
                1_700_000_020,
                1_700_000_110,
                1_700_000_120,
            ],
            "subreddit": ["testsub"] * 4,
            "body": [
                "Original poster reply",
                "First commenter record",
                "Second commenter record",
                "Deleted account record",
            ],
            "author": ["OriginalPoster", "Commenter", "COMMENTER", "[deleted]"],
            "score": [1.0, 2.0, 3.0, 4.0],
        }
    ).write_parquet(comments_path)

    submission_output, comment_output = ctx._process_bucket(
        "bucket-1",
        {
            "submissions": [str(submissions_path)],
            "comments": [str(comments_path)],
        },
        output_path,
        "unit",
    )

    assert submission_output is not None
    assert comment_output is not None
    submissions = pl.read_parquet(submission_output)
    comments = pl.read_parquet(comment_output)
    assert "author" not in submissions.columns
    assert "author" not in comments.columns
    assert submissions.filter(pl.col("title") == "First post").item(
        0, "author_key"
    ) == ctx._pseudonymize_author("originalposter")

    commenter_keys = comments.filter(
        pl.col("report_text").str.contains("commenter record")
    )["author_key"]
    assert commenter_keys.n_unique() == 1
    assert commenter_keys[0] == ctx._pseudonymize_author("Commenter")
    assert commenter_keys[0] != "Commenter"
    assert (
        comments.filter(pl.col("report_text") == "Deleted account record").item(
            0, "author_key"
        )
        is None
    )


def test_curation_requests_pseudonymous_author_key():
    stage = RedditCurateStage(num_workers=1)

    assert "author_key" in stage._curation_columns
