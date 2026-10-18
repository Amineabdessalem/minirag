from pathlib import Path

from minirag.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_ask_prints_ranked_hits(capsys):
    main(["ask", "--docs", str(EXAMPLES / "docs"), "-k", "2", "make container images smaller"])
    out = capsys.readouterr().out
    assert out.startswith("[1]")
    assert "docker.md" in out.splitlines()[0]


def test_ask_prompt_mode(capsys):
    main(["ask", "--docs", str(EXAMPLES / "docs"), "--prompt", "what is a pod?"])
    out = capsys.readouterr().out
    assert "Question: what is a pod?" in out
    assert "(kubernetes.md)" in out


def test_eval_compares_retrievers(capsys):
    main(
        [
            "eval",
            "--docs", str(EXAMPLES / "docs"),
            "--queries", str(EXAMPLES / "queries.jsonl"),
            "--compare", "bm25", "hybrid",
            "-k", "3",
        ]
    )
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines] == ["bm25", "hybrid"]
    assert all("queries=17" in line for line in lines)


def test_index_then_ask_from_saved_index(tmp_path, capsys):
    out = tmp_path / "idx"
    main(["index", "--docs", str(EXAMPLES / "docs"), "--out", str(out)])
    assert "saved 8 chunks" in capsys.readouterr().out
    main(["ask", "--index", str(out), "-k", "1", "what is a pod in kubernetes"])
    assert "kubernetes.md" in capsys.readouterr().out


def test_ask_with_doc_prefix(capsys):
    main(["ask", "--docs", str(EXAMPLES / "docs"), "--doc-prefix", "docker", "-k", "3", "containers"])
    out = capsys.readouterr().out
    doc_lines = [line for line in out.splitlines() if line.startswith("[")]
    assert doc_lines and all("docker.md" in line for line in doc_lines)
