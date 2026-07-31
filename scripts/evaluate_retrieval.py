import json
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.rag_pipeline import RAGPipeline


def load_eval_questions():
    eval_path = PROJECT_ROOT / "tests" / "eval_questions.json"

    if not eval_path.exists():
        raise FileNotFoundError(f"评测文件不存在：{eval_path}")

    with eval_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_source_from_doc(doc):
    metadata = doc.metadata or {}

    return (
        metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("filename")
        or "未知来源"
    )


def evaluate_case(rag: RAGPipeline, case: dict):
    question = case["question"]
    expected_source = case["expected_source"]
    expected_keywords = case.get("expected_keywords", [])

    docs = rag.retrieve(question)

    retrieved_sources = [get_source_from_doc(doc) for doc in docs]
    retrieved_text = "\n".join(doc.page_content for doc in docs)

    source_hit = expected_source in retrieved_sources

    keyword_hits = [
        keyword
        for keyword in expected_keywords
        if keyword in retrieved_text
    ]

    keyword_hit = len(keyword_hits) > 0 if expected_keywords else True

    return {
        "id": case["id"],
        "question": question,
        "expected_source": expected_source,
        "retrieved_sources": retrieved_sources,
        "expected_keywords": expected_keywords,
        "keyword_hits": keyword_hits,
        "source_hit": source_hit,
        "keyword_hit": keyword_hit,
        "doc_count": len(docs),
    }


def print_case_result(result: dict):
    status = "通过" if result["source_hit"] and result["keyword_hit"] else "未通过"

    print("-" * 60)
    print(f"用例编号：{result['id']}")
    print(f"问题：{result['question']}")
    print(f"期望来源：{result['expected_source']}")
    print(f"召回来源：{result['retrieved_sources']}")
    print(f"期望关键词：{result['expected_keywords']}")
    print(f"命中关键词：{result['keyword_hits']}")
    print(f"召回文档数：{result['doc_count']}")
    print(f"来源命中：{result['source_hit']}")
    print(f"关键词命中：{result['keyword_hit']}")
    print(f"结果：{status}")


def main():
    load_dotenv()

    eval_cases = load_eval_questions()
    rag = RAGPipeline()

    results = []

    for case in eval_cases:
        result = evaluate_case(rag, case)
        results.append(result)
        print_case_result(result)

    total = len(results)
    source_hit_count = sum(1 for r in results if r["source_hit"])
    keyword_hit_count = sum(1 for r in results if r["keyword_hit"])
    passed_count = sum(
        1 for r in results
        if r["source_hit"] and r["keyword_hit"]
    )

    print("=" * 60)
    print("检索评测汇总")
    print(f"总问题数：{total}")
    print(f"通过数量：{passed_count}/{total}")
    print(f"来源命中：{source_hit_count}/{total}")
    print(f"关键词命中：{keyword_hit_count}/{total}")
    print(f"通过率：{passed_count / total:.2%}")
    print(f"来源命中率：{source_hit_count / total:.2%}")
    print(f"关键词命中率：{keyword_hit_count / total:.2%}")


if __name__ == "__main__":
    main()