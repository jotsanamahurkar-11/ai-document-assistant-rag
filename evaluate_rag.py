import json
from pathlib import Path

from rag import (
    extract_pages,
    create_chunks,
    create_embeddings,
    build_vector_index,
    search_chunks,
    generate_answer,
)


DATASET_PATH = Path("evaluation_cases.json")
SIMILARITY_THRESHOLD = 0.65


def load_rag_system(pdf_path):
    """Build the retrieval index for one PDF."""

    pages = extract_pages(str(pdf_path))

    if not pages:
        raise ValueError("No readable text found in the PDF.")

    chunks = create_chunks(pages)

    if not chunks:
        raise ValueError("No chunks were created.")

    chunks = create_embeddings(chunks)
    index = build_vector_index(chunks)

    return chunks, index


def answer_was_not_found(answer):
    """Recognize the not-found response used by our current RAG app."""

    answer = answer.casefold()

    phrases = [
        "could not find this information in the document",
        "could not find relevant information",
    ]

    return any(phrase in answer for phrase in phrases)


def evaluate_case(case, cached_systems):
    """Run retrieval and answer checks for one test case."""

    pdf_path = Path(case["pdf_path"])

    if not pdf_path.exists():
        return {
            "id": case["id"],
            "status": "SKIP",
            "reason": f"PDF not found: {pdf_path}",
        }

    try:
        # Build the index once per PDF and reuse it.
        pdf_key = str(pdf_path)

        if pdf_key not in cached_systems:
            cached_systems[pdf_key] = load_rag_system(pdf_path)

        chunks, index = cached_systems[pdf_key]

        question = case["question"]

        results = search_chunks(
            question=question,
            chunks=chunks,
            index=index,
            top_k=3,
            similarity_threshold=SIMILARITY_THRESHOLD,
        )

        answer = generate_answer(
            question=question,
            retrieved_chunks=results,
        )

        retrieved_text = "\n".join(
            result["text"] for result in results
        ).casefold()

        should_answer = case["should_answer"]
        expected_context = case.get("expected_context_contains")

        # Check retrieval separately from answer quality.
        if expected_context is None:
            retrieval_pass = None
        else:
            retrieval_pass = (
                expected_context.casefold() in retrieved_text
            )

        # Check the generated answer.
        if should_answer:
            expected_terms = case["expected_answer_contains"]

            answer_pass = (
                bool(expected_terms)
                and all(
                    term.casefold() in answer.casefold()
                    for term in expected_terms
                )
                and not answer_was_not_found(answer)
            )

        else:
            answer_pass = answer_was_not_found(answer)

        return {
            "id": case["id"],
            "status": "PASS" if answer_pass else "FAIL",
            "retrieval_pass": retrieval_pass,
            "answer_pass": answer_pass,
            "answer": answer,
            "retrieved_count": len(results),
        }

    except Exception as error:
        return {
            "id": case["id"],
            "status": "ERROR",
            "reason": str(error),
        }


def main():
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        return

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    cached_systems = {}
    completed = []

    print("\n========== RAG EVALUATION ==========\n")

    # Run all test cases
    for case in cases:
        print(f"Running {case['id']}: {case['question']}")

        result = evaluate_case(case, cached_systems)
        completed.append(result)

        print("Status:", result["status"])

        if result["status"] in ("SKIP", "ERROR"):
            print("Reason:", result["reason"])

        else:
            retrieval = result["retrieval_pass"]

            if retrieval is None:
                retrieval_text = "N/A"
            else:
                retrieval_text = (
                    "PASS" if retrieval else "FAIL"
                )

            print("Retrieval:", retrieval_text)

            print(
                "Answer check:",
                "PASS" if result["answer_pass"] else "FAIL"
            )

            print("Answer:", result["answer"])

        print("-" * 60)

    # Calculate summary metrics
    executed = [
        result for result in completed
        if result["status"] in ("PASS", "FAIL")
    ]

    retrieval_tests = [
        result for result in executed
        if result["retrieval_pass"] is not None
    ]

    answer_passes = sum(
        result["answer_pass"] for result in executed
    )

    retrieval_passes = sum(
        result["retrieval_pass"] for result in retrieval_tests
    )

    skipped = sum(
        result["status"] == "SKIP"
        for result in completed
    )

    errors = sum(
        result["status"] == "ERROR"
        for result in completed
    )

    # Print summary
    print("\n========== SUMMARY ==========")
    print("Total test cases:", len(cases))
    print("Executed:", len(executed))
    print("Skipped:", skipped)
    print("Errors:", errors)

    print(
        f"Answer checks passed: "
        f"{answer_passes}/{len(executed)}"
    )

    print(
        f"Retrieval checks passed: "
        f"{retrieval_passes}/{len(retrieval_tests)}"
    )

    # Prepare a detailed JSON report
    report = {
        "total_test_cases": len(cases),
        "executed": len(executed),
        "skipped": skipped,
        "errors": errors,
        "answer_checks_passed": answer_passes,
        "answer_checks_total": len(executed),
        "retrieval_checks_passed": retrieval_passes,
        "retrieval_checks_total": len(retrieval_tests),
        "results": completed,
    }

    # Save the report
    report_path = Path("evaluation_report.json")

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nDetailed report saved to: "
        f"{report_path.resolve()}"
    )


if __name__ == "__main__":
    main()