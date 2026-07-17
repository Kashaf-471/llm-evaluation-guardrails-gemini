import os
import time
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from main import agent, RETRIEVED_CONTEXTS

load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

judge_llm = ChatVertexAI(
    model_name="gemini-2.5-flash",
    project=GCP_PROJECT_ID,
    location=GCP_LOCATION,
    temperature=0,
)
judge_embeddings = VertexAIEmbeddings(
    model_name="text-embedding-005",
    project=GCP_PROJECT_ID,
    location=GCP_LOCATION,
)

ragas_llm = LangchainLLMWrapper(judge_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(judge_embeddings)

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
for m in METRICS:
    m.llm = ragas_llm
    if hasattr(m, "embeddings"):
        m.embeddings = ragas_embeddings


def build_eval_dataset(questions: list[dict]) -> Dataset:
    """questions: list of {"question": str, "ground_truth": str}"""
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in questions:
        RETRIEVED_CONTEXTS.clear()  # isolate this question's batches
        result = agent.invoke({"messages": [{"role": "user", "content": item["question"]}]})

        answer_text = ""
        for msg in result.get("messages", []):
            if getattr(msg, "text", None):
                answer_text = msg.text

        # flatten every chunk batch retrieved while answering this question
        contexts = [text for batch in RETRIEVED_CONTEXTS.values() for text in batch]

        rows["question"].append(item["question"])
        rows["answer"].append(answer_text)
        rows["contexts"].append(contexts or ["(no context retrieved)"])
        rows["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(rows)


def evaluate_with_retry(dataset: Dataset, max_attempts: int = 5, initial_delay: float = 30.0):
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return evaluate(dataset, metrics=METRICS)
        except Exception as e:
            msg = str(e)
            if ("RESOURCE_EXHAUSTED" in msg or "429" in msg) and attempt < max_attempts:
                print(f"Judge rate-limited; retrying in {delay:.0f}s (attempt {attempt})")
                time.sleep(delay)
                delay = min(delay * 2, 180.0)
                continue
            raise


if __name__ == "__main__":
    test_questions = [
    {
        "question": "How do I stream intermediate tool results from a subagent?",
        "ground_truth": (
            "Use agent.stream() with stream_mode=\"custom\" and subgraphs=True "
            "to enable custom event streaming from subagents. Subagent events "
            "are identified by checking if chunk[\"type\"] == \"custom\" and "
            "whether any namespace in chunk[\"ns\"] starts with \"tools:\"; the "
            "intermediate results are available in chunk[\"data\"]. "
            "Alternatively, use stream_events, which provides typed "
            "projections including separate iterators for subagents via "
            "stream.subagents, giving access to each subagent's .name, "
            ".messages, .tool_calls, and .output."
        ),
    },
    ]
ds = build_eval_dataset(test_questions)
result = evaluate_with_retry(ds)
print(result.to_pandas().to_string())