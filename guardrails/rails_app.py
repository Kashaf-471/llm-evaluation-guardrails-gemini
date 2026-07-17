import os
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig
from langchain_google_vertexai import ChatVertexAI

load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

config = RailsConfig.from_path("./guardrails")

vertex_llm = ChatVertexAI(
    model_name="gemini-2.5-flash",
    project=GCP_PROJECT_ID,
    location=GCP_LOCATION,
    temperature=0,
)

# Pass the LangChain LLM directly to the constructor — this is the
# supported path for custom/non-default LLMs, not register_action_param.
rails = LLMRails(config, llm=vertex_llm)


def guarded_invoke(agent, user_message: str) -> str:
    """Sanitize input, run the deepagents agent, sanitize output."""
    input_check = rails.generate(
        messages=[{"role": "user", "content": user_message}]
    )
    if isinstance(input_check, dict) and input_check.get("blocked"):
        return "Request blocked by input guardrail."

    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    answer_text = ""
    for msg in result.get("messages", []):
        if getattr(msg, "text", None):
            answer_text = msg.text

    output_check = rails.generate(
        messages=[{"role": "assistant", "content": answer_text}]
    )
    if isinstance(output_check, dict) and output_check.get("blocked"):
        return "Response blocked by output guardrail."

    return answer_text