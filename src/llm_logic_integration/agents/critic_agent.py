from enum import StrEnum

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_logic_integration.system_prompts.system_prompts import SystemPrompt
from llm_logic_integration.utils.llm_factory import create_llm


class VerificationStatus(StrEnum):
    OK = "OK"
    FAILURE = "FAILURE"


class CriticOutput(BaseModel):
    status: VerificationStatus = Field(description="The evaluation status.")
    reasoning: str = Field(description="Explanation of your heuristic analysis.")
    feedback: str = Field(
        description="Specific instructions to fix the answer if it failed, else 'None'"
    )


class CriticAgent:
    def __init__(self, provider: str, model_name: str, api_key: str | None = None):
        self.model = create_llm(provider, model_name, api_key, temperature=0.1)

        self.structured_model = self.model.with_structured_output(CriticOutput)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SystemPrompt.CRITIC_PROMPT.value),
                (
                    "human",
                    "ORIGINAL TEXT: {original_sentence}\nGENERATOR'S ANSWER: {generator_answer}\n LOGIC VERIFIER'S ANSWER: {verifier_answer}",
                ),
            ]
        )
        self.chain = self.prompt | self.structured_model

    def evaluate(
        self, original_sentence: str, generator_answer: str, verifier_answer: str
    ) -> CriticOutput:
        if hasattr(generator_answer, "model_dump_json"):
            generator_answer = generator_answer.model_dump_json()  # ty:ignore[call-non-callable]
        elif not isinstance(generator_answer, str):
            generator_answer = str(generator_answer)

        return self.chain.invoke(
            {
                "original_sentence": original_sentence,
                "generator_answer": generator_answer,
                "verifier_answer": verifier_answer,
            }
        )  # ty:ignore[invalid-return-type]
