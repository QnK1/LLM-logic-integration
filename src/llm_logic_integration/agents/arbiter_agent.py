import json

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_logic_integration.system_prompts.system_prompts import SystemPrompt
from llm_logic_integration.utils.llm_factory import create_llm


class ArbiterOutput(BaseModel):
    final_answer: str = Field(
        description="The definitive, final conclusion based on the agents' work."
    )


class ArbiterAgent:
    def __init__(self, provider: str, model_name: str, api_key: str | None = None):
        self.model = create_llm(provider, model_name, api_key, temperature=0.0)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SystemPrompt.ARBITER_PROMPT.value),
                (
                    "human",
                    "ORIGINAL TEXT: {original_sentence}\nANSWER DATA: {approved_data}",
                ),
            ]
        )

    def decide(
        self,
        original_sentence: str,
        approved_data: dict,
        output_schema: type[BaseModel] | None = None,
    ) -> BaseModel:
        if hasattr(approved_data, "model_dump_json"):
            approved_data = approved_data.model_dump_json()  # ty:ignore[call-non-callable]
        elif not isinstance(approved_data, str):
            approved_data = str(approved_data)  # ty:ignore[invalid-assignment]

        schema = output_schema if output_schema is not None else ArbiterOutput
        chain = self.prompt | self.model.with_structured_output(schema)

        return chain.invoke(
            {
                "original_sentence": original_sentence,
                "approved_data": json.dumps(approved_data, indent=2),
            }
        )  # ty:ignore[invalid-return-type]
