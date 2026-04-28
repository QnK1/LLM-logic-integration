import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from llm_logic_integration.solvers.solver import Solver
from llm_logic_integration.system_prompts.system_prompts import SystemPrompt
from llm_logic_integration.utils.llm_factory import create_llm


class LogicVerifierAgent:
    def __init__(
        self,
        provider: str,
        model_name: str,
        solver: Solver,
        api_key: str | None = None,
        max_retries: int = 3,
    ):
        self.model = create_llm(provider, model_name, api_key, temperature=0.0)
        self.solver = solver
        self.max_retries = max_retries

        self.translator_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SystemPrompt.VERIFIER_TRANSLATOR_PROMPT.value),
                (
                    "human",
                    "ORIGINAL TEXT: {original_sentence}\nGENERATOR'S ANSWER: {generator_answer}\nFEEDBACK: {feedback}",
                ),
            ]
        )
        self.translator_chain = self.translator_prompt | self.model | JsonOutputParser()

        self.evaluator_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SystemPrompt.VERIFIER_EVALUATOR_PROMPT.value),
                (
                    "human",
                    "GENERATOR'S ANSWER: {generator_answer}\nSOLVER STATUS: {solver_status}",
                ),
            ]
        )
        self.evaluator_chain = self.evaluator_prompt | self.model | JsonOutputParser()

    def verify(self, original_sentence: str, generator_answer: str) -> dict:
        feedback = "None"

        for i in range(self.max_retries):
            logger.info(f"[Logic Verifier] Translation Attempt {i + 1}")
            translation = self.translator_chain.invoke(
                {
                    "original_sentence": original_sentence,
                    "generator_answer": generator_answer,
                    "feedback": feedback,
                }
            )

            premises = translation.get("premises", [])
            goal = translation.get("goal", "")

            solver_status = self.solver.return_status(premises, goal)
            logger.info(f"[Logic Verifier] Solver status: {solver_status['status']}")

            if (
                solver_status["status"] == "FAILURE"
                and solver_status.get("error_type") == "SYNTAX_ERROR"
            ):
                logger.warning(
                    f"[Logic Verifier] Syntax error caught: {solver_status['message']}. Retrying translation."
                )
                feedback = solver_status["message"]
                continue

            logger.info("[Logic Verifier] Syntax OK, evaluating logical alignment.")
            evaluation = self.evaluator_chain.invoke(
                {
                    "generator_answer": generator_answer,
                    "solver_status": json.dumps(solver_status, indent=2),
                }
            )
            return evaluation

        logger.error("[Logic Verifier] Failed to generate valid logic syntax.")
        return {
            "status": "FAILURE",
            "reasoning": "Could not translate to valid formal logic.",
            "feedback": "The logical constraints of the problem are too complex to parse. Try simplifying the answer.",
        }
