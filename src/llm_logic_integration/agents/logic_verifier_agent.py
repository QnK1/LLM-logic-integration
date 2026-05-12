import json

import z3
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

    def _evaluate_z3_strings(self, translation: dict):
        """Safely evaluates the LLM's string outputs into actual Z3 objects."""
        local_env = {"z3": z3}

        for var in translation.get("variables", []):
            local_env[var] = z3.Bool(var)

        try:
            premises = [
                eval(p, {"__builtins__": {}}, local_env)
                for p in translation.get("premises", [])
            ]
            goal = eval(
                translation.get("goal", "True"), {"__builtins__": {}}, local_env
            )
            return premises, goal, None
        except Exception as e:
            return None, None, str(e)

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

            # Convert strings to Z3 objects
            premises, goal, error = self._evaluate_z3_strings(translation)

            if error:
                logger.warning(
                    f"[Logic Verifier] Python Evaluation Error: {error}. Retrying."
                )
                feedback = f"Python evaluation failed: {error}. Ensure you are only using valid Z3 Python syntax."
                continue

            self.solver.set_premises(premises)
            self.solver.set_goal(goal)
            solver_status = self.solver.return_status()
            logger.info(f"[Logic Verifier] Solver status: {solver_status['type']}")

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
            "reasoning": "Could not translate to valid Z3 logic.",
            "feedback": "The logical constraints of the problem are too complex to parse. Try simplifying the answer.",
        }
