import json
from enum import StrEnum

import z3
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from llm_logic_integration.solvers.solver import Solver
from llm_logic_integration.system_prompts.system_prompts import SystemPrompt
from llm_logic_integration.utils.llm_factory import create_llm


class EvaluationStatus(StrEnum):
    OK = "OK"
    FAILURE = "FAILURE"


class LogicTranslatorOutput(BaseModel):
    variables: list[str] = Field(
        description="List of boolean variable names in lowercase with underscores"
    )
    premises: list[str] = Field(
        description="List of valid Z3 Python expressions as strings"
    )
    goal: str = Field(description="The final Z3 expression to prove")


class LogicEvaluatorOutput(BaseModel):
    status: EvaluationStatus = Field(description="The evaluation status.")
    reasoning: str = Field(description="Explanation of the logical check.")
    feedback: str = Field(
        description="Specific logical corrections for the Generator if it failed, else 'None'."
    )


class LogicVerifierAgent:
    def __init__(
        self,
        provider: str,
        model_name: str,
        solver: Solver,
        api_key: str | None = None,
        max_retries: int = 3,
    ):
        self.model = create_llm(
            provider, model_name, api_key, temperature=0.0, max_tokens=800
        )
        self.solver = solver
        self.max_retries = max_retries

        self.translator_model = self.model.with_structured_output(LogicTranslatorOutput)
        self.evaluator_model = self.model.with_structured_output(LogicEvaluatorOutput)

        self.translator_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SystemPrompt.VERIFIER_TRANSLATOR_PROMPT.value),
                (
                    "human",
                    "ORIGINAL TEXT: {original_sentence}\nGENERATOR'S ANSWER: {generator_answer}\nFEEDBACK: {feedback}",
                ),
            ]
        )
        self.translator_chain = self.translator_prompt | self.translator_model

        self.evaluator_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SystemPrompt.VERIFIER_EVALUATOR_PROMPT.value),
                (
                    "human",
                    "GENERATOR'S ANSWER: {generator_answer}\nSOLVER STATUS: {solver_status}",
                ),
            ]
        )
        self.evaluator_chain = self.evaluator_prompt | self.evaluator_model

    def _clean_code_string(self, s: str) -> str:
        if not isinstance(s, str):
            return str(s)

        s = s.strip()

        if s.startswith("```"):
            lines = s.split("\n")
            if len(lines) > 1:
                lines = lines[1:]
            if len(lines) > 0 and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()

        s = s.replace("`", "")

        if s.lower().startswith("python"):
            s = s[6:].strip()

        return s

    def _evaluate_z3_strings(self, translation: LogicTranslatorOutput):
        class DynamicZ3Env(dict):
            def __getitem__(self, key):
                if key.startswith("__"):
                    raise KeyError(key)
                if key not in self:
                    self[key] = z3.Bool(key)
                return super().__getitem__(key)

        local_env = DynamicZ3Env()
        local_env["z3"] = z3

        for var in translation.variables:
            clean_var = var.strip().replace(" ", "_")
            local_env[clean_var] = z3.Bool(clean_var)

        try:
            premises = []
            for p in translation.premises:
                clean_p = self._clean_code_string(p)
                if clean_p:
                    premises.append(eval(clean_p, {"__builtins__": {}}, local_env))

            clean_goal = self._clean_code_string(translation.goal)
            goal = eval(clean_goal, {"__builtins__": {}}, local_env)

            return premises, goal, None

        except SyntaxError as e:
            return None, None, f"Syntax error in Z3 expression: {e}"
        except Exception as e:
            return None, None, str(e)

    def verify(
        self, original_sentence: str, generator_answer: str | BaseModel
    ) -> LogicEvaluatorOutput:

        if hasattr(generator_answer, "model_dump_json"):
            generator_answer = generator_answer.model_dump_json()  # ty:ignore[call-non-callable]
        elif not isinstance(generator_answer, str):
            generator_answer = str(generator_answer)

        feedback = "None"

        for i in range(self.max_retries):
            logger.info(f"[Logic Verifier] Translation Attempt {i + 1}")

            try:
                translation: LogicTranslatorOutput = self.translator_chain.invoke(
                    {
                        "original_sentence": original_sentence,
                        "generator_answer": generator_answer,
                        "feedback": feedback,
                    }
                )  # ty:ignore[invalid-assignment]
            except Exception as e:
                logger.error(f"[Logic Verifier] LLM Execution Failed: {e}")
                feedback = f"Chain failure: {e}. Try a simpler output."
                continue

            premises, goal, error = self._evaluate_z3_strings(translation)

            if error:
                logger.warning(
                    f"[Logic Verifier] Python Evaluation Error: {error}. Retrying."
                )
                feedback = f"Python evaluation failed: {error}. Ensure you are only using valid Z3 Python syntax and strictly NO markdown backticks."
                continue

            self.solver.set_premises(premises)
            self.solver.set_goal(goal)
            solver_status = self.solver.return_status()
            logger.info(f"[Logic Verifier] Solver status: {solver_status['type']}")

            logger.info("[Logic Verifier] Syntax OK, evaluating logical alignment.")
            try:
                evaluation = self.evaluator_chain.invoke(
                    {
                        "generator_answer": generator_answer,
                        "solver_status": json.dumps(solver_status, indent=2),
                    }
                )
                return evaluation  # ty:ignore[invalid-return-type]
            except Exception as e:
                logger.error(f"[Logic Verifier] Evaluator LLM Failed: {e}")
                return LogicEvaluatorOutput(
                    status=EvaluationStatus.FAILURE,
                    reasoning=f"Evaluator agent crashed: {e}",
                    feedback="Internal verifier error.",
                )

        logger.error("[Logic Verifier] Failed to generate valid logic syntax.")
        return LogicEvaluatorOutput(
            status=EvaluationStatus.FAILURE,
            reasoning="Could not translate to valid Z3 logic.",
            feedback="The logical constraints of the problem are too complex to parse. Try simplifying the answer.",
        )
