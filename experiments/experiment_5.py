import os
import sys
from typing import Any, Literal

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from llm_logic_integration.utils.llm_factory import create_llm

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

PROVIDER = "ollama"
MODEL_NAME = "qwen2.5-coder"
ITERATIONS_PER_EXPERIMENT = 10


def get_api_key(provider: str) -> str | None:
    provider = provider.lower()
    if provider == "gemini":
        return str(os.getenv("GEMINI_API_KEY"))
    elif provider == "openai":
        return str(os.getenv("OPENAI_API_KEY"))
    return None


class UntypedOutput(BaseModel):
    extracted_data: dict[str, Any] = Field(
        description="Extract the organizational graph as a JSON dictionary."
    )


class Node(BaseModel):
    id: str = Field(description="Unique identifier for the entity.")
    name: str
    type: Literal["Person", "Department", "Role"]


class Edge(BaseModel):
    source: str = Field(description="ID of the source node.")
    target: str = Field(description="ID of the target node.")
    relation: Literal["MANAGES", "BELONGS_TO", "REPORTS_TO"]


class TypedGraphOutput(BaseModel):
    nodes: list[Node]
    edges: list[Edge]

    @model_validator(mode="after")
    def verify_relational_constraints(self) -> "TypedGraphOutput":
        node_ids = {node.id for node in self.nodes}
        node_type_map = {node.id: node.type for node in self.nodes}

        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(
                    f"Structural Error: Source node '{edge.source}' does not exist."
                )
            if edge.target not in node_ids:
                raise ValueError(
                    f"Structural Error: Target node '{edge.target}' does not exist."
                )

            src_type = node_type_map[edge.source]
            tgt_type = node_type_map[edge.target]

            if edge.relation == "BELONGS_TO" and tgt_type != "Department":
                raise ValueError(
                    f"Ontology Error: '{edge.source}' BELONGS_TO '{edge.target}', but '{edge.target}' is not a Department."
                )

            if edge.relation == "MANAGES" and src_type != "Person":
                raise ValueError(
                    f"Ontology Error: A {src_type} cannot MANAGE someone. Only a Person can MANAGE."
                )

        return self


def validate_untyped_output(data: dict) -> list[str]:
    errors = []
    if "nodes" not in data or "edges" not in data:
        return ["Missing root 'nodes' or 'edges' keys."]

    node_ids = set()
    for n in data.get("nodes", []):
        if "id" not in n:
            errors.append("Node missing 'id'.")
        else:
            node_ids.add(n["id"])

    for e in data.get("edges", []):
        if "source" not in e or "target" not in e:
            errors.append(f"Edge {e} missing source/target.")
            continue
        if e["source"] not in node_ids:
            errors.append(f"Dangling source pointer: {e['source']}")
        if e["target"] not in node_ids:
            errors.append(f"Dangling target pointer: {e['target']}")

    return errors


def main():
    load_dotenv()
    api_key = str(get_api_key(PROVIDER))

    experiments = {
        "Corporate Ontology Extraction": {
            "prompt": (
                "Extract the organizational graph from this text: "
                "Alice is the CEO. Bob and Charlie report to Alice. "
                "Bob manages the Engineering department. Charlie belongs to the Sales department. "
                "Dave reports to Bob."
                "\n\nFor System A (if applicable): Output a dict with 'nodes' (id, name, type) and 'edges' (source, target, relation). "
                "Valid types: Person, Department. Valid relations: MANAGES, REPORTS_TO, BELONGS_TO."
            )
        }
    }

    logger.info(
        f"Initializing for Ontology Experiment using {PROVIDER.upper()}: {MODEL_NAME}"
    )

    base_model = create_llm(PROVIDER, MODEL_NAME, api_key, temperature=0.2)

    system_a = base_model.with_structured_output(UntypedOutput)
    system_b = base_model.with_structured_output(TypedGraphOutput)

    stats = {
        "A: Untyped": {"runs": 0, "success": 0, "structural_errors": 0},
        "B: Typed + Constraints": {"runs": 0, "success": 0, "structural_errors": 0},
    }

    for exp_name, exp_data in experiments.items():
        sentence = exp_data["prompt"]

        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT E.5: {exp_name}")
        print(f"PROMPT: {sentence}")
        print(f"{'=' * 80}")

        print("\n[ RUNNING SYSTEM A: UNTYPED ]")
        for i in range(ITERATIONS_PER_EXPERIMENT):
            stats["A: Untyped"]["runs"] += 1
            try:
                result_a = system_a.invoke(sentence)
                errors = validate_untyped_output(result_a.extracted_data)  # ty:ignore[unresolved-attribute]

                if errors:
                    stats["A: Untyped"]["structural_errors"] += 1
                    print(f"  Run {i + 1} Failed: {errors[0]}")
                else:
                    stats["A: Untyped"]["success"] += 1
                    print(f"  Run {i + 1} Success.")
            except Exception as e:
                stats["A: Untyped"]["structural_errors"] += 1
                print(f"  Run {i + 1} Exception/Parse Error: {e}")

        print("\n[ RUNNING SYSTEM B: TYPED + CONSTRAINTS ]")
        for i in range(ITERATIONS_PER_EXPERIMENT):
            stats["B: Typed + Constraints"]["runs"] += 1
            try:
                result_b = system_b.invoke(sentence)
                stats["B: Typed + Constraints"]["success"] += 1

                print(f"  Run {i + 1} Success. Valid Graph Extracted:")
                for line in result_b.model_dump_json(indent=2).split("\n"):  # ty:ignore[unresolved-attribute]
                    print(f"    {line}")

            except Exception as e:
                stats["B: Typed + Constraints"]["structural_errors"] += 1
                error_msg = str(e).split("\n")[0]
                print(f"  Run {i + 1} Structural Constraint Caught Error: {error_msg}")

    print(f"\n{'=' * 80}")
    print("E.5 EXPERIMENT RESULTS: TYPING AND ONTOLOGY")
    print(f"{'=' * 80}")

    for sys_name, sys_stats in stats.items():
        runs = sys_stats["runs"]
        success = sys_stats["success"]
        errors = sys_stats["structural_errors"]
        acc = (success / runs) * 100 if runs > 0 else 0

        print(f"\n{sys_name}")
        print(f"  Total Runs:        {runs}")
        print(f"  Valid Structures:  {success} ({acc:.1f}%)")
        print(f"  Structural Errors: {errors}")


if __name__ == "__main__":
    main()
