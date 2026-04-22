import pytest
from src.solvers.nltk_solver import NLTKSolver


def test_modus_ponens():
    """Test standard valid logical inference."""
    solver = NLTKSolver()
    premises = ["P(x) -> Q(x)", "P(x)"]
    goal = "Q(x)"

    status = solver.return_status(premises, goal)

    assert status["status"] == "SUCCESS"
    assert status["result"] == "TRUE"


def test_invalid_inference():
    """Test a case where the goal does not follow from the premises."""
    solver = NLTKSolver()
    premises = ["P(x) -> Q(x)"]
    goal = "Q(x)"

    status = solver.return_status(premises, goal)

    assert status["status"] == "SUCCESS"
    assert status["result"] == "UNKNOWN"


def test_first_order_logic():
    """Test simple predicate logic with quantifiers."""
    solver = NLTKSolver()
    premises = [
        "all x.(human(x) -> mortal(x))",
        "human(Socrates)"
    ]
    goal = "mortal(Socrates)"

    status = solver.return_status(premises, goal)

    assert status["status"] == "SUCCESS"
    assert status["result"] == "TRUE"


def test_syntax_error_handling():
    """Ensure the solver catches malformed FOL strings."""
    solver = NLTKSolver()
    premises = ["all x human(x)"]  # missing dot
    goal = "human(x)"

    status = solver.return_status(premises, goal)

    assert status["status"] == "FAILURE"
    assert status["error_type"] == "SYNTAX_ERROR"
    assert "message" in status