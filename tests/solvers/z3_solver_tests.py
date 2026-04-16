import z3
import pytest
from src.solvers.z3_solver import Z3Solver


def test_modus_ponens():
    """Test standard valid logical inference."""
    solver = Z3Solver()
    p, q = z3.Bools('p q')

    # Premises: p -> q, p
    # Goal: q
    solver.set_premises([z3.Implies(p, q), p])
    solver.set_goal(q)

    assert solver.prove_goal() is True
    feedback = solver.return_status()
    assert feedback["type"] == "SUCCESS"


def test_invalid_inference():
    """Test a case where the goal does not follow from the premises."""
    solver = Z3Solver()
    p, q = z3.Bools('p q')

    # Premises: p -> q
    # Goal: q (This is the 'affirming the consequent' fallacy)
    solver.set_premises([z3.Implies(p, q)])
    solver.set_goal(q)

    assert solver.prove_goal() is False
    feedback = solver.return_status()
    assert feedback["type"] == "COUNTERMODEL"
    assert "counterexample" in feedback


def test_inconsistent_premises():
    """Test how the solver handles contradictory premises."""
    solver = Z3Solver()
    p = z3.Bool('p')

    # Premises: p, NOT p
    solver.set_premises([p, z3.Not(p)])
    solver.set_goal(p)  # Goal doesn't matter here

    # In classical logic, from contradiction anything follows,
    # but our solver should flag the premises.
    is_valid = solver.prove_goal()
    feedback = solver.return_status()

    assert feedback["type"] == "UNSAT_PREMISES"
    assert "p0" in feedback["conflicting_premises_indices"]
    assert "p1" in feedback["conflicting_premises_indices"]


def test_first_order_logic():
    """Test simple predicate logic with quantifiers."""
    solver = Z3Solver()
    Object = z3.DeclareSort('Object')
    human = z3.Function('human', Object, z3.BoolSort())
    mortal = z3.Function('mortal', Object, z3.BoolSort())
    socrates = z3.Const('socrates', Object)
    x = z3.Const('x', Object)

    # Premises: All humans are mortal, Socrates is human
    # Goal: Socrates is mortal
    solver.set_premises([
        z3.ForAll([x], z3.Implies(human(x), mortal(x))),
        human(socrates)
    ])
    solver.set_goal(mortal(socrates))

    assert solver.prove_goal() is True


def test_missing_goal_error():
    """Ensure the solver raises an error if goal is not set."""
    solver = Z3Solver()
    with pytest.raises(ValueError, match="Goal has not been set."):
        solver.prove_goal()
