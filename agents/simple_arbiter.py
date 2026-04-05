class SimpleArbiter:
    def __init__(self):
        self.generated_logic = None

    def set_generated_logic(self, generated_logic: dict) -> None:
        self.generated_logic = generated_logic

    def follow_critics_instructions(self, critic_input: dict, generator_agent, solver):
        if critic_input["status"] == "OK":
            return self._pass_logic_to_solver(solver)
        elif critic_input["status"] == "ERROR":
            return self._pass_feedback_to_generator(critic_input["feedback"], generator_agent)

    def _pass_logic_to_solver(self, solver):
        solver.set_premises(self.generated_logic["premises"])
        solver.set_goal(self.generated_logic["goal"])
        return solver.prove_goal()

    def _pass_feedback_to_generator(self, feedback: str, generator_agent):
        return generator_agent.create_prompt(feedback)