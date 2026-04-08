# ------------------------------
# agent.py
# ------------------------------

class QAgent:
    def __init__(self, q_table):
        """
        Q-table agent that chooses actions based on the trained Q-table.
        q_table: list of lists, q_table[state][action] = value
        """
        self.q_table = q_table

    def choose_action(self, state):
        """
        Chooses the best action for the current state.
        state: dict with 'ambulance_lane' key
        """
        state_lane = state["ambulance_lane"]
        # Pick the action with the highest Q-value for this state
        action = self.q_table[state_lane].index(max(self.q_table[state_lane]))
# ------------------------------
# agent.py
# ------------------------------

class QAgent:
    def __init__(self, q_table):
        """
        Q-table agent that chooses actions based on the trained Q-table.
        q_table: list of lists, q_table[state][action] = value
        """
        self.q_table = q_table

    def choose_action(self, state):
        """
        Chooses the best action for the current state.
        state: dict with 'ambulance_lane' key
        """
        state_lane = state["ambulance_lane"]
        # Pick the action with the highest Q-value for this state
        action = self.q_table[state_lane].index(max(self.q_table[state_lane]))
        return action