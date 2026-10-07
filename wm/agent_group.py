class AgentGroup:
    """Lifecycle operations shared by containers of local agents."""

    def process_dones(self, dones):
        for agent in self.agents:
            agent.process_dones(dones)

    def clear_completed(self):
        for agent in self.agents:
            agent.clear_completed()

    def start_agents(self):
        for agent in self.agents:
            agent.episode_start()
