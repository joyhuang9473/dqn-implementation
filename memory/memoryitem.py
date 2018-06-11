class MemoryItem:
    def __init__(self, observation, action, reward, done, info):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.done = done
        self.info = info
