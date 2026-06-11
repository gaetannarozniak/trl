from kimina_client import KiminaClient

def _lean_reward(completions, **kwargs) -> list[float]:
    client = KiminaClient()
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        result = client.check(completion).results[0]
        rewards.append(result)
    return rewards

if __name__ == "__main__":
    reward = _lean_reward(["#check Nat"])
    print(reward)
