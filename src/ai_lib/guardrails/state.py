from langgraph.graph import MessagesState


class GuardrailState(MessagesState):
    route: str
    turn_agents: list[str]
    pii_map: dict
    judge_retries: int
    judge_status: str
