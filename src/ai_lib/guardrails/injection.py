import re

INJECTION_PATTERNS = [
    r"ignore(?:\s+todas)?\s+as\s+instru[cç][oõ]es",
    r"esque[cç]a\s+(?:as\s+)?instru[cç][oõ]es",
    r"you\s+are\s+now",
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
    r"\[inst\]",
    r"modo\s+(?:irrestrito|desenvolvedor|admin)",
    r"revele\s+(?:seu|o)\s+prompt",
    r"aja\s+como\s+se\s+voc[eê]\s+n[aã]o\s+tivesse\s+regras",
]

INTERNAL_DATA_KEYWORDS = [
    "system_core",
    "prompt do sistema",
    "system prompt",
    "arquitetura interna",
    "credenciais",
    "chave de api",
    "api key",
    "código-fonte",
    "codigo fonte",
]


def matches_injection_pattern(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)


def matches_internal_data_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in INTERNAL_DATA_KEYWORDS)
