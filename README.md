# ai-lib

Blocos compartilhados dos serviços de IA da Solaria. Publicado no PyPI como `solaria-lib`; o import é `ai_lib`. Requer Python 3.14 ou superior.

```bash
pip install solaria-lib
```

## O que tem

| Módulo | Conteúdo |
|---|---|
| `ai_lib.registry` | Cliente do corretor de chaves do `google-registry` (`GET /v1/llm/keys`, `POST /v1/llm/keys/{id}/report`) |
| `ai_lib.llm` | Chat model LangChain que pede uma chave ao corretor a cada chamada, `LeasedEmbeddings`, tabela de preços |
| `ai_lib.guardrails` | Anonimização de PII, detecção de injection, parsers de veredito e nós LangGraph de entrada, saída e juiz |
| `ai_lib.security` | `decode_user_id` (JWT RS256 via JWKS) |
| `ai_lib.observability` | `StepTracker`, callback LangChain que mede custo, tokens e latência |

## Configuração

Só o `google-registry` tem as chaves de LLM. Os serviços recebem duas variáveis:

| Variável | Uso |
|---|---|
| `REGISTRY_URL` | URL base do `google-registry` |
| `REGISTRY_CONSUMER_TOKEN` | Token de consumidor (`Authorization: Bearer`) |

Não existe fallback para chave local: se o registry não responde, a chamada falha.

## Uso

```python
from ai_lib.llm import get_chat_model, get_chat_model_with_fallback

llm = get_chat_model_with_fallback()
llm.invoke("olá")

groq = get_chat_model("groq", model="openai/gpt-oss-120b")
structured = get_chat_model("gemini", purpose="vision").with_structured_output(MySchema)
```

Cada chamada pede uma chave, reporta `ok`, `rate_limited` ou `invalid` ao registry e, em 429, repete com outra chave (`exclude`). `bind_tools` e `with_structured_output` funcionam como no LangChain.

### Política de falha do registry

Timeout de 10 s por chamada. Em erro de conexão ou 503 sem `Retry-After`: até 3 tentativas com espera crescente (teto de 60 s). Em 503 com `Retry-After`: espera (máx. 30 s) e tenta mais uma vez. Em 401 e 404: falha na hora. Todos os valores vêm de `ai_lib.registry.RetryPolicy`.

### Guardrail

```python
from ai_lib.guardrails import (
    build_input_prompt,
    build_judge_prompt,
    build_output_prompt,
    make_input_guardrail_node,
    make_judge_node,
    make_output_guardrail_node,
)

input_node = make_input_guardrail_node(prompt=build_input_prompt("Solaria", "energia solar"))
output_node = make_output_guardrail_node(prompt=build_output_prompt("Solaria", "energia solar"))
judge_node = make_judge_node(prompt=build_judge_prompt("Solaria", "energia solar"))
```

O estado do grafo estende `ai_lib.guardrails.GuardrailState`. Serviço com necessidade específica passa o próprio prompt (string com `{mensagem}` ou `{resposta}`) ou mantém o nó próprio.

## Desenvolvimento

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

Commits seguem `tipo: mensagem` (inglês, minúsculo). O versionamento é feito pelo `release-please`; o comando `/release` na PR de release publica no GitHub e o `publish.yml` envia ao PyPI por Trusted Publishing (ambiente `pypi`).
