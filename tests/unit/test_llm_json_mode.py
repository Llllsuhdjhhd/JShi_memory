from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from rems.llm.provider import LLMProvider
from rems.eval.client import EvaluationLLM


def provider(cls=LLMProvider):
    model = cls.__new__(cls)
    model.config = SimpleNamespace(llm=SimpleNamespace(temperature=0.7, extra_body={}))
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2, model_dump=lambda: {})
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'), finish_reason='stop')],
                               usage=usage, model='fixture', id='fixture')
    create = Mock(return_value=response)
    model._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    model._invocations = []
    return model, create


@pytest.mark.parametrize('continuation', [False, True])
def test_complete_json_enables_deepseek_mode_only(continuation):
    model, create = provider()
    messages = [{'role': 'user', 'content': 'Summarize'}]
    if continuation:
        result, history = model.complete_json_continue('summary', messages, 'Continue', model='deepseek-v4-flash', temperature=0)
        assert history[-1]['role'] == 'assistant'
    else:
        result = model.complete_json('summary', messages, model='deepseek-v4-flash', temperature=0)
        assert len(messages) == 1
    assert result == {'ok': True}
    body = create.call_args.kwargs
    assert body['response_format'] == {'type': 'json_object'}
    assert body['temperature'] == 0
    assert 'JSON' in body['messages'][-1]['content']


def test_plain_text_and_other_providers_keep_their_contract():
    model, create = provider()
    model.complete('summary', [], model='deepseek-v4-flash')
    assert 'response_format' not in create.call_args.kwargs
    model.complete_json('summary', [], model='other-provider')
    assert 'response_format' not in create.call_args.kwargs


def test_eval_subclass_remains_compatible(tmp_path):
    model, create = provider(EvaluationLLM)
    model.max_output, model.ledger, model.audit_path = 4096, Mock(), tmp_path / 'audit.jsonl'
    assert model.complete_json('core', [], model='deepseek-flash') == {'ok': True}
    assert create.call_args.kwargs['response_format'] == {'type': 'json_object'}
