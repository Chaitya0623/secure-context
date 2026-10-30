import json
import pytest
import requests

from secure_context.llm import GroqLLM, get_backend
from secure_context.detect import GroqPIIDetector


class Response:
    def __init__(self, status=200, content='OK', finish='stop'):
        self.status_code = status
        self.ok = status < 400
        self.headers = {'retry-after': '0'}
        self.content = content
        self.finish = finish

    def json(self):
        return {'choices': [{'message': {'content': self.content}, 'finish_reason': self.finish}]}


def test_retry_cache_and_model_isolation(monkeypatch, tmp_path):
    monkeypatch.setenv('GROQ_API_KEY', 'test-key')
    responses = iter([Response(429), Response(), Response(content='different')])
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs)
        return next(responses)
    monkeypatch.setattr(requests, 'post', post)
    assert GroqLLM('a', tmp_path).complete('system', 'prompt') == 'OK'
    assert GroqLLM('a', tmp_path).complete('system', 'prompt') == 'OK'
    assert GroqLLM('b', tmp_path).complete('system', 'prompt') == 'different'
    assert len(calls) == 3
    assert all('test-key' not in p.read_text() for p in tmp_path.iterdir())


def test_incomplete_answer_not_cached(monkeypatch, tmp_path):
    monkeypatch.setenv('GROQ_API_KEY', 'test-key')
    monkeypatch.setattr(requests, 'post', lambda *a, **k: Response(finish='length'))
    with pytest.raises(RuntimeError, match='incomplete'):
        GroqLLM('a', tmp_path).complete('s', 'p')
    assert not list(tmp_path.iterdir())


def test_unknown_backend():
    with pytest.raises(ValueError, match='Unknown'):
        get_backend('typo')


def test_detector_matches_source_only():
    class Fake:
        def complete(self, *args, **kwargs):
            return json.dumps({'items': [{'text': 'Alice', 'type': 'PERSON'},
                                         {'text': 'Invented', 'type': 'PERSON'}]})
    text = 'Alice called Alice.'
    spans = GroqPIIDetector(Fake()).detect(text)
    assert [(s.start, s.end) for s in spans] == [(0, 5), (13, 18)]
    assert all(text[s.start:s.end] == s.text for s in spans)


def test_detector_rejects_bad_schema():
    class Fake:
        def complete(self, *args, **kwargs):
            return '{"wrong": []}'
    with pytest.raises(ValueError, match='schema'):
        GroqPIIDetector(Fake()).detect('Alice')


def test_evaluation_sessions_resume(monkeypatch, tmp_path):
    from eval import llm_eval
    monkeypatch.setattr(llm_eval, 'ROOT', tmp_path)
    first = llm_eval.evaluation_session('client:task')
    resumed = llm_eval.evaluation_session('client:task')
    other = llm_eval.evaluation_session('other:task')
    assert first.token('PERSON', 'Alice') == resumed.token('PERSON', 'Alice')
    assert first.key != other.key
