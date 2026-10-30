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
            assert kwargs['json_schema'] == GroqPIIDetector.SCHEMA
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


def test_api_error_details_redact_key_and_omit_generation(monkeypatch, tmp_path):
    monkeypatch.setenv('GROQ_API_KEY', 'test-secret')
    monkeypatch.setattr('time.sleep', lambda _: None)
    class ErrorResponse(Response):
        def json(self):
            return {'error': {'code': 'json_validate_failed',
                              'message': 'Invalid JSON test-secret',
                              'failed_generation': 'sensitive generated content'}}
    monkeypatch.setattr(requests, 'post', lambda *a, **k: ErrorResponse(400))
    with pytest.raises(RuntimeError, match='json_validate_failed') as error:
        GroqLLM('a', tmp_path).complete('s', 'p', json_mode=True)
    assert 'test-secret' not in str(error.value)
    assert 'sensitive generated content' not in str(error.value)
    assert not list(tmp_path.iterdir())


def test_strict_schema_request_and_cache_isolation(monkeypatch, tmp_path):
    monkeypatch.setenv('GROQ_API_KEY', 'test-key')
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs['json'])
        return Response(content='{"items": []}')
    monkeypatch.setattr(requests, 'post', post)
    llm = GroqLLM('openai/gpt-oss-20b', tmp_path)
    llm.complete('s', 'p', json_mode=True)
    llm.complete('s', 'p', json_schema=GroqPIIDetector.SCHEMA)
    llm.complete('s', 'p', json_schema=GroqPIIDetector.SCHEMA)
    assert len(calls) == 2
    response_format = calls[1]['response_format']
    assert response_format['type'] == 'json_schema'
    assert response_format['json_schema']['strict'] is True
    assert response_format['json_schema']['schema'] == GroqPIIDetector.SCHEMA


def test_transient_json_failure_retries_without_caching_error(monkeypatch, tmp_path):
    monkeypatch.setenv('GROQ_API_KEY', 'test-key')
    monkeypatch.setattr('time.sleep', lambda _: None)
    class InvalidJSON(Response):
        def json(self):
            return {'error': {'code': 'json_validate_failed', 'message': 'Invalid JSON'}}
    responses = iter([InvalidJSON(400), Response(content='{"items": []}')])
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        return next(responses)
    monkeypatch.setattr(requests, 'post', post)
    llm = GroqLLM('openai/gpt-oss-20b', tmp_path)
    assert llm.complete('s', 'p', json_schema=GroqPIIDetector.SCHEMA) == '{"items": []}'
    assert llm.complete('s', 'p', json_schema=GroqPIIDetector.SCHEMA) == '{"items": []}'
    assert len(calls) == 2
    assert len(list(tmp_path.iterdir())) == 1


def test_other_400_errors_are_not_retried(monkeypatch, tmp_path):
    monkeypatch.setenv('GROQ_API_KEY', 'test-key')
    calls = []
    class BadRequest(Response):
        def json(self):
            return {'error': {'code': 'invalid_parameter', 'message': 'Unsupported format'}}
    def post(*a, **k):
        calls.append(k)
        return BadRequest(400)
    monkeypatch.setattr(requests, 'post', post)
    with pytest.raises(RuntimeError, match='invalid_parameter'):
        GroqLLM('a', tmp_path).complete('s', 'p', json_schema=GroqPIIDetector.SCHEMA)
    assert len(calls) == 1


def test_format_fallback_validates_before_caching(monkeypatch, tmp_path):
    monkeypatch.setenv('GROQ_API_KEY', 'test-key')
    monkeypatch.setattr('time.sleep', lambda _: None)
    class InvalidJSON(Response):
        def json(self):
            return {'error': {'code': 'json_validate_failed', 'message': 'Invalid JSON'}}
    responses = iter([InvalidJSON(400)] * 3 + [
        Response(content='{"items": [{"text": "Alice", "type": "INVALID"}]}'),
        Response(content='{"items": []}')])
    formats = []
    def post(*args, **kwargs):
        formats.append(kwargs['json'].get('response_format'))
        return next(responses)
    monkeypatch.setattr(requests, 'post', post)
    llm = GroqLLM('a', tmp_path)
    assert llm.complete('s', 'p', json_schema=GroqPIIDetector.SCHEMA) == '{"items": []}'
    assert formats[:3] == [formats[0]] * 3
    assert formats[3:] == [None, None]
    assert llm.complete('s', 'p', json_schema=GroqPIIDetector.SCHEMA) == '{"items": []}'
    assert len(formats) == 5
    assert len(list(tmp_path.iterdir())) == 1
