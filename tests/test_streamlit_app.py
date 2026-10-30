from pathlib import Path

from streamlit.testing.v1 import AppTest

from secure_context import real_data


def test_app_pipeline_and_source_switch(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(real_data, 'DATA', root / 'data/sample')
    app = AppTest.from_file(str(root / 'app/streamlit_app.py'), default_timeout=90).run()
    app.toggle[0].set_value(False).run()

    def healthy():
        assert not app.exception, [e.message for e in app.exception]

    def radio(label):
        return next(r for r in app.radio if r.label == label)

    healthy()

    for step in radio('Pipeline step').options:
        radio('Pipeline step').set_value(step).run()
        healthy()
    for mode in ('default', 'masked', 'custom', 'adaptive'):
        radio('Context sharing').set_value(mode).run()
        healthy()
    for results in ('Synthetic', 'Public datasets'):
        radio('Results').set_value(results).run()
        healthy()
    send = next(b for b in app.button if b.label.startswith('Send:'))
    send.click().run()
    healthy()
    assert app.session_state['log']
    old_engine = app.session_state['ledgers']['default'].engine
    radio('Data').set_value('Fully synthetic').run()
    healthy()
    assert not app.session_state['log']
    assert app.session_state['ledgers']['default'].engine is not old_engine
    assert app.session_state['ledger_source'] == 'Fully synthetic'
    source = next(option for option in radio('Data').options if option.startswith('Public'))
    radio('Data').set_value(source).run()
    healthy()


def test_guided_demo_needs_no_population_or_model(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Guided demo must not load data or call a model")
    from secure_context import llm
    monkeypatch.setattr(real_data, 'load_real_environment', forbidden)
    monkeypatch.setattr(llm, 'get_backend', forbidden)
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / 'app/streamlit_app.py')).run()
    for step in app.radio[0].options:
        app.radio[0].set_value(step).run()
        assert not app.exception
    assert '2 tokens restored' in app.success[0].value
