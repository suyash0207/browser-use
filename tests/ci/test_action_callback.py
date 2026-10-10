"""Tests for `Agent.register_action_callback` (the "before / after every action" hook).

These tests do not open a browser or call an LLM. They build a bare Agent object
and give it tiny fake pieces (a fake browser session, a fake model output), then
check that the callback is called at the right moments.
"""

from types import SimpleNamespace

from browser_use.agent.action_event import ActionEvent
from browser_use.agent.service import Agent


class _FakeBrowserSession:
	"""Pretends to be a browser session: knows one element and one url."""

	def __init__(self):
		self.url = 'https://example.com/form'
		self.element = SimpleNamespace(tag_name='input')
		self.id = 'session-0000'  # read by the agent's `logger` property
		self.agent_focus = None

	async def get_dom_element_by_index(self, index: int):
		return self.element if index == 12 else None

	async def get_current_page_url(self) -> str:
		return self.url


def _make_agent(callback):
	"""An Agent with only the attributes the hook helpers touch."""
	agent = Agent.__new__(Agent)
	agent.register_action_callback = callback
	agent.browser_session = _FakeBrowserSession()
	agent.task_id = 'task-0000'  # the `logger` property builds its name from this
	agent.state = SimpleNamespace(
		n_steps=4,
		last_model_output=SimpleNamespace(next_goal='Fill address line 1', evaluation_previous_goal='Verdict: Success'),
	)
	return agent


async def test_no_callback_means_nothing_happens():
	agent = _make_agent(None)
	event = await agent._notify_action_before('input', {'input': {'index': 12, 'text': 'xyz'}}, 0, 1)
	assert event is None
	await agent._notify_action_after(event, None)  # must not raise


async def test_before_event_carries_the_live_element_and_the_llm_context():
	seen: list[ActionEvent] = []
	agent = _make_agent(lambda e: seen.append((e.phase, e.element, e.params, e.step, e.next_goal, e.evaluation)))

	await agent._notify_action_before('input', {'input': {'index': 12, 'text': 'xyz'}}, 0, 1)

	phase, element, params, step, next_goal, evaluation = seen[0]
	assert phase == 'before'
	assert element is agent.browser_session.element
	assert params == {'index': 12, 'text': 'xyz'}
	assert step == 4
	assert next_goal == 'Fill address line 1'
	assert evaluation == 'Verdict: Success'


async def test_after_call_fills_in_success_and_error_on_the_same_event():
	events: list[ActionEvent] = []

	async def callback(event: ActionEvent):  # an async callback must work too
		events.append(event)

	agent = _make_agent(callback)
	event = await agent._notify_action_before('click', {'click': {'index': 12}}, 0, 1)
	await agent._notify_action_after(event, 'Element not clickable')

	assert events[0] is events[1]  # same object, updated in place
	assert events[1].phase == 'after'
	assert events[1].success is False
	assert events[1].error == 'Element not clickable'
	assert events[1].url_after == 'https://example.com/form'


async def test_actions_without_an_element_index_have_no_element():
	seen: list[ActionEvent] = []
	agent = _make_agent(seen.append)
	await agent._notify_action_before('navigate', {'navigate': {'url': 'https://x.test'}}, 0, 1)
	assert seen[0].element is None
	assert seen[0].params == {'url': 'https://x.test'}


async def test_a_crashing_callback_never_breaks_the_agent():
	def broken(event: ActionEvent):
		raise RuntimeError('boom')

	agent = _make_agent(broken)
	event = await agent._notify_action_before('input', {'input': {'index': 12, 'text': 'x'}}, 0, 1)
	await agent._notify_action_after(event, None)  # neither call may raise
