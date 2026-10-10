"""A small "envelope" that describes ONE action the agent is about to do or just did.

What this file is for
---------------------
`Agent.register_action_callback` lets outside code watch every single action the
agent performs (click, type, select...). Each time, the callback receives one
`ActionEvent`: everything we know about that action in one place.

Why it exists
-------------
The agent talks about elements by temporary numbers like `[12]`. Those numbers
mean nothing after the page changes. The only moment we can ask "which real
element is number 12?" is just BEFORE the action runs. So the agent calls the
callback twice for every action, with the same event object:

    phase == "before"  ->  the element is still live on the page; look at it now
    phase == "after"   ->  the result is filled in (worked or failed)

The callback is optional. If nobody sets one, nothing here is used and the agent
behaves exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

if TYPE_CHECKING:
	from browser_use.dom.views import EnhancedDOMTreeNode


@dataclass
class ActionEvent:
	"""Everything we know about one action, before and after it runs."""

	phase: Literal['before', 'after']  # which of the two calls this is

	step: int  # the agent's step number (the LLM is asked once per step)
	position: int  # place of this action inside the step (0 = first)
	actions_in_step: int  # how many actions the LLM chose in this step

	action_name: str  # e.g. "click", "input", "navigate", "done"
	params: dict[str, Any]  # e.g. {"index": 12, "text": "xyz"}

	# The real page element this action points at, if it points at one (it is
	# looked up from the element number). None for actions like "navigate".
	# Only trustworthy in the "before" phase, while the page has not changed.
	element: EnhancedDOMTreeNode | None = None

	# What the LLM wrote at the start of this step.
	next_goal: str | None = None  # what it is trying to do now
	evaluation: str | None = None  # its verdict on the PREVIOUS step

	url_before: str | None = None  # page address when the action started

	# Filled in for the "after" phase only:
	url_after: str | None = None
	success: bool | None = None  # True = no error, False = the action reported an error
	error: str | None = None  # the error text when success is False


# The shape of the callback: takes one ActionEvent. It may be a normal function
# or an `async def`; the agent handles both.
ActionCallback = Callable[[ActionEvent], 'Awaitable[None] | None']
