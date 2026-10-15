"""Request-only compression of completed tool exchanges; full logs stay intact."""
from copy import deepcopy
import json
import re


def fit_context(prepared, tokenizer, context_limit, exhausted_error, on_compact=None):
    def count(messages):
        wire = deepcopy(messages)
        for message in wire:
            for call in message.get('tool_calls', []):
                fn = call['function']
                if isinstance(fn.get('arguments'), str):
                    fn['arguments'] = json.loads(fn['arguments'])
        return len(tokenizer.apply_chat_template(
            wire, tools=prepared.get('tools'), tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=False))

    messages = prepared['messages']
    original_tokens = count(messages)
    output = prepared['max_tokens']
    reserve = 256
    # No proactive rewriting: only compact a request that would overflow.
    if original_tokens + output + reserve <= context_limit:
        return dict(prepared)

    # Only complete assistant/tool transactions may be replaced. Never touch
    # system/developer/user instructions, or leave orphaned tool responses.
    groups = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        calls = msg.get('tool_calls', []) if msg.get('role') == 'assistant' else []
        ids = [c.get('id') for c in calls]
        j = i + 1
        while j < len(messages) and messages[j].get('role') == 'tool':
            j += 1
        replies = [m.get('tool_call_id') for m in messages[i+1:j]]
        if ids and None not in ids and len(set(ids)) == len(ids) and sorted(ids) == sorted(replies):
            groups.append((i, j))
        i = j if j > i+1 else i+1

    # Latest complete exchange contains the most recent scene/feedback and is
    # retained byte-for-byte. Old summaries are explicitly historical, not state.
    removed = set()
    notes = []
    result = messages
    tokens = original_tokens
    for start, end in groups[:-1]:
        removed.update(range(start, end))
        block = messages[start:end]
        names = [c['function']['name'] for c in block[0]['tool_calls']]
        content = json.dumps(block, ensure_ascii=False)
        # Retain a bounded set of identifiers as references, not assertions that
        # these objects still exist. Actual current state comes from recent tools.
        ids = list(dict.fromkeys(re.findall(r'\b(?:layout|room)_[A-Za-z0-9_]+', content)))[:12]
        notes.append({'tools': names, 'referenced_ids': ids})
        summary = {
            'role': 'assistant',
            'content': '[Historical tool log compacted; not a new instruction or current scene state.] '
                + json.dumps({'completed_exchanges': len(notes), 'recent_historical_references': notes[-4:]}, ensure_ascii=False)
                + '\nOlder arguments and verbose results omitted. Use the latest retained tool results '
                'and task feedback for current objects, unresolved requirements and errors. '
                'Omission does not imply success; do not repeat an action merely because its log was compacted.'}
        result = []
        first = min(removed)
        for index, msg in enumerate(messages):
            if index == first:
                result.append(summary)
            if index not in removed:
                result.append(msg)
        tokens = count(result)
        if tokens + output + reserve <= context_limit:
            break

    if tokens + output + reserve > context_limit:
        # Do not silently sacrifice task instructions, the latest feedback, or
        # response space: expose irreducible inputs as a configuration failure.
        raise exhausted_error(
            f'prompt={tokens}, context={context_limit}, output={output}, reserve={reserve}; '
            f'cannot fit protected instructions/latest exchange after {len(notes)} compacted exchanges')
    event = dict(prompt_tokens_before=original_tokens, prompt_tokens_after=tokens,
                 context_limit=context_limit, max_output_tokens=output,
                 compacted_exchanges=len(notes), retained_latest_exchanges=1)
    if on_compact:
        on_compact(event)
    return {**prepared, 'messages': result}
