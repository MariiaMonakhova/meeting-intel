"""Token counting for chunk-size budgeting.

This is a local estimate used only to decide how many utterances fit into a
chunk before an API call is made. It is never used for cost accounting -
real token counts for billing come from the API response's usage field.
"""

WORDS_PER_TOKEN = 0.75


def count_tokens(text: str) -> int:
    words = text.split()
    if not words:
        return 0
    return max(1, round(len(words) / WORDS_PER_TOKEN))
