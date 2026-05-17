"""Calendar/scheduling task: temporal + conflict reasoning.

Each scenario is a "week" with N events scheduled at specific day-of-week
and time slots. Each event gets one passage. Then a question asks about
the calendar:
- free_at: "am I free at <day> <time>?" (yes/no)
- conflict_with: "if I want to schedule X at <day> <time>, what conflicts?"
- next_event_on: "what's my next event on <day>?"
- busy_count_on: "how many events do I have on <day>?"

These exercise temporal reasoning (interval overlap, ordering) in a way
single-fact retrieval cannot. An AI agent that handles calendars needs
exactly this.
"""
