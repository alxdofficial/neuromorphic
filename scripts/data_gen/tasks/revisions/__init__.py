"""Fact revision / correction task — pure negation+overwrite test.

A scenario is the "state of a project" with several attributes, each
revised 1-3 times across passages. The model must keep the latest value
of each attribute. Different from preferences in that:
- Multiple attributes are tracked simultaneously
- Revisions are explicit corrections ("Actually X, not Y")
- The chain can be longer (>2 revisions)

Question types:
- current_value: "what's the latest X?"
- how_many_revisions: "how many times was X updated?"
"""
