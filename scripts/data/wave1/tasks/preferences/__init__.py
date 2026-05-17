"""Preferences task: persona/preference tracking with optional cancellation.

Each scenario has a user stating one or more preferences. Each preference
gets its own passage. A question then asks the user's preference about
some topic — including cases where the preference was later RETRACTED
(testing the model's ability to handle negation/cancellation, which is
crucial for real AI agents).

Question types:
- preference_recall: "what does X prefer for Y?"
- preference_cancelled: "what did X *actually* end up preferring for Y?"
  (where an earlier preference was retracted)
"""
