"""Knights & Knaves puzzle task.

N characters live on the island. Each is either a KNIGHT (always tells
the truth) or a KNAVE (always lies). Each character makes a statement
about themselves or another character. The model must figure out
who is which.

Generator constraints:
- Every emitted scenario must have exactly ONE truth-value assignment
  consistent with all statements (puzzle uniqueness).
- We enumerate all 2^N assignments and accept only scenarios where one
  is consistent.

Question type: identity_of — "Is character X a knight or a knave?"
"""
