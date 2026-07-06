"""Theory of Mind task — 1st-order false belief.

Each scenario is a small "room scene":
- Characters present in a room
- One object in a container
- A character leaves
- Another character moves the object
- The original character returns

State the model must track:
- True location of each object
- Each character's *belief* about each object's location
  (= the last event involving that object that they witnessed)

Question types:
- where_belief: "Where does X think the Y is?"  (tests false belief)
- where_actually: "Where is the Y actually?"     (tests true location)
- has_seen: "Did X see the Y being moved?"        (tests witness tracking)
"""
