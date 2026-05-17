"""Passphrase task: verbatim and descriptor recall.

A simple task family that stresses verbatim memorization. Each passage
introduces a "passphrase" via a natural sentence ("My secret phrase is
..."). The question asks the model to recall it.

Two question types:
- verbatim_recall: exact-string answer
- descriptor_recall: "what's my X" where X is a slot attached to the phrase

Used as the simplest test of the composite framework. If the architecture
can't memorize a phrase across a few intervening windows, something is
fundamentally broken.
"""
