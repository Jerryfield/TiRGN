"""RCEV-NoLLM evidence package.

Additive experiment infrastructure for the TiRGN baseline. Nothing in this
package modifies the original model logic in model.py / rrgcn.py / decoder.py;
the only hook into the original code is a candidate-dump call in main.py right
after model.predict().
"""
