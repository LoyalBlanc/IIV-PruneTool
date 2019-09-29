"""
    if A+B=0: del A,B           Error: Feature changes after ReLU
    if A-B=0: double A del B
"""
from Basic import Basic


class BasicSimilarity(Basic):
    pass


if __name__ == "__main__":
    from Basic.Abstract import prune_test

    basic_similarity = BasicSimilarity(4, 10, 2)
    prune_test(basic_similarity)
