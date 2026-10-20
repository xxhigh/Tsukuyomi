"""Unreal Engine RAG 에이전트."""


def build_graph(*args, **kwargs):
    from .graph import build_graph as factory
    return factory(*args, **kwargs)


__all__ = ["build_graph"]
