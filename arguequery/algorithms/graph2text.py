import random
import typing as t

random.seed(0)

import arguebuf as ag

from arguequery.config import config
from arguequery.types import Graph2TextAlgorithm

SCHEME_RECONSTRUCTION: dict[t.Type[ag.Scheme], str] = {
    ag.Support: "This is true because",
    ag.Attack: "On the contrary,",
}


def _node_id(g: ag.Graph) -> str:
    return " ".join(
        node.plain_text for node in sorted(g.atom_nodes.values(), key=lambda x: x.id)
    )


def _original_resource(g: ag.Graph) -> str:
    return " ".join(resource.plain_text for resource in g.resources.values())


def _random(g: ag.Graph) -> str:
    nodes = list(g.atom_nodes.values())
    random.shuffle(nodes)

    return " ".join(node.plain_text for node in nodes)


def _traverse_nodes(
    g: ag.Graph,
    func: t.Callable[
        [ag.Node, t.Callable[[ag.Node], t.AbstractSet[ag.Node]]], t.List[ag.Node]
    ],
) -> t.List[ag.Node]:
    start = g.major_claim or g.root_node
    assert start is not None

    incoming_nodes = list(reversed(func(start, g.incoming_nodes)))
    outgoing_nodes = func(start, g.outgoing_nodes)
    return incoming_nodes + [start] + outgoing_nodes


def _traverse_texts(
    g: ag.Graph,
    func: t.Callable[
        [ag.Node, t.Callable[[ag.Node], t.AbstractSet[ag.Node]]], t.List[ag.Node]
    ],
) -> t.List[str]:
    nodes = _traverse_nodes(g, func)

    return [node.plain_text for node in nodes if isinstance(node, ag.AtomNode)]


def _dfs(g: ag.Graph) -> str:
    func = lambda start, connections: ag.traversal.dfs(
        start, connections, include_start=False
    )
    texts = _traverse_texts(g, func)

    return " ".join(texts)


def _dfs_reconstruction(g: ag.Graph) -> str:
    func = lambda start, connections: ag.traversal.dfs(
        start, connections, include_start=False
    )
    nodes = _traverse_nodes(g, func)
    texts = []

    for node in nodes:
        if isinstance(node, ag.AtomNode):
            texts.append(node.plain_text)
        elif (
            isinstance(node, ag.SchemeNode)
            and node.scheme is not None
            and type(node.scheme) in SCHEME_RECONSTRUCTION
        ):
            texts.append(SCHEME_RECONSTRUCTION[type(node.scheme)])

    return " ".join(texts)


def _bfs(g: ag.Graph) -> str:
    func = lambda start, connections: ag.traversal.bfs(
        start, connections, include_start=False
    )
    texts = _traverse_texts(g, func)

    return " ".join(texts)


algorithm_map: dict[Graph2TextAlgorithm, t.Callable[[ag.Graph], str]] = {
    "node_id": _node_id,
    "original_resource": _original_resource,
    "random": _random,
    "bfs": _bfs,
    "dfs": _dfs,
    "dfs_reconstruction": _dfs_reconstruction,
}


def graph2text(g: ag.Graph, algorithm: Graph2TextAlgorithm) -> str:
    return algorithm_map[algorithm](g)
