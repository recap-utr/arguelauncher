import random
import typing as t

import arguebuf as ag

from arguelauncher.config import RetrievalGraph2TextAlgorithm

random.seed(0)

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
        [
            ag.AbstractNode,
            t.Callable[[ag.AbstractNode], t.AbstractSet[ag.AbstractNode]],
        ],
        t.List[ag.AbstractNode],
    ],
) -> t.List[ag.AbstractNode]:
    start = g.major_claim or g.root_node
    assert start is not None

    incoming_nodes = list(reversed(func(start, g.incoming_nodes)))
    outgoing_nodes = func(start, g.outgoing_nodes)
    return incoming_nodes + [start] + outgoing_nodes


def _traverse_texts(
    g: ag.Graph,
    func: t.Callable[
        [
            ag.AbstractNode,
            t.Callable[[ag.AbstractNode], t.AbstractSet[ag.AbstractNode]],
        ],
        t.List[ag.AbstractNode],
    ],
) -> t.List[str]:
    nodes = _traverse_nodes(g, func)

    return [node.plain_text for node in nodes if isinstance(node, ag.AtomNode)]


def _dfs(g: ag.Graph) -> str:
    def func(start, connections):
        return ag.traverse.dfs(start, connections, include_start=False)

    texts = _traverse_texts(g, func)

    return " ".join(texts)


def _dfs_reconstruction(g: ag.Graph) -> str:
    def func(start, connections):
        return ag.traverse.dfs(start, connections, include_start=False)

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
    def func(start, connections):
        return ag.traverse.bfs(start, connections, include_start=False)

    texts = _traverse_texts(g, func)

    return " ".join(texts)


algorithm_map: dict[RetrievalGraph2TextAlgorithm, t.Callable[[ag.Graph], str]] = {
    RetrievalGraph2TextAlgorithm.NODE_ID: _node_id,
    RetrievalGraph2TextAlgorithm.ORIGINAL_RESOURCE: _original_resource,
    RetrievalGraph2TextAlgorithm.RANDOM: _random,
    RetrievalGraph2TextAlgorithm.BFS: _bfs,
    RetrievalGraph2TextAlgorithm.DFS: _dfs,
    RetrievalGraph2TextAlgorithm.DFS_RECONSTRUCTION: _dfs_reconstruction,
}


def graph2text(g: ag.Graph, algorithm: RetrievalGraph2TextAlgorithm) -> str:
    return algorithm_map[algorithm](g)
