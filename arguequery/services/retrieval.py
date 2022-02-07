from __future__ import absolute_import, annotations

import bisect
import logging
import multiprocessing
import random
import typing as t
from typing import Dict, List, Optional, Tuple, Union

import arguebuf as ag
from arguequery.models.mapping import Mapping, SearchNode
from arguequery.models.result import Result
from arguequery.services import nlp

logger = logging.getLogger("recap")
from arguequery.config import config


def fac(mac_results: List[Result], query_graph: ag.Graph) -> List[Result]:
    """Perform an in-depth analysis of the prefilter results"""

    results: List[Result] = []
    params = [
        (mac_result.graph, query_graph, i, len(mac_results))
        for i, mac_result in enumerate(mac_results)
    ]

    logger.info(f"A* Search for query '{query_graph.name}'.")

    if config["debug"]:
        results = [a_star_search(*param) for param in params]
    else:
        with multiprocessing.Pool() as pool:
            results = pool.starmap(a_star_search, params)

    results.sort(key=lambda result: result.similarity, reverse=True)

    return results


# According to Bergmann and Gil, 2014
def a_star_search(
    case_graph: ag.Graph,
    query_graph: ag.Graph,
    current_iteration: int,
    total_iterations: int,
) -> Result:
    """Perform an A* analysis of the case base and the query"""

    q: List[SearchNode] = []
    s0 = SearchNode(
        len(query_graph.nodes),
        len(query_graph.edges),
        query_graph.nodes.values(),
        query_graph.edges.values(),
    )

    bisect.insort(q, s0)

    while q[-1].nodes or q[-1].edges:
        q = _expand(q, case_graph, query_graph)

    candidate = q[-1]

    logger.debug(
        f"A* search for {case_graph.name} finished. ({current_iteration}/{total_iterations})"
    )

    return Result(case_graph, candidate.mapping.similarity)


def _expand(
    q: List[SearchNode], case_graph: ag.Graph, query_graph: ag.Graph
) -> List[SearchNode]:
    """Expand a given node and its queue"""

    s = q[-1]
    mapped = False
    query_obj, iterator = select1(s, query_graph, case_graph)

    if query_obj and iterator:
        for case_obj in iterator:
            if s.mapping.is_legal_mapping(query_obj, case_obj):
                s_new = SearchNode(
                    len(query_graph.nodes),
                    len(query_graph.edges),
                    s.nodes,
                    s.edges,
                    s.mapping,
                )
                s_new.mapping.map(query_obj, case_obj)
                s_new.remove(query_obj)
                s_new.f = g(s_new, query_graph) + h2(s_new, query_graph, case_graph)
                bisect.insort(q, s_new)
                mapped = True

        if mapped:
            q.remove(s)
        else:
            s.remove(query_obj)

    return (
        q[len(q) - config["a_star_queue_limit"] :]
        if config["a_star_queue_limit"] > 0
        else q
    )


def select1(
    s: SearchNode, query_graph: ag.Graph, case_graph: ag.Graph
) -> t.Tuple[
    t.Optional[t.Union[ag.Node, ag.Edge, None]],
    t.Optional[t.Iterable[t.Union[ag.Node, ag.Edge]]],
]:
    query_obj = None
    candidates = None

    if s.nodes:
        query_obj = random.choice(tuple(s.nodes))
        candidates = (
            case_graph.atom_nodes.values()
            if isinstance(query_obj, ag.AtomNode)
            else case_graph.scheme_nodes.values()
        )
    elif s.edges:
        query_obj = random.choice(tuple(s.edges))
        candidates = case_graph.edges.values()

    return query_obj, candidates


# def select2(s: SearchNode, query_graph: ag.Graph, case_graph: ag.Graph) -> Tuple:
#     pass


def h1(s: SearchNode, query_graph: ag.Graph, case_graph: ag.Graph) -> float:
    """Heuristic to compute future costs"""

    return (len(s.nodes) + len(s.edges)) / (
        len(query_graph.nodes) + len(query_graph.edges)
    )


def h2(s: SearchNode, query_graph: ag.Graph, case_graph: ag.Graph) -> float:
    h_val = 0

    for x in s.nodes:
        max_sim = max(
            nlp.similarity(x, y)
            for y in (
                case_graph.atom_nodes.values()
                if isinstance(x, ag.AtomNode)
                else case_graph.scheme_nodes.values()
            )
        )

        h_val += max_sim

    for x in s.edges:
        max_sim = max(nlp.similarity(x, y) for y in case_graph.edges.values())

        h_val += max_sim

    return h_val / (len(query_graph.nodes) + len(query_graph.edges))


def g(s: SearchNode, query_graph: ag.Graph) -> float:
    """Function to compute the costs of all previous steps"""

    return s.mapping.similarity
