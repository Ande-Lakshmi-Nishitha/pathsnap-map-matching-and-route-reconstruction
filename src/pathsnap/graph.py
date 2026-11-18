import heapq
from collections import defaultdict


class RoadGraph:
    def __init__(self, trip):
        self.adj = defaultdict(list)
        for a in trip.local_arcs:
            self.adj[int(trip.arc_from[a])].append((int(trip.arc_to[a]), float(trip.arc_len[a]), int(a)))
        self.cache = {}

    def dijkstra(self, src, bound):
        hit = self.cache.get(src)
        if hit is not None and hit[0] >= bound:
            return hit[1], hit[2]
        dist, pred = {src: 0.0}, {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float("inf")):
                continue
            for v, w, arc in self.adj.get(u, ()):
                nd = d + w
                if nd <= bound and nd < dist.get(v, float("inf")):
                    dist[v], pred[v] = nd, (u, arc)
                    heapq.heappush(pq, (nd, v))
        self.cache[src] = (bound, dist, pred)
        return dist, pred

    def path_arcs(self, src, dst, bound=5000.0):
        dist, pred = self.dijkstra(src, bound)
        if dst not in dist:
            return None
        out, v = [], dst
        while v != src:
            u, arc = pred[v]
            out.append(arc)
            v = u
        return out[::-1]
