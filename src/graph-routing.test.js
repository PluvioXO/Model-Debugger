import test from "node:test";
import assert from "node:assert/strict";

import {
  roundedOrthogonalPath,
  routeGraphEdge,
  routeGraphEdges,
  routeIntersectsNodes,
  routesShareParallelSegment,
  tensorStackDepth
} from "./graph-routing.js";

test("tensor rank controls the number of visible rear cards", () => {
  assert.equal(tensorStackDepth(1), 0);
  assert.equal(tensorStackDepth(2), 1);
  assert.equal(tensorStackDepth(3), 2);
  assert.equal(tensorStackDepth(8), 3);
});

test("adjacent vertical edges start outside the complete tensor stack", () => {
  const route = routeGraphEdge(
    { feedback: false, kind: "activation" },
    { x: 360, y: 414, tensorRank: 3 },
    { x: 360, y: 552, tensorRank: 0 }
  );
  assert.deepEqual(route.points, [
    { x: 482, y: 510 },
    { x: 482, y: 552 }
  ]);
});

test("long same-column edges use a clear side lane", () => {
  const route = routeGraphEdge(
    { feedback: false, kind: "activation" },
    { x: 360, y: 414, tensorRank: 3 },
    { x: 360, y: 1104, tensorRank: 0 }
  );
  assert.deepEqual(route.points, [
    { x: 616, y: 456 },
    { x: 640, y: 456 },
    { x: 640, y: 1146 },
    { x: 604, y: 1146 }
  ]);
});

test("unused layout rows do not turn a clear same-column dependency into a side-lane edge", () => {
  const edge = { from: "qkv", to: "attention", feedback: false, kind: "activation" };
  const nodes = new Map([
    ["qkv", { x: 360, y: 414, tensorRank: 3 }],
    ["attention", { x: 360, y: 690, tensorRank: 3 }],
  ]);

  const route = routeGraphEdges([edge], nodes).get(edge);

  assert.equal(route.sourceSide, "bottom");
  assert.equal(route.targetSide, "top");
  assert.deepEqual(route.points, [
    { x: 482, y: 510 },
    { x: 482, y: 690 },
  ]);
});

test("same-column dependencies use a side lane only when a real node blocks the corridor", () => {
  const edge = { from: "source", to: "target", feedback: false, kind: "activation" };
  const nodes = new Map([
    ["source", { x: 360, y: 414, tensorRank: 0 }],
    ["blocker", { x: 360, y: 552, tensorRank: 2 }],
    ["target", { x: 360, y: 828, tensorRank: 0 }],
  ]);

  const route = routeGraphEdges([edge], nodes).get(edge);

  assert.equal(route.sourceSide, "right");
  assert.equal(route.targetSide, "right");
  assert.ok(route.points.some((point) => point.x > 604));
});

test("nodes in neighbouring columns do not obstruct a direct vertical corridor", () => {
  const edge = { from: "source", to: "target", feedback: false, kind: "activation" };
  const nodes = new Map([
    ["source", { x: 360, y: 414, tensorRank: 0 }],
    ["neighbour", { x: 720, y: 552, tensorRank: 4 }],
    ["target", { x: 360, y: 828, tensorRank: 0 }],
  ]);

  const route = routeGraphEdges([edge], nodes).get(edge);

  assert.deepEqual(route.points, [
    { x: 482, y: 498 },
    { x: 482, y: 828 },
  ]);
});

test("feedback edges remain in side lanes even when the vertical corridor is empty", () => {
  const edge = { from: "cache", to: "attention", feedback: true, kind: "activation" };
  const nodes = new Map([
    ["attention", { x: 360, y: 414, tensorRank: 0 }],
    ["cache", { x: 360, y: 828, tensorRank: 0 }],
  ]);

  const route = routeGraphEdges([edge], nodes).get(edge);

  assert.equal(route.sourceSide, "right");
  assert.equal(route.targetSide, "right");
  assert.ok(route.points.some((point) => point.x > 604));
});

test("cross-column edges turn within the empty column gutter", () => {
  const route = routeGraphEdge(
    { feedback: false, kind: "activation" },
    { x: 360, y: 966, tensorRank: 0 },
    { x: 720, y: 276, tensorRank: 0 }
  );
  assert.equal(route.points[1].x, 662);
  assert.ok(route.points[1].x > 604 && route.points[1].x < 720);
  assert.match(route.path, / Q /);
});

test("cross-column routing moves its lane rather than crossing an intermediate node", () => {
  const edge = { from: "source", to: "target", feedback: false, kind: "activation" };
  const nodes = new Map([
    ["source", { x: 0, y: 0, tensorRank: 0 }],
    ["blocker", { x: 360, y: 100, tensorRank: 3 }],
    ["target", { x: 720, y: 276, tensorRank: 0 }],
  ]);

  const route = routeGraphEdges([edge], nodes).get(edge);
  const resolvedNodes = [...nodes].map(([id, node]) => ({ id, ...node }));

  assert.notEqual(route.routeOffset, 0);
  assert.equal(routeIntersectsNodes(route.points, resolvedNodes, edge), false);
});

test("rounded routes retain a straight terminal arrowhead run", () => {
  const path = roundedOrthogonalPath([
    { x: 604, y: 1146 },
    { x: 652, y: 1146 },
    { x: 652, y: 732 },
    { x: 604, y: 732 }
  ]);
  assert.match(path, /Q 652 732 640 732 L 604 732$/);
});

test("arrows from different origins receive separate parallel lanes", () => {
  const edges = [
    { from: "upper-source", to: "upper-target", feedback: false, kind: "activation" },
    { from: "lower-source", to: "lower-target", feedback: false, kind: "activation" }
  ];
  const nodes = new Map([
    ["upper-source", { x: 360, y: 0, tensorRank: 0 }],
    ["upper-target", { x: 360, y: 414, tensorRank: 0 }],
    ["lower-source", { x: 360, y: 138, tensorRank: 0 }],
    ["lower-target", { x: 360, y: 552, tensorRank: 0 }]
  ]);

  const routes = routeGraphEdges(edges, nodes);
  const upper = routes.get(edges[0]);
  const lower = routes.get(edges[1]);

  assert.notEqual(upper.points[1].x, lower.points[1].x);
  assert.equal(routesShareParallelSegment(upper.points, lower.points), false);
});

test("arrows from different origins enter a shared node at distinct ports", () => {
  const edges = [
    { from: "far-source", to: "shared-target", feedback: false, kind: "activation" },
    { from: "near-source", to: "shared-target", feedback: false, kind: "activation" }
  ];
  const nodes = new Map([
    ["far-source", { x: 0, y: 0, tensorRank: 0 }],
    ["near-source", { x: 360, y: 138, tensorRank: 0 }],
    ["shared-target", { x: 720, y: 276, tensorRank: 0 }]
  ]);

  const routes = routeGraphEdges(edges, nodes);
  const farEndpoint = routes.get(edges[0]).points.at(-1);
  const nearEndpoint = routes.get(edges[1]).points.at(-1);

  assert.equal(farEndpoint.x, 720);
  assert.equal(nearEndpoint.x, 720);
  assert.notEqual(farEndpoint.y, nearEndpoint.y);
  assert.equal(routes.get(edges[0]).targetSide, "left");
  assert.equal(routes.get(edges[1]).targetSide, "left");
});

test("incoming and outgoing arrows with different origins do not reuse a node-side port", () => {
  const edges = [
    { from: "below", to: "junction", feedback: false, kind: "activation" },
    { from: "junction", to: "right", feedback: false, kind: "residual-stream" }
  ];
  const nodes = new Map([
    ["below", { x: 0, y: 276, tensorRank: 0 }],
    ["blocker", { x: 0, y: 138, tensorRank: 0 }],
    ["junction", { x: 0, y: 0, tensorRank: 0 }],
    ["right", { x: 720, y: 0, tensorRank: 0 }]
  ]);

  const routes = routeGraphEdges(edges, nodes);
  const incoming = routes.get(edges[0]);
  const outgoing = routes.get(edges[1]);

  assert.equal(incoming.targetSide, "right");
  assert.equal(outgoing.sourceSide, "right");
  assert.notDeepEqual(incoming.points.at(-1), outgoing.points[0]);
  assert.equal(routesShareParallelSegment(incoming.points, outgoing.points), false);
});
