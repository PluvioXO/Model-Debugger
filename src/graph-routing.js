const DEFAULT_NODE_WIDTH = 244;
const DEFAULT_NODE_HEIGHT = 84;
const DEFAULT_ROW_STRIDE = 138;
const DEFAULT_STACK_OFFSET = 6;
const DEFAULT_CORNER_RADIUS = 12;
const DEFAULT_EDGE_SPACING = 10;
const DEFAULT_PORT_SPACING = 14;
const DEFAULT_APPROACH_CLEARANCE = 18;
const DEFAULT_APPROACH_SPACING = 5;

export function tensorStackDepth(rank) {
  return rank < 2 ? 0 : Math.min(3, rank - 1);
}

function samePoint(left, right) {
  return left.x === right.x && left.y === right.y;
}

function cleanPoints(points) {
  const unique = [];
  points.forEach((point) => {
    if (!unique.length || !samePoint(unique.at(-1), point)) unique.push(point);
  });
  return unique.filter((point, index) => {
    if (index === 0 || index === unique.length - 1) return true;
    const previous = unique[index - 1];
    const next = unique[index + 1];
    const sameHorizontal = previous.y === point.y && point.y === next.y;
    const sameVertical = previous.x === point.x && point.x === next.x;
    return !sameHorizontal && !sameVertical;
  });
}

function moveToward(from, to, distance) {
  if (from.x === to.x) return { x: from.x, y: from.y + Math.sign(to.y - from.y) * distance };
  return { x: from.x + Math.sign(to.x - from.x) * distance, y: from.y };
}

function segmentLength(from, to) {
  return Math.abs(to.x - from.x) + Math.abs(to.y - from.y);
}

function edgeSegments(points) {
  const cleaned = cleanPoints(points);
  return cleaned.slice(0, -1).map((point, index) => ({
    from: point,
    to: cleaned[index + 1]
  }));
}

export function routeIntersectsNodes(points, nodes, edge, options = {}) {
  const nodeWidth = options.nodeWidth ?? DEFAULT_NODE_WIDTH;
  const nodeHeight = options.nodeHeight ?? DEFAULT_NODE_HEIGHT;
  const stackOffset = options.stackOffset ?? DEFAULT_STACK_OFFSET;
  const clearance = options.nodeClearance ?? 3;
  return edgeSegments(points).some((segment) => nodes.some((node) => {
    if (node.id === edge.from || node.id === edge.to) return false;
    const extent = tensorStackDepth(node.tensorRank ?? 0) * stackOffset;
    const left = node.x - clearance;
    const right = node.x + nodeWidth + extent + clearance;
    const top = node.y - clearance;
    const bottom = node.y + nodeHeight + extent + clearance;
    if (segment.from.x === segment.to.x) {
      return segment.from.x > left
        && segment.from.x < right
        && overlappingRange(segment.from.y, segment.to.y, top, bottom);
    }
    if (segment.from.y === segment.to.y) {
      return segment.from.y > top
        && segment.from.y < bottom
        && overlappingRange(segment.from.x, segment.to.x, left, right);
    }
    return false;
  }));
}

function overlappingRange(firstStart, firstEnd, secondStart, secondEnd) {
  const firstMin = Math.min(firstStart, firstEnd);
  const firstMax = Math.max(firstStart, firstEnd);
  const secondMin = Math.min(secondStart, secondEnd);
  const secondMax = Math.max(secondStart, secondEnd);
  return Math.min(firstMax, secondMax) - Math.max(firstMin, secondMin) > 0.001;
}

export function routesShareParallelSegment(firstPoints, secondPoints) {
  return edgeSegments(firstPoints).some((first) => {
    const firstHorizontal = first.from.y === first.to.y;
    const firstVertical = first.from.x === first.to.x;
    if (!firstHorizontal && !firstVertical) return false;
    return edgeSegments(secondPoints).some((second) => {
      if (
        firstHorizontal &&
        second.from.y === second.to.y &&
        first.from.y === second.from.y
      ) {
        return overlappingRange(first.from.x, first.to.x, second.from.x, second.to.x);
      }
      if (
        firstVertical &&
        second.from.x === second.to.x &&
        first.from.x === second.from.x
      ) {
        return overlappingRange(first.from.y, first.to.y, second.from.y, second.to.y);
      }
      return false;
    });
  });
}

function routeKind(edge, source, target, options = {}) {
  const rowStride = options.rowStride ?? DEFAULT_ROW_STRIDE;
  const nodeWidth = options.nodeWidth ?? DEFAULT_NODE_WIDTH;
  const nodeHeight = options.nodeHeight ?? DEFAULT_NODE_HEIGHT;
  if (source.x === target.x) {
    const directVertical = options.directVertical
      ?? Math.abs(target.y - source.y) <= rowStride * 1.1;
    return !edge.feedback && directVertical ? "vertical-direct" : "vertical-lane";
  }
  const sourceCenterY = source.y + nodeHeight / 2;
  const targetCenterY = target.y + nodeHeight / 2;
  if (
    sourceCenterY === targetCenterY &&
    (edge.kind === "residual-stream" || Math.abs(target.x - source.x) <= nodeWidth * 2)
  ) {
    return "horizontal-direct";
  }
  return "cross-column";
}

function endpointSides(edge, source, target, options = {}) {
  const kind = routeKind(edge, source, target, options);
  if (kind === "vertical-direct") {
    const downward = target.y >= source.y;
    return {
      source: downward ? "bottom" : "top",
      target: downward ? "top" : "bottom"
    };
  }
  if (kind === "vertical-lane") return { source: "right", target: "right" };
  const rightward = target.x > source.x;
  return {
    source: rightward ? "right" : "left",
    target: rightward ? "left" : "right"
  };
}

function centeredPortOffset(index, count, availableSpan, spacing) {
  if (count < 2) return 0;
  const actualSpacing = Math.min(spacing, availableSpan / (count - 1));
  return (index - (count - 1) / 2) * actualSpacing;
}

function routeOffsetForAttempt(attempt, spacing) {
  if (attempt === 0) return 0;
  const step = Math.ceil(attempt / 2) * spacing;
  return attempt % 2 === 1 ? step : -step;
}

function approachDistance(clearance, offset) {
  return Math.max(6, clearance - offset);
}

export function roundedOrthogonalPath(inputPoints, radius = DEFAULT_CORNER_RADIUS) {
  const points = cleanPoints(inputPoints);
  if (points.length < 2) return "";
  let path = `M ${points[0].x} ${points[0].y}`;
  for (let index = 1; index < points.length - 1; index += 1) {
    const previous = points[index - 1];
    const corner = points[index];
    const next = points[index + 1];
    const cornerRadius = Math.min(
      radius,
      segmentLength(previous, corner) / 2,
      segmentLength(corner, next) / 2
    );
    const before = moveToward(corner, previous, cornerRadius);
    const after = moveToward(corner, next, cornerRadius);
    path += ` L ${before.x} ${before.y} Q ${corner.x} ${corner.y} ${after.x} ${after.y}`;
  }
  const last = points.at(-1);
  return `${path} L ${last.x} ${last.y}`;
}

export function routeGraphEdge(edge, source, target, options = {}) {
  const nodeWidth = options.nodeWidth ?? DEFAULT_NODE_WIDTH;
  const nodeHeight = options.nodeHeight ?? DEFAULT_NODE_HEIGHT;
  const rowStride = options.rowStride ?? DEFAULT_ROW_STRIDE;
  const stackOffset = options.stackOffset ?? DEFAULT_STACK_OFFSET;
  const routeOffset = options.routeOffset ?? 0;
  const sourcePortOffset = options.sourcePortOffset ?? 0;
  const targetPortOffset = options.targetPortOffset ?? 0;
  const sourceApproachOffset = options.sourceApproachOffset ?? 0;
  const targetApproachOffset = options.targetApproachOffset ?? 0;
  const approachClearance = options.approachClearance ?? DEFAULT_APPROACH_CLEARANCE;
  const sourceExtent = tensorStackDepth(source.tensorRank ?? 0) * stackOffset;
  const targetExtent = tensorStackDepth(target.tensorRank ?? 0) * stackOffset;
  const sourceCenter = { x: source.x + nodeWidth / 2, y: source.y + nodeHeight / 2 };
  const targetCenter = { x: target.x + nodeWidth / 2, y: target.y + nodeHeight / 2 };
  let points;
  let labelPosition;

  if (source.x === target.x) {
    const directVertical = routeKind(edge, source, target, options) === "vertical-direct";
    if (directVertical) {
      const downward = target.y >= source.y;
      const direction = downward ? 1 : -1;
      const start = downward
        ? { x: sourceCenter.x + sourcePortOffset, y: source.y + nodeHeight + sourceExtent }
        : { x: sourceCenter.x + sourcePortOffset, y: source.y };
      const end = downward
        ? { x: targetCenter.x + targetPortOffset, y: target.y }
        : { x: targetCenter.x + targetPortOffset, y: target.y + nodeHeight + targetExtent };
      points = downward
        ? [start, end]
        : [start, end];
      if (routeOffset || sourcePortOffset || targetPortOffset) {
        const trackX = sourceCenter.x + sourcePortOffset + routeOffset;
        const sourceApproachY = start.y + direction * approachDistance(approachClearance, sourceApproachOffset);
        const approachY = end.y - direction * approachDistance(approachClearance, targetApproachOffset);
        points = [
          start,
          { x: start.x, y: sourceApproachY },
          { x: trackX, y: sourceApproachY },
          { x: trackX, y: approachY },
          { x: end.x, y: approachY },
          end
        ];
      }
      labelPosition = {
        x: sourceCenter.x + sourcePortOffset + routeOffset + 8,
        y: (start.y + end.y) / 2 - 4
      };
    } else {
      const outerRight = Math.max(
        source.x + nodeWidth + sourceExtent,
        target.x + nodeWidth + targetExtent
      );
      const laneX = outerRight + (edge.feedback ? 48 : 24) + routeOffset;
      const start = {
        x: source.x + nodeWidth + sourceExtent,
        y: sourceCenter.y + sourcePortOffset
      };
      const end = {
        x: target.x + nodeWidth + targetExtent,
        y: targetCenter.y + targetPortOffset
      };
      points = [start, { x: laneX, y: start.y }, { x: laneX, y: end.y }, end];
      labelPosition = { x: laneX + 8, y: (start.y + end.y) / 2 };
    }
  } else {
    const rightward = target.x > source.x;
    const start = rightward
      ? { x: source.x + nodeWidth + sourceExtent, y: sourceCenter.y + sourcePortOffset }
      : { x: source.x, y: sourceCenter.y + sourcePortOffset };
    const end = rightward
      ? { x: target.x, y: targetCenter.y + targetPortOffset }
      : { x: target.x + nodeWidth + targetExtent, y: targetCenter.y + targetPortOffset };

    const directHorizontal = sourceCenter.y === targetCenter.y && (
      edge.kind === "residual-stream" || Math.abs(target.x - source.x) <= nodeWidth * 2
    );
    if (directHorizontal) {
      if (!routeOffset && !sourcePortOffset && !targetPortOffset) {
        points = [start, end];
      } else {
        const direction = rightward ? 1 : -1;
        const startLeadX = start.x + direction * approachDistance(approachClearance, sourceApproachOffset);
        const endLeadX = end.x - direction * approachDistance(approachClearance, targetApproachOffset);
        const trackY = start.y + routeOffset;
        const targetTrackY = end.y + routeOffset;
        points = [
          start,
          { x: startLeadX, y: start.y },
          { x: startLeadX, y: trackY },
          { x: endLeadX, y: trackY },
          { x: endLeadX, y: targetTrackY },
          { x: endLeadX, y: end.y },
          end
        ];
      }
      labelPosition = {
        x: (start.x + end.x) / 2,
        y: (start.y + routeOffset) - 8
      };
    } else {
      const direction = rightward ? 1 : -1;
      const laneX = (start.x + end.x) / 2 + routeOffset;
      if (!routeOffset && !sourcePortOffset && !targetPortOffset) {
        points = [start, { x: laneX, y: start.y }, { x: laneX, y: end.y }, end];
      } else {
        const startLeadX = start.x + direction * approachDistance(approachClearance, sourceApproachOffset);
        const endLeadX = end.x - direction * approachDistance(approachClearance, targetApproachOffset);
        const startTrackY = start.y + routeOffset;
        const endTrackY = end.y + routeOffset;
        points = [
          start,
          { x: startLeadX, y: start.y },
          { x: startLeadX, y: startTrackY },
          { x: laneX, y: startTrackY },
          { x: laneX, y: endTrackY },
          { x: endLeadX, y: endTrackY },
          { x: endLeadX, y: end.y },
          end
        ];
      }
      labelPosition = {
        x: laneX + (rightward ? 8 : -8),
        y: (start.y + end.y) / 2 + routeOffset - 4
      };
    }
  }

  return {
    path: roundedOrthogonalPath(points),
    labelPosition,
    points: cleanPoints(points),
    routeOffset,
    sourcePortOffset,
    targetPortOffset
  };
}

function resolvedEndpoint(entry, options) {
  return {
    x: entry.x,
    y: entry.y,
    tensorRank: entry.tensorRank ?? options.tensorRank?.(entry) ?? 0
  };
}

function verticalCorridorIsClear(edge, source, target, nodes, options) {
  if (edge.feedback || source.x !== target.x) return false;
  const nodeWidth = options.nodeWidth ?? DEFAULT_NODE_WIDTH;
  const nodeHeight = options.nodeHeight ?? DEFAULT_NODE_HEIGHT;
  const stackOffset = options.stackOffset ?? DEFAULT_STACK_OFFSET;
  const sourceExtent = tensorStackDepth(source.tensorRank ?? 0) * stackOffset;
  const targetExtent = tensorStackDepth(target.tensorRank ?? 0) * stackOffset;
  const downward = target.y >= source.y;
  const startY = downward ? source.y + nodeHeight + sourceExtent : source.y;
  const endY = downward ? target.y : target.y + nodeHeight + targetExtent;
  const corridorTop = Math.min(startY, endY);
  const corridorBottom = Math.max(startY, endY);
  if (corridorBottom - corridorTop <= 0.001) return true;
  return !nodes.some((node) => {
    if (node.id === edge.from || node.id === edge.to) return false;
    const extent = tensorStackDepth(node.tensorRank ?? 0) * stackOffset;
    const horizontallyBlocks = overlappingRange(
      source.x,
      source.x + nodeWidth + Math.max(sourceExtent, targetExtent),
      node.x,
      node.x + nodeWidth + extent
    );
    const verticallyBlocks = overlappingRange(
      corridorTop,
      corridorBottom,
      node.y,
      node.y + nodeHeight + extent
    );
    return horizontallyBlocks && verticallyBlocks;
  });
}

function compareEndpointOrigins(left, right, side) {
  const verticalSide = side === "left" || side === "right";
  const leftOther = left.endpoint === "source" ? left.descriptor.target : left.descriptor.source;
  const rightOther = right.endpoint === "source" ? right.descriptor.target : right.descriptor.source;
  const primary = verticalSide ? leftOther.y - rightOther.y : leftOther.x - rightOther.x;
  if (primary) return primary;
  const secondary = verticalSide ? leftOther.x - rightOther.x : leftOther.y - rightOther.y;
  return secondary || left.descriptor.edge.from.localeCompare(right.descriptor.edge.from);
}

function endpointPortAssignments(descriptors, options) {
  const nodeWidth = options.nodeWidth ?? DEFAULT_NODE_WIDTH;
  const nodeHeight = options.nodeHeight ?? DEFAULT_NODE_HEIGHT;
  const portSpacing = options.portSpacing ?? DEFAULT_PORT_SPACING;
  const approachSpacing = options.approachSpacing ?? DEFAULT_APPROACH_SPACING;
  const groups = new Map();
  descriptors.forEach((descriptor) => {
    const sides = endpointSides(descriptor.edge, descriptor.source, descriptor.target, {
      ...options,
      directVertical: descriptor.directVertical
    });
    descriptor.sourceSide = sides.source;
    descriptor.targetSide = sides.target;
    [
      { endpoint: "source", nodeId: descriptor.edge.from, side: sides.source },
      { endpoint: "target", nodeId: descriptor.edge.to, side: sides.target }
    ].forEach((record) => {
      const key = `${record.nodeId}:${record.side}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push({ ...record, descriptor });
    });
  });

  const assignments = new Map();
  groups.forEach((group) => {
    const side = group[0].side;
    const byOrigin = new Map();
    group.forEach((record) => {
      const origin = record.descriptor.edge.from;
      if (!byOrigin.has(origin)) byOrigin.set(origin, record);
    });
    const origins = [...byOrigin.values()].sort((left, right) => compareEndpointOrigins(left, right, side));
    const originIndex = new Map(origins.map((record, index) => [record.descriptor.edge.from, index]));
    const availableSpan = side === "left" || side === "right"
      ? nodeHeight - 28
      : nodeWidth - 36;
    group.forEach((record) => {
      const edge = record.descriptor.edge;
      const index = originIndex.get(edge.from);
      const assignment = assignments.get(edge) ?? {
        sourceOffset: 0,
        sourceApproachOffset: 0,
        targetOffset: 0,
        targetApproachOffset: 0
      };
      assignment[`${record.endpoint}Offset`] = centeredPortOffset(
        index,
        origins.length,
        availableSpan,
        portSpacing
      );
      assignment[`${record.endpoint}ApproachOffset`] = index * approachSpacing;
      assignments.set(edge, assignment);
    });
  });
  return assignments;
}

export function routeGraphEdges(edges, byId, options = {}) {
  const nodes = [...byId.entries()].map(([id, entry]) => ({
    id,
    ...resolvedEndpoint(entry, options)
  }));
  const descriptors = edges.map((edge) => {
    const sourceEntry = byId.get(edge.from);
    const targetEntry = byId.get(edge.to);
    if (!sourceEntry || !targetEntry) return null;
    const source = resolvedEndpoint(sourceEntry, options);
    const target = resolvedEndpoint(targetEntry, options);
    return {
      edge,
      source,
      target,
      directVertical: verticalCorridorIsClear(edge, source, target, nodes, options),
      sourceSide: "",
      targetSide: ""
    };
  }).filter(Boolean);
  const portAssignments = endpointPortAssignments(descriptors, options);
  const edgeSpacing = options.edgeSpacing ?? DEFAULT_EDGE_SPACING;
  const maxAttempts = Math.max(65, descriptors.length * 2 + 1);
  const routed = [];
  const routes = new Map();

  descriptors.forEach((descriptor) => {
    const port = portAssignments.get(descriptor.edge) ?? {
      sourceOffset: 0,
      sourceApproachOffset: 0,
      targetOffset: 0,
      targetApproachOffset: 0
    };
    let geometry;
    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
      const routeOffset = routeOffsetForAttempt(attempt, edgeSpacing);
      const candidate = routeGraphEdge(descriptor.edge, descriptor.source, descriptor.target, {
        ...options,
        directVertical: descriptor.directVertical,
        routeOffset,
        sourcePortOffset: port.sourceOffset,
        sourceApproachOffset: port.sourceApproachOffset,
        targetPortOffset: port.targetOffset,
        targetApproachOffset: port.targetApproachOffset
      });
      const conflicts = routed.some((previous) => (
        previous.edge.from !== descriptor.edge.from &&
        routesShareParallelSegment(candidate.points, previous.geometry.points)
      ));
      const crossesNode = routeIntersectsNodes(candidate.points, nodes, descriptor.edge, options);
      if (!conflicts && !crossesNode) {
        geometry = candidate;
        break;
      }
    }
    if (!geometry) {
      throw new Error(`Could not allocate a non-overlapping route for ${descriptor.edge.from} → ${descriptor.edge.to}`);
    }
    geometry.sourceSide = descriptor.sourceSide;
    geometry.targetSide = descriptor.targetSide;
    routes.set(descriptor.edge, geometry);
    routed.push({ edge: descriptor.edge, geometry });
  });
  return routes;
}
