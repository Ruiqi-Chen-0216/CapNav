// ============================================================================
// PATHFINDING MODULE
// ============================================================================
// Implements simple path enumeration (no vertex repetition) for graph traversal

/**
 * Find all simple paths between two nodes in a graph
 * Simple path = no vertex repetition
 *
 * @param {Object} graph - Graph with nodes and edges arrays
 * @param {string} startNodeId - Starting node ID
 * @param {string} endNodeId - Ending node ID
 * @returns {Array} Array of paths, where each path is an array of node IDs
 */
export function findAllSimplePaths(graph, startNodeId, endNodeId) {
  const paths = [];
  const visited = new Set();

  function dfs(currentNodeId, path) {
    // Add current node to path and visited set
    path.push(currentNodeId);
    visited.add(currentNodeId);

    // If we reached the end node, save this path
    if (currentNodeId === endNodeId) {
      paths.push([...path]); // Clone the path
    } else {
      // Find all neighbors of current node
      const neighbors = getNeighbors(graph, currentNodeId);

      // Explore each neighbor
      for (const neighborId of neighbors) {
        if (!visited.has(neighborId)) {
          dfs(neighborId, path);
        }
      }
    }

    // Backtrack: remove current node from path and visited set
    path.pop();
    visited.delete(currentNodeId);
  }

  dfs(startNodeId, []);
  return paths;
}

/**
 * Get all neighbor nodes connected to a given node
 *
 * @param {Object} graph - Graph with nodes and edges arrays
 * @param {string} nodeId - Node ID to find neighbors for
 * @returns {Array} Array of neighbor node IDs
 */
function getNeighbors(graph, nodeId) {
  const neighbors = [];

  for (const edge of graph.edges) {
    // Edges are undirected, so check both directions
    if (edge.from === nodeId) {
      neighbors.push(edge.to);
    } else if (edge.to === nodeId) {
      neighbors.push(edge.from);
    }
  }

  return neighbors;
}

/**
 * Convert a node path to edge segments
 * Example: [A, B, C] → [{from: A, to: B}, {from: B, to: C}]
 *
 * @param {Array} nodePath - Array of node IDs
 * @param {Object} graph - Graph with nodes and edges arrays
 * @returns {Array} Array of edge objects with directional order preserved
 */
export function pathToEdges(nodePath, graph) {
  const edges = [];

  for (let i = 0; i < nodePath.length - 1; i++) {
    const fromId = nodePath[i];
    const toId = nodePath[i + 1];

    // Find the edge between these nodes
    const edge = graph.edges.find(e =>
      (e.from === fromId && e.to === toId) ||
      (e.from === toId && e.to === fromId)
    );

    if (edge) {
      // IMPORTANT: Preserve direction from path (fromId → toId)
      // This ensures we annotate in the correct direction
      const directedEdge = {
        id: edge.id,
        from: fromId,  // Always use the path direction
        to: toId,      // Not normalized alphabetically
      };
      edges.push(directedEdge);
    }
  }

  return edges;
}

/**
 * Find node by name or ID (case-insensitive, partial match for names)
 * Supports multiple formats:
 * - "node_27" (just ID)
 * - "Living room" (just name)
 * - "node_27 — Living room" (ID — name with em dash)
 * - "node_27 - Living room" (ID - name with hyphen)
 *
 * @param {Object} graph - Graph with nodes array
 * @param {string} nodeName - Node name or ID to search for
 * @returns {Object|null} Node object or null if not found
 */
export function findNodeByName(graph, nodeName) {
  const trimmed = nodeName.trim();
  let normalizedName = trimmed.toLowerCase();

  // Check if format is "nodeID — nodename" or "nodeID - nodename"
  // Try em dash first, then regular hyphen
  let nodeIdFromComposite = null;
  if (trimmed.includes(' — ')) {
    // Format: "node_128 — Living room"
    nodeIdFromComposite = trimmed.split(' — ')[0].trim();
  } else if (trimmed.includes(' - ')) {
    // Format: "node_128 - Living room"
    nodeIdFromComposite = trimmed.split(' - ')[0].trim();
  }

  // If we extracted a node ID from composite format, try that first
  if (nodeIdFromComposite) {
    const node = graph.nodes.find(n =>
      n.id && n.id.toLowerCase() === nodeIdFromComposite.toLowerCase()
    );
    if (node) {
      return node;
    }
  }

  // Try exact ID match (for "node_27" format)
  let node = graph.nodes.find(n =>
    n.id && n.id.toLowerCase() === normalizedName
  );

  // If not found, try exact name match
  if (!node) {
    node = graph.nodes.find(n =>
      n.name && n.name.toLowerCase() === normalizedName
    );
  }

  // If not found, try partial name match
  if (!node) {
    node = graph.nodes.find(n =>
      n.name && n.name.toLowerCase().includes(normalizedName)
    );
  }

  return node || null;
}

/**
 * Process a QA question to extract all path segments
 *
 * @param {Object} question - QA question object with nodes_involved
 * @param {Object} graph - Graph with nodes and edges arrays
 * @returns {Object} Result with paths and unique segments, or error
 */
export function processQuestion(question, graph) {
  const { start, intermediate = [], end } = question.nodes_involved;

  // Build sequence of nodes to traverse
  const nodeSequence = [start, ...intermediate, end];

  // Find nodes by name
  const resolvedNodes = [];
  for (const nodeName of nodeSequence) {
    const node = findNodeByName(graph, nodeName);
    if (!node) {
      return {
        error: `Node "${nodeName}" not found in graph`,
        paths: [],
        segments: [],
      };
    }
    resolvedNodes.push(node);
  }

  // Find all paths through the sequence
  const allPaths = [];
  const allEdges = [];

  if (nodeSequence.length === 2) {
    // Direct path from start to end
    const paths = findAllSimplePaths(graph, resolvedNodes[0].id, resolvedNodes[1].id);
    allPaths.push(...paths);
  } else {
    // Path with intermediate nodes: find paths between consecutive pairs
    // Then combine them (Cartesian product)
    const pathSegments = [];

    for (let i = 0; i < resolvedNodes.length - 1; i++) {
      const fromNode = resolvedNodes[i];
      const toNode = resolvedNodes[i + 1];
      const paths = findAllSimplePaths(graph, fromNode.id, toNode.id);

      if (paths.length === 0) {
        return {
          error: `No path found between "${nodeSequence[i]}" and "${nodeSequence[i + 1]}"`,
          paths: [],
          segments: [],
        };
      }

      pathSegments.push(paths);
    }

    // Combine path segments (Cartesian product)
    function combinePathSegments(segments, index = 0, currentPath = []) {
      if (index >= segments.length) {
        // Flatten and deduplicate node IDs
        const flatPath = [];
        currentPath.forEach((segment, i) => {
          if (i === 0) {
            flatPath.push(...segment);
          } else {
            // Skip first node of segment (it's the last node of previous segment)
            flatPath.push(...segment.slice(1));
          }
        });
        allPaths.push(flatPath);
        return;
      }

      for (const path of segments[index]) {
        combinePathSegments(segments, index + 1, [...currentPath, path]);
      }
    }

    combinePathSegments(pathSegments);
  }

  // Convert all paths to edges
  for (const path of allPaths) {
    const edges = pathToEdges(path, graph);
    allEdges.push(...edges);
  }

  // Get unique segments
  const uniqueSegments = [];
  const segmentSet = new Set();

  for (const edge of allEdges) {
    const key = `${edge.from}|${edge.to}`;
    if (!segmentSet.has(key)) {
      segmentSet.add(key);
      uniqueSegments.push(edge);
    }
  }

  return {
    error: null,
    paths: allPaths,
    segments: uniqueSegments,
    nodeSequence: resolvedNodes.map(n => ({ id: n.id, name: n.name })),
  };
}
