# Indoor Data Annotation Project

Single source documentation for the current implementation in this folder.

## Overview

This app is a Babylon.js + Vite web tool for:

1. Task 1: building a navigation graph on top of a `.glb` indoor model.
2. Task 2: annotating directed path segments for multiple agent types.

Task 2 is implemented (it is not a placeholder).

## Quick Start

```bash
cd annotator
npm install
npm run dev
```

Open the local Vite URL in Chrome or Edge (File System Access API required).

## Folder And File Conventions

When you select a folder, the app currently looks for the first `.glb` file it can find.

For a model named `building.glb`, the app uses:

- Input model: `building.glb`
- Graph file: `building-graph.json`
- QA file: `building_QA.json`
- Traverse output: `building-traverse.json`

## Task 1: Graph Annotation

Core behaviors:

- Double-click in the viewer to place a node on the current clipping plane.
- Single-click a node to select it.
- Click a second node while one is selected to create an edge.
- Edit/delete nodes from popup or graph list.
- Double-click an edge entry in the graph list to delete that edge.
- Graph auto-saves with debounce to `<model>-graph.json`.

Viewer controls:

- Drag: orbit camera
- Scroll: zoom
- `Back to Top-Down View`: reset to near top-down
- `Adjust Slice Height` slider: horizontal clipping plane (0-200%)
- `Swap Y-Z Axis`: rotate model orientation when needed

## Task 2: Navigability Annotation

Task 2 becomes available when:

- QA file exists (`<model>_QA.json`)
- Graph has at least one node and one edge

Workflow:

- Click `Start Annotation`.
- App iterates by question, agent type, and directed segment.
- Segment nodes are highlighted (route context + current segment emphasis).
- `No` requires a note.
- Annotations are saved to `<model>-traverse.json` after each submit.
- Existing annotations are shown and can be overwritten.
- Questions can be marked as hallucinated and skipped.

Agent test controls:

- Spawn at segment start node
- `W/S`: move forward/back
- `A/D`: turn
- `Arrow Up/Down`: adjust Y
- `Q/E`: previous/next task when no active agent and not typing in an input

Important implementation note:

- Agent movement is currently free movement without collision physics.

## Agent Types (Current `agentParams.js`)

- `HUMAN`: cylinder, auto-approved in Task 2
- `WHEELCHAIR`: box
- `ROBOT`: cylinder
- `SWEEPER`: cylinder
- `QUADRUPEDAL`: box

## Data Formats

Graph format (`<model>-graph.json`):

```json
{
  "nodes": [
    {
      "id": "node_0",
      "name": "Entrance",
      "description": "",
      "position": { "x": 0, "y": 0, "z": 0 }
    }
  ],
  "edges": [
    { "id": "edge_0", "from": "node_0", "to": "node_1" }
  ]
}
```

QA format (`<model>_QA.json`):

```json
{
  "questions": [
    {
      "question": "Can the agent move from entrance to kitchen via hallway?",
      "nodes_involved": {
        "start": "entrance",
        "intermediate": ["hallway"],
        "end": "kitchen",
        "start_detail": "front door",
        "end_detail": "sink area"
      }
    }
  ]
}
```

Traverse format (`<model>-traverse.json`):

```json
{
  "WHEELCHAIR": {
    "node_0|node_3": {
      "traversable": false,
      "note": "Step too high",
      "timestamp": "2026-02-20T00:00:00.000Z",
      "question": "Can the agent move from entrance to kitchen via hallway?"
    }
  },
  "_hallucinatedQuestions": []
}
```

## Project Layout (Active Files)

- `src/main.js`: UI, Task 1 behavior, Task 2 wiring
- `src/annotation.js`: Task 2 workflow + file I/O
- `src/pathfinding.js`: path enumeration + segment extraction
- `src/agents.js`: agent visuals + movement manager
- `src/agentParams.js`: agent definitions
- `src/style.css`: app styles

Legacy files exist in `src/pages`, `src/styles`, and `src/utils/indexedDB.js` but are not used by `index.html` (which loads `src/main.js`).

## Known Limits

- Model loading currently targets `.glb` only.
- The folder loader picks the first `.glb` it finds.
- Path enumeration uses all simple paths; large graphs can become expensive.
