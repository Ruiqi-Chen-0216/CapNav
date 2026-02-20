import './style.css';
import * as BABYLON from '@babylonjs/core';
import '@babylonjs/loaders';
import { AgentManager } from './agents.js';
import { AnnotationManager } from './annotation.js';
import { processQuestion } from './pathfinding.js';

// ============================================================================
// GLOBAL STATE
// ============================================================================
let canvas;
let engine;
let scene;
let camera;
let modelMeshes = [];
let modelBoundingBox = null;
let clippingPlane = null;
let highlightLayer = null;

// Graph data
let graph = {
  nodes: [],
  edges: []
};

// Selection state
let selectedNode = null;

// File handle for saving
let directoryHandle = null;
let modelFileName = '';

// ID counters
let nodeIdCounter = 0;
let edgeIdCounter = 0;

// Y-Z swap state
let isYZSwapped = false;
let modelRootTransform = null;

// Save debounce
let saveTimeout = null;

// Click/double-click handling
let clickTimeout = null;
let clickPreventedByDoubleClick = false;

// Task 2: Navigability Annotation
let agentManager = null;
let annotationManager = null;
let lastFrameTime = Date.now();

// ============================================================================
// INITIALIZATION
// ============================================================================
document.querySelector('#app').innerHTML = `
  <div class="container">
    <!-- Left Panel: 3D Viewer -->
    <div class="viewer-panel">
      <div class="controls-bar">
        <button id="topDownBtn" class="control-btn">Back to Top-Down View</button>
        <div class="slider-group">
          <label for="sliceSlider">Adjust Slice Height:</label>
          <input type="range" id="sliceSlider" min="0" max="200" value="100" step="0.1">
          <span id="sliceValue">100%</span>
        </div>
      </div>
      <canvas id="renderCanvas"></canvas>
      <div class="help-text">
        Double-click: Create node | Click node: Select | Click 2 nodes: Create edge | Click canvas: Deselect | Drag: Rotate | Scroll: Zoom
      </div>
    </div>

    <!-- Right Panel: Controls and Graph View -->
    <div class="info-panel">
      <h2>Indoor Data Annotation</h2>

      <div class="section">
        <h3>Load Model</h3>
        <button id="loadFolderBtn" class="primary-btn">Select Folder</button>
        <div id="modelInfo" class="info-text">No model loaded</div>
      </div>

      <div class="section">
        <h3>Model Orientation</h3>
        <button id="swapYZBtn" class="secondary-btn" disabled>Swap Y-Z Axis</button>
        <div class="info-text small">Use this if model appears sideways</div>
      </div>

      <div class="section">
        <h3>Task 1: Graph Annotation</h3>
        <div class="stats">
          <div class="stat-item">
            <span class="stat-label">Nodes:</span>
            <span id="nodeCount" class="stat-value">0</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Edges:</span>
            <span id="edgeCount" class="stat-value">0</span>
          </div>
        </div>
      </div>

      <div class="section">
        <h3>Graph View</h3>
        <div id="graphList" class="graph-list"></div>
      </div>

      <div class="section" id="task2Section">
        <h3>Task 2: Navigability Annotation</h3>
        <div id="task2Status" class="info-text">
          <p>Load a folder with QA file to begin</p>
        </div>
        <button id="startAnnotationBtn" class="primary-btn" style="display: none;">Start Annotation</button>

        <div id="annotationPanel" style="display: none;">
          <div class="annotation-progress">
            <div class="progress-text">
              Question <span id="questionNum">0</span>/<span id="questionTotal">0</span> |
              Progress: <span id="annotationProgress">0/0</span>
            </div>
          </div>

          <div class="current-question">
            <h4>Current Question:</h4>
            <p id="questionText"></p>
            <button id="hallucinatedQuestionBtn" class="danger-btn" style="margin-top: 10px;">⚠ Mark as Hallucinated Question</button>
          </div>

          <div class="current-agent">
            <h4>Agent:</h4>
            <p id="agentName"></p>
          </div>

          <div class="current-segment">
            <h4>Path Segment: <span id="annotationStatus" style="font-weight: normal; color: #666;"></span></h4>
            <p><span id="segmentFrom"></span> → <span id="segmentTo"></span></p>
            <p class="info-text small">Segment <span id="segmentNum">0</span>/<span id="segmentTotal">0</span></p>
          </div>

          <div class="annotation-controls">
            <h4>Can the agent traverse this path?</h4>
            <div class="button-group">
              <button id="annotateYesBtn" class="primary-btn">✓ Yes</button>
              <button id="annotateNoBtn" class="danger-btn">✗ No</button>
              <button id="goBackBtn" class="secondary-btn">← Go Back</button>
              <button id="goNextBtn" class="secondary-btn" style="display: none;">Next →</button>
            </div>
            <div id="existingAnnotation" style="display: none; margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px;">
              <strong>Previous annotation:</strong>
              <div id="existingAnnotationContent"></div>
            </div>
          </div>

          <div class="form-group">
            <label for="annotationNote">Note (required if "No"):</label>
            <textarea id="annotationNote" placeholder="Add explanation (required when marking as not traversable)..." rows="2"></textarea>
          </div>

          <div class="agent-controls">
            <h4>Optional: Test with Agent</h4>
            <button id="spawnAgentBtn" class="secondary-btn">Spawn Agent at Start</button>
            <button id="removeAgentBtn" class="secondary-btn" style="display: none;">Remove Agent</button>
            <div class="help-text small">
              Controls: W/S - Forward/Back | A/D - Turn Left/Right
            </div>
          </div>
        </div>
      </div>

      <div class="auto-save-status" id="saveStatus">Auto-save: Ready</div>
    </div>
  </div>

  <!-- Node Edit Popup -->
  <div id="nodePopup" class="popup-overlay" style="display: none;">
    <div class="popup-content">
      <h3>Edit Node</h3>
      <div class="form-group">
        <label for="nodeName">Name:</label>
        <input type="text" id="nodeName" placeholder="Enter node name">
      </div>
      <div class="form-group">
        <label for="nodeDesc">Description:</label>
        <textarea id="nodeDesc" placeholder="Enter description" rows="3"></textarea>
      </div>
      <div class="popup-actions">
        <button id="saveNodeBtn" class="primary-btn">Save</button>
        <button id="deleteNodeBtn" class="danger-btn">Delete</button>
        <button id="closePopupBtn" class="secondary-btn">Close</button>
      </div>
    </div>
  </div>
`;

// Initialize Babylon.js
initBabylon();
setupEventListeners();

// ============================================================================
// BABYLON.JS SETUP
// ============================================================================
function initBabylon() {
  canvas = document.getElementById('renderCanvas');
  engine = new BABYLON.Engine(canvas, true, {
    preserveDrawingBuffer: true,
    stencil: true
  });

  // Create scene
  scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color3(0.2, 0.2, 0.2);

  // Create highlight layer for green outline on selected nodes
  highlightLayer = new BABYLON.HighlightLayer("highlightLayer", scene);
  highlightLayer.outerGlow = true;
  highlightLayer.innerGlow = false;

  // Create camera - 45 degree downward view
  camera = new BABYLON.ArcRotateCamera(
    "camera",
    0, // alpha (horizontal rotation)
    Math.PI / 4, // beta (45 degrees from vertical)
    10, // radius
    BABYLON.Vector3.Zero(),
    scene
  );
  camera.attachControl(canvas, true);
  camera.wheelPrecision = 50;
  camera.minZ = 0.1;
  camera.lowerRadiusLimit = 1;
  camera.upperRadiusLimit = 100;

  // Lighting - bright setup for indoor scenes
  const hemisphericLight = new BABYLON.HemisphericLight(
    "hemisphericLight",
    new BABYLON.Vector3(0, 1, 0),
    scene
  );
  hemisphericLight.intensity = 1.5;
  hemisphericLight.groundColor = new BABYLON.Color3(0.5, 0.5, 0.5); // Reduce shadow darkness

  // Add directional light for better definition
  const directionalLight = new BABYLON.DirectionalLight(
    "directionalLight",
    new BABYLON.Vector3(-1, -2, -1),
    scene
  );
  directionalLight.intensity = 0.5;

  // Initialize AgentManager for Task 2
  agentManager = new AgentManager(scene);

  // Enable agents (physics disabled for free movement)
  agentManager.enablePhysics().then(() => {
    console.log('Agent system initialized (free movement mode)');
  });

  // Start render loop
  engine.runRenderLoop(() => {
    // Update physics (Task 2)
    const currentTime = Date.now();
    const deltaTime = (currentTime - lastFrameTime) / 1000; // Convert to seconds
    lastFrameTime = currentTime;

    if (agentManager) {
      agentManager.update(deltaTime);
    }

    scene.render();
  });

  window.addEventListener('resize', () => {
    engine.resize();
  });
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================
function setupEventListeners() {
  // Load folder button
  document.getElementById('loadFolderBtn').addEventListener('click', loadFolder);

  // Top-down view button
  document.getElementById('topDownBtn').addEventListener('click', setTopDownView);

  // Swap Y-Z axis button
  document.getElementById('swapYZBtn').addEventListener('click', toggleYZSwap);

  // Slice height slider
  document.getElementById('sliceSlider').addEventListener('input', (e) => {
    const percentage = e.target.value;
    document.getElementById('sliceValue').textContent = percentage + '%';
    updateSliceHeight(percentage);
  });

  // Canvas interactions
  canvas.addEventListener('dblclick', handleDoubleClick);
  canvas.addEventListener('click', handleClick);
  canvas.addEventListener('contextmenu', handleRightClick);

  // Popup buttons
  document.getElementById('saveNodeBtn').addEventListener('click', saveNodeEdit);
  document.getElementById('deleteNodeBtn').addEventListener('click', deleteNode);
  document.getElementById('closePopupBtn').addEventListener('click', closePopup);
  document.getElementById('nodePopup').addEventListener('click', (e) => {
    if (e.target.id === 'nodePopup') closePopup();
  });

  // Task 2: Annotation buttons
  document.getElementById('startAnnotationBtn').addEventListener('click', startAnnotation);
  document.getElementById('annotateYesBtn').addEventListener('click', () => submitAnnotation(true));
  document.getElementById('annotateNoBtn').addEventListener('click', () => submitAnnotation(false));
  document.getElementById('goBackBtn').addEventListener('click', goBackAnnotation);
  document.getElementById('goNextBtn').addEventListener('click', goNextAnnotation);
  document.getElementById('hallucinatedQuestionBtn').addEventListener('click', markQuestionHallucinated);
  document.getElementById('spawnAgentBtn').addEventListener('click', spawnAgentAtStart);
  document.getElementById('removeAgentBtn').addEventListener('click', removeAgent);

  // Keyboard controls for agent movement and annotation navigation
  window.addEventListener('keydown', handleKeyboardControls);
}

// ============================================================================
// FOLDER LOADING
// ============================================================================
async function loadFolder() {
  try {
    // Request folder access
    directoryHandle = await window.showDirectoryPicker();

    // Find .glb file
    let glbFile = null;
    for await (const entry of directoryHandle.values()) {
      if (entry.kind === 'file' && entry.name.toLowerCase().endsWith('.glb')) {
        const fileHandle = await directoryHandle.getFileHandle(entry.name);
        glbFile = await fileHandle.getFile();
        modelFileName = entry.name;
        break;
      }
    }

    if (!glbFile) {
      alert('No .glb file found in the selected folder');
      return;
    }

    // Load the model
    await loadModel(glbFile);

    // Try to load existing graph data
    await loadGraphData();

    // Initialize Task 2: Annotation Manager
    await initializeTask2();

    // Update UI
    const nodeCount = graph.nodes.length;
    const edgeCount = graph.edges.length;
    if (nodeCount > 0 || edgeCount > 0) {
      document.getElementById('modelInfo').textContent = `Loaded: ${modelFileName} (${nodeCount} nodes, ${edgeCount} edges)`;
    } else {
      document.getElementById('modelInfo').textContent = `Loaded: ${modelFileName}`;
    }

  } catch (error) {
    console.error('Error loading folder:', error);
    if (error.name !== 'AbortError') {
      alert('Failed to load folder: ' + error.message);
    }
  }
}

// ============================================================================
// GRAPH DATA LOADING
// ============================================================================
async function loadGraphData() {
  if (!directoryHandle || !modelFileName) return;

  try {
    // Construct the graph file name
    const graphFileName = modelFileName.replace('.glb', '-graph.json');

    // Try to get the graph file
    const graphFileHandle = await directoryHandle.getFileHandle(graphFileName);
    const graphFile = await graphFileHandle.getFile();
    const graphText = await graphFile.text();
    const loadedGraph = JSON.parse(graphText);

    console.log('Loaded existing graph data:', loadedGraph);

    // Update the graph data
    graph.nodes = loadedGraph.nodes || [];
    graph.edges = loadedGraph.edges || [];

    // Update node ID counter to avoid conflicts
    if (graph.nodes.length > 0) {
      const maxNodeId = Math.max(...graph.nodes.map(n => {
        const match = n.id.match(/node_(\d+)/);
        return match ? parseInt(match[1]) : 0;
      }));
      nodeIdCounter = maxNodeId + 1;
    }

    // Update edge ID counter to avoid conflicts
    if (graph.edges.length > 0) {
      const maxEdgeId = Math.max(...graph.edges.map(e => {
        const match = e.id.match(/edge_(\d+)/);
        return match ? parseInt(match[1]) : 0;
      }));
      edgeIdCounter = maxEdgeId + 1;
    }

    // Recreate all node and edge meshes
    graph.nodes.forEach(node => {
      createNodeMesh(node);
    });

    graph.edges.forEach(edge => {
      createEdgeMesh(edge);
    });

    // Update the UI
    updateGraphView();

    console.log(`Loaded ${graph.nodes.length} nodes and ${graph.edges.length} edges`);
  } catch (error) {
    // File doesn't exist or couldn't be read - this is fine for new annotations
    if (error.name === 'NotFoundError') {
      console.log('No existing graph file found - starting fresh');
    } else {
      console.error('Error loading graph data:', error);
    }
  }
}

// ============================================================================
// PHYSICS FOR MODEL (DISABLED)
// ============================================================================
function addPhysicsToModel(meshes) {
  // Physics disabled - using free movement mode
  console.log('Physics disabled - agents use free movement without collision');
}

// ============================================================================
// MODEL LOADING
// ============================================================================
async function loadModel(file) {
  // Clear previous model
  modelMeshes.forEach(mesh => {
    mesh.dispose();
  });
  modelMeshes = [];
  if (modelRootTransform) {
    modelRootTransform.dispose();
    modelRootTransform = null;
  }
  graph.nodes.forEach(node => {
    const sphere = scene.getMeshByName(node.id);
    if (sphere) sphere.dispose();
  });
  graph.edges.forEach(edge => {
    const line = scene.getMeshByName(edge.id);
    if (line) line.dispose();
  });
  // Clear arrays instead of creating new object to preserve references
  graph.nodes.length = 0;
  graph.edges.length = 0;
  isYZSwapped = false;
  updateGraphView();

  // Create blob URL
  const url = URL.createObjectURL(file);

  return new Promise((resolve, reject) => {
    BABYLON.SceneLoader.ImportMesh(
      "",
      "",
      url,
      scene,
      (meshes) => {
        console.log('Model loaded:', meshes.length, 'meshes');

        modelMeshes = meshes;

        // Add physics bodies to model meshes for collision
        addPhysicsToModel(meshes);

        // Calculate bounding box
        calculateBoundingBox(meshes);

        // Setup camera
        if (modelBoundingBox) {
          const center = modelBoundingBox.center;
          const size = modelBoundingBox.size;
          const maxDim = Math.max(size.x, size.y, size.z);

          camera.target = center;
          camera.radius = maxDim * 2.5;
          camera.upperRadiusLimit = maxDim * 10;
          camera.lowerRadiusLimit = maxDim * 0.1;

          // Set 45 degree view
          camera.alpha = 0;
          camera.beta = Math.PI / 4;

          // Setup clipping plane
          setupClippingPlane();
        }

        // Enable Y-Z swap button
        document.getElementById('swapYZBtn').disabled = false;

        URL.revokeObjectURL(url);
        resolve();
      },
      null,
      (scene, message) => {
        console.error('Error loading model:', message);
        URL.revokeObjectURL(url);
        reject(new Error(message));
      },
      '.glb'
    );
  });
}

// ============================================================================
// BOUNDING BOX CALCULATION
// ============================================================================
function calculateBoundingBox(meshes) {
  let minBounds = new BABYLON.Vector3(Infinity, Infinity, Infinity);
  let maxBounds = new BABYLON.Vector3(-Infinity, -Infinity, -Infinity);

  meshes.forEach(mesh => {
    if (mesh.getTotalVertices() > 0) {
      mesh.computeWorldMatrix(true);
      const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);

      if (positions) {
        const worldMatrix = mesh.getWorldMatrix();

        for (let i = 0; i < positions.length; i += 3) {
          const vertex = new BABYLON.Vector3(positions[i], positions[i + 1], positions[i + 2]);
          const worldVertex = BABYLON.Vector3.TransformCoordinates(vertex, worldMatrix);

          minBounds = BABYLON.Vector3.Minimize(minBounds, worldVertex);
          maxBounds = BABYLON.Vector3.Maximize(maxBounds, worldVertex);
        }
      }
    }
  });

  if (isFinite(minBounds.x)) {
    const center = BABYLON.Vector3.Center(minBounds, maxBounds);
    const size = maxBounds.subtract(minBounds);

    modelBoundingBox = {
      min: minBounds,
      max: maxBounds,
      center: center,
      size: size
    };

    console.log('Bounding box calculated:', modelBoundingBox);
  } else {
    console.error('Invalid bounding box');
    modelBoundingBox = null;
  }
}

// ============================================================================
// CAMERA CONTROLS
// ============================================================================
function setTopDownView() {
  camera.alpha = 0;
  camera.beta = 0.01; // Almost top-down
}

function setupClippingPlane() {
  if (!modelBoundingBox) return;

  // ALWAYS use world Y-axis for horizontal slicing (Babylon's up direction)
  // The clipping plane works in world space, not model local space
  const maxY = modelBoundingBox.max.y;

  // Create clipping plane pointing UP along world Y-axis
  // Start at max to show entire model (100%)
  clippingPlane = new BABYLON.Plane(0, 1, 0, -maxY);
  scene.clipPlane = clippingPlane;

  // Set slider to 100% (show entire model)
  document.getElementById('sliceSlider').value = 100;
  document.getElementById('sliceValue').textContent = '100%';

  console.log('Clipping plane setup - World Y range:', modelBoundingBox.min.y, 'to', maxY);
}

function updateSliceHeight(percentage) {
  if (!modelBoundingBox || !clippingPlane) return;

  // ALWAYS use world Y-axis (vertical in Babylon world space)
  const minY = modelBoundingBox.min.y;
  const maxY = modelBoundingBox.max.y;
  const range = maxY - minY;

  // Calculate height (0% = bottom, 100% = top, 200% = double height above)
  // This allows revealing models that extend beyond initial bounds after axis swap
  const height = minY + range * (percentage / 100);

  // Update plane distance (negative because plane points up)
  clippingPlane.d = -height;
}

function toggleYZSwap() {
  if (!modelMeshes.length) return;

  isYZSwapped = !isYZSwapped;

  // Create or remove root transform
  if (isYZSwapped) {
    // Create transform node and rotate
    modelRootTransform = new BABYLON.TransformNode("modelRoot", scene);
    modelRootTransform.rotation.x = -Math.PI / 2; // Rotate -90° around X to swap Y-Z

    // Parent all meshes to transform
    modelMeshes.forEach(mesh => {
      if (!mesh.parent || mesh.parent === scene) {
        mesh.parent = modelRootTransform;
      }
    });
  } else {
    // Remove transform - unparent meshes
    modelMeshes.forEach(mesh => {
      if (mesh.parent === modelRootTransform) {
        mesh.parent = null;
      }
    });

    if (modelRootTransform) {
      modelRootTransform.dispose();
      modelRootTransform = null;
    }
  }

  // Recalculate bounding box with new orientation
  const originalMin = modelBoundingBox.min.clone();
  const originalMax = modelBoundingBox.max.clone();

  if (isYZSwapped) {
    // Apply Y-Z swap to bounding box: (x,y,z) -> (x,-z,y)
    modelBoundingBox = {
      min: new BABYLON.Vector3(
        originalMin.x,
        -originalMax.z,
        originalMin.y
      ),
      max: new BABYLON.Vector3(
        originalMax.x,
        -originalMin.z,
        originalMax.y
      ),
      center: null,
      size: null
    };
  } else {
    // Reverse Y-Z swap: (x,y,z) -> (x,z,-y)
    modelBoundingBox = {
      min: new BABYLON.Vector3(
        originalMin.x,
        originalMin.z,
        -originalMax.y
      ),
      max: new BABYLON.Vector3(
        originalMax.x,
        originalMax.z,
        -originalMin.y
      ),
      center: null,
      size: null
    };
  }

  // Recalculate center and size
  modelBoundingBox.center = BABYLON.Vector3.Center(modelBoundingBox.min, modelBoundingBox.max);
  modelBoundingBox.size = modelBoundingBox.max.subtract(modelBoundingBox.min);

  // Update camera target
  camera.target = modelBoundingBox.center;

  // Recreate clipping plane
  setupClippingPlane();

  console.log('Y-Z swap toggled:', isYZSwapped, 'New bbox:', modelBoundingBox);
}

// ============================================================================
// NODE AND EDGE CREATION
// ============================================================================
/**
 * Custom pick function that respects the clipping plane
 * Returns the closest hit that is on the visible side of the clipping plane
 */
function pickWithClipping(x, y) {
  // Create picking ray
  const ray = scene.createPickingRay(x, y, BABYLON.Matrix.Identity(), camera);

  // Get all ray intersections
  const results = scene.multiPickWithRay(ray);
  console.log('results', results);

  if (!results || results.length === 0) {
    console.log('No hits found');
    return { hit: false };
  }

  // If no clipping plane is active, return the first (closest) hit
  if (!clippingPlane) {
    return results[0];
  }

  // Filter results to only include hits on the visible side of the clipping plane
  // Then find the closest one to the camera
  let bestHit = null;
  let bestDistance = Infinity;

  for (const pickInfo of results) {
    if (pickInfo.hit && pickInfo.pickedPoint) {
      // Calculate signed distance from point to plane
      // Plane equation: ax + by + cz + d = 0
      // For our plane (0, 1, 0, d): y + d = 0
      const signedDistance = pickInfo.pickedPoint.y + clippingPlane.d;

 
      if (signedDistance < 0) {
        // This hit is visible, check if it's closer than previous hits
        if (pickInfo.distance < bestDistance) {
          bestDistance = pickInfo.distance;
          bestHit = pickInfo;
        }
      }
    }
  }

  if (bestHit) {
    console.log('Found hit at:', bestHit.pickedPoint, 'distance:', bestDistance);
    return bestHit;
  } else {
    console.log('No valid hit found (all hits were clipped)');
    return { hit: false };
  }
}

/**
 * Calculates the intersection point between a ray and a plane
 * @param {BABYLON.Ray} ray - The picking ray
 * @param {BABYLON.Plane} plane - The clipping plane
 * @returns {BABYLON.Vector3|null} - The intersection point or null if no intersection
 */
function intersectRayWithPlane(ray, plane) {
  // Ray equation: P = origin + t * direction
  // Plane equation: normal · P + d = 0

  const normal = new BABYLON.Vector3(plane.normal.x, plane.normal.y, plane.normal.z);
  const denom = BABYLON.Vector3.Dot(normal, ray.direction);

  // Check if ray is parallel to plane (denominator near zero)
  if (Math.abs(denom) < 1e-6) {
    console.log('Ray is parallel to plane, no intersection');
    return null;
  }

  // Calculate t: t = -(normal · origin + d) / (normal · direction)
  const t = -(BABYLON.Vector3.Dot(normal, ray.origin) + plane.d) / denom;

  // Check if intersection is behind the ray origin
  if (t < 0) {
    console.log('Intersection is behind camera, t =', t);
    return null;
  }

  // Calculate intersection point: P = origin + t * direction
  const intersection = ray.origin.add(ray.direction.scale(t));
  console.log('Ray-plane intersection at:', intersection, 't =', t);

  return intersection;
}

function handleDoubleClick(evt) {
  // Prevent the click handler from firing
  clickPreventedByDoubleClick = true;
  if (clickTimeout) {
    clearTimeout(clickTimeout);
    clickTimeout = null;
  }

  if (document.getElementById('nodePopup').style.display !== 'none') return;

  // Intersect with clipping plane directly instead of mesh
  if (clippingPlane) {
    const ray = scene.createPickingRay(scene.pointerX, scene.pointerY, BABYLON.Matrix.Identity(), camera);
    const intersection = intersectRayWithPlane(ray, clippingPlane);

    if (intersection) {
      const position = intersection.clone();
      // Small offset upward in world Y-axis (Babylon's up direction)
      position.y -= 0.2;

      console.log('Creating node at clipping plane:', position);
      createNode(position);
    }
  }

  // Reset the flag after a short delay
  setTimeout(() => {
    clickPreventedByDoubleClick = false;
  }, 100);
}

function handleClick(evt) {
  // Capture the pick result IMMEDIATELY, filtering to ONLY pick node spheres
  const pickResult = scene.pick(scene.pointerX, scene.pointerY, (mesh) => {
    // Only pick meshes that are nodes
    return mesh.metadata?.type === 'node';
  });

  console.log('Click captured, pickResult:', pickResult);
  if (pickResult.hit && pickResult.pickedMesh) {
    console.log('Picked node mesh:', pickResult.pickedMesh.name, 'metadata:', pickResult.pickedMesh.metadata);
  } else {
    console.log('No node mesh picked (clicked on canvas or model)');
  }

  // Delay click handling to check if it's part of a double-click
  if (clickTimeout) {
    clearTimeout(clickTimeout);
  }

  clickTimeout = setTimeout(() => {
    // If this was prevented by a double-click, ignore it
    if (clickPreventedByDoubleClick) {
      console.log('Click prevented by double-click');
      return;
    }

    console.log('handleClick processing after delay');

    if (pickResult.hit && pickResult.pickedMesh) {
      // We know it's a node because of the predicate filter
      const nodeId = pickResult.pickedMesh.metadata.nodeId;
      console.log('Clicked on node:', nodeId, 'selectedNode:', selectedNode?.id);

      // If a node is already selected and clicked node is different
      if (selectedNode && nodeId !== selectedNode.id) {
        // Create edge between them
        console.log('Creating edge between', selectedNode.id, 'and', nodeId);
        createEdge(selectedNode.id, nodeId);
        // Deselect both nodes
        deselectNode();
      } else {
        // Select the clicked node (without showing popup)
        console.log('Selecting node:', nodeId);
        selectNode(nodeId, false);
      }
    } else {
      // Clicked on canvas (not a node), deselect
      console.log('Clicked on canvas, deselecting');
      deselectNode();
    }
  }, 200); // 200ms delay to detect double-click
}

function handleRightClick(evt) {
  evt.preventDefault();
  // Right-click no longer used for edge creation
}

function createNode(position) {
  const nodeId = `node_${nodeIdCounter++}`;

  const node = {
    id: nodeId,
    name: '', // Leave blank for user to fill
    description: '', // Leave blank for user to fill
    position: {
      x: position.x,
      y: position.y,
      z: position.z
    }
  };

  graph.nodes.push(node);
  createNodeMesh(node);
  selectNode(nodeId, true); // Select AND show popup for editing
  updateGraphView();
  saveGraph();
}

function createNodeMesh(node) {
  const sphere = BABYLON.MeshBuilder.CreateSphere(
    node.id,
    { diameter: 0.3 },
    scene
  );

  sphere.position = new BABYLON.Vector3(
    node.position.x,
    node.position.y,
    node.position.z
  );

  const material = new BABYLON.StandardMaterial(node.id + '_mat', scene);
  material.diffuseColor = new BABYLON.Color3(1, 0, 0);
  material.emissiveColor = new BABYLON.Color3(0.3, 0, 0);
  material.disableDepthWrite = false; // Allow depth write for clipping
  sphere.material = material;

  // Put nodes in rendering group 2 for better organization
  sphere.renderingGroupId = 2;

  // Disable depth test to render on top, but keep clipping
  sphere.onBeforeRenderObservable.add(() => {
    engine.setDepthFunction(BABYLON.Engine.ALWAYS);
  });

  sphere.onAfterRenderObservable.add(() => {
    engine.setDepthFunction(BABYLON.Engine.LEQUAL);
  });

  // IMPORTANT: Set metadata to identify this as a node for picking
  sphere.metadata = {
    type: 'node',
    nodeId: node.id
  };

  // Ensure the sphere is pickable
  sphere.isPickable = true;

  console.log('Created node mesh:', node.id, 'isPickable:', sphere.isPickable, 'metadata:', sphere.metadata);
}

function createEdge(fromId, toId) {
  // Check if edge already exists
  const exists = graph.edges.some(e =>
    (e.from === fromId && e.to === toId) ||
    (e.from === toId && e.to === fromId)
  );

  if (exists) {
    console.log('Edge already exists');
    return;
  }

  const edgeId = `edge_${edgeIdCounter++}`;

  const edge = {
    id: edgeId,
    from: fromId,
    to: toId
  };

  graph.edges.push(edge);
  createEdgeMesh(edge);
  updateGraphView();
  saveGraph();
}

function createEdgeMesh(edge) {
  const fromNode = graph.nodes.find(n => n.id === edge.from);
  const toNode = graph.nodes.find(n => n.id === edge.to);

  if (!fromNode || !toNode) return;

  const points = [
    new BABYLON.Vector3(fromNode.position.x, fromNode.position.y, fromNode.position.z),
    new BABYLON.Vector3(toNode.position.x, toNode.position.y, toNode.position.z)
  ];

  const line = BABYLON.MeshBuilder.CreateLines(
    edge.id,
    { points: points },
    scene
  );

  line.color = new BABYLON.Color3(0, 0.5, 1);
  line.renderingGroupId = 1;

  line.metadata = {
    type: 'edge',
    edgeId: edge.id
  };
}

// ============================================================================
// NODE SELECTION AND EDITING
// ============================================================================
function selectNode(nodeId, showPopup = true) {
  deselectNode();

  selectedNode = graph.nodes.find(n => n.id === nodeId);

  if (selectedNode) {
    // Add green highlight outline
    const sphere = scene.getMeshByName(selectedNode.id);
    if (sphere && highlightLayer) {
      highlightLayer.addMesh(sphere, BABYLON.Color3.Green());
    }

    // Only show popup if explicitly requested
    if (showPopup) {
      showNodePopup(selectedNode);
    }
    updateGraphView();
  }
}

function deselectNode() {
  if (selectedNode) {
    // Remove green highlight outline
    const sphere = scene.getMeshByName(selectedNode.id);
    if (sphere && highlightLayer) {
      highlightLayer.removeMesh(sphere);
    }

    selectedNode = null;
    closePopup();
    updateGraphView();
  }
}

function showNodePopup(node) {
  document.getElementById('nodeName').value = node.name;
  document.getElementById('nodeDesc').value = node.description;
  document.getElementById('nodePopup').style.display = 'flex';
}

function closePopup() {
  document.getElementById('nodePopup').style.display = 'none';
}

function saveNodeEdit() {
  if (selectedNode) {
    selectedNode.name = document.getElementById('nodeName').value;
    selectedNode.description = document.getElementById('nodeDesc').value;

    updateGraphView();
    saveGraph();
    closePopup();
    deselectNode();
  }
}

function deleteNode() {
  if (!selectedNode) return;

  if (!confirm(`Delete node "${selectedNode.name}"?`)) return;

  const nodeId = selectedNode.id;

  // Remove node
  graph.nodes = graph.nodes.filter(n => n.id !== nodeId);

  // Remove connected edges
  const edgesToRemove = graph.edges.filter(e => e.from === nodeId || e.to === nodeId);
  edgesToRemove.forEach(edge => {
    const line = scene.getMeshByName(edge.id);
    if (line) line.dispose();
  });
  graph.edges = graph.edges.filter(e => e.from !== nodeId && e.to !== nodeId);

  // Remove mesh
  const sphere = scene.getMeshByName(nodeId);
  if (sphere) sphere.dispose();

  selectedNode = null;
  closePopup();
  updateGraphView();
  saveGraph();
}

function deleteEdge(edgeId) {
  if (!edgeId) return;

  // Find the edge
  const edge = graph.edges.find(e => e.id === edgeId);
  if (!edge) return;

  // Get node names for confirmation message
  const fromNode = graph.nodes.find(n => n.id === edge.from);
  const toNode = graph.nodes.find(n => n.id === edge.to);

  if (!fromNode || !toNode) return;

  const fromName = fromNode.name || fromNode.id;
  const toName = toNode.name || toNode.id;

  // Confirm deletion
  if (!confirm(`Delete edge from "${fromName}" to "${toName}"?`)) return;

  // Remove edge from graph
  graph.edges = graph.edges.filter(e => e.id !== edgeId);

  // Remove mesh
  const line = scene.getMeshByName(edgeId);
  if (line) line.dispose();

  // Update UI and save
  updateGraphView();
  saveGraph();
}

// ============================================================================
// GRAPH VIEW UI
// ============================================================================
function updateGraphView() {
  document.getElementById('nodeCount').textContent = graph.nodes.length;
  document.getElementById('edgeCount').textContent = graph.edges.length;

  const graphList = document.getElementById('graphList');
  graphList.innerHTML = '';

  // Nodes section
  const nodesHeader = document.createElement('h4');
  nodesHeader.textContent = 'Nodes';
  graphList.appendChild(nodesHeader);

  graph.nodes.forEach(node => {
    const item = document.createElement('div');
    item.className = 'graph-item';
    if (selectedNode?.id === node.id) {
      item.classList.add('selected');
    }

    // Display node name or show ID if name is empty
    const displayName = node.name || node.id;
    item.innerHTML = `
      <div class="graph-item-name">${displayName}</div>
      <div class="graph-item-id">${node.id}</div>
      ${node.description ? `<div class="graph-item-desc">${node.description}</div>` : ''}
    `;

    // Single-click: Select node only (no popup)
    item.addEventListener('click', () => {
      selectNode(node.id, false);
    });

    // Double-click: Select node and show popup
    item.addEventListener('dblclick', () => {
      selectNode(node.id, true);
    });

    graphList.appendChild(item);
  });

  // Edges section
  const edgesHeader = document.createElement('h4');
  edgesHeader.textContent = 'Edges';
  graphList.appendChild(edgesHeader);

  graph.edges.forEach(edge => {
    const fromNode = graph.nodes.find(n => n.id === edge.from);
    const toNode = graph.nodes.find(n => n.id === edge.to);

    if (fromNode && toNode) {
      const item = document.createElement('div');
      item.className = 'graph-item edge';
      const fromName = fromNode.name || fromNode.id;
      const toName = toNode.name || toNode.id;
      item.textContent = `${fromName} → ${toName}`;

      // Double-click to delete edge
      item.addEventListener('dblclick', () => {
        deleteEdge(edge.id);
      });

      graphList.appendChild(item);
    }
  });
}

// ============================================================================
// AUTO-SAVE TO JSON
// ============================================================================
function saveGraph() {
  if (!directoryHandle || !modelFileName) return;

  // Debounce saves to avoid excessive file writes
  if (saveTimeout) {
    clearTimeout(saveTimeout);
  }

  const status = document.getElementById('saveStatus');
  status.textContent = 'Auto-save: Pending...';
  status.className = 'auto-save-status saving';

  // Debounce for 500ms
  saveTimeout = setTimeout(() => {
    saveGraphImmediate();
  }, 500);
}

async function saveGraphImmediate() {
  if (!directoryHandle || !modelFileName) return;

  const status = document.getElementById('saveStatus');
  status.textContent = 'Auto-save: Saving...';
  status.className = 'auto-save-status saving';

  try {
    const jsonFileName = modelFileName.replace('.glb', '-graph.json');
    const fileHandle = await directoryHandle.getFileHandle(jsonFileName, { create: true });
    const writable = await fileHandle.createWritable();

    const jsonData = JSON.stringify(graph, null, 2);
    await writable.write(jsonData);
    await writable.close();

    status.textContent = 'Auto-save: Saved';
    status.className = 'auto-save-status saved';

    setTimeout(() => {
      status.textContent = 'Auto-save: Ready';
      status.className = 'auto-save-status';
    }, 2000);

  } catch (error) {
    console.error('Error saving graph:', error);
    status.textContent = 'Auto-save: Error';
    status.className = 'auto-save-status error';
  }
}

// ============================================================================
// TASK 2: NAVIGABILITY ANNOTATION
// ============================================================================

/**
 * Initialize Task 2 annotation workflow
 */
async function initializeTask2() {
  // Create annotation manager
  annotationManager = new AnnotationManager(directoryHandle, modelFileName, graph);

  // Load QA and traverse files
  await annotationManager.loadQAFile();
  await annotationManager.loadTraverseData();

  // Update UI based on availability
  updateTask2UI();
}

/**
 * Update Task 2 UI based on current state
 */
function updateTask2UI() {
  const statusEl = document.getElementById('task2Status');
  const startBtn = document.getElementById('startAnnotationBtn');

  if (!annotationManager) {
    statusEl.innerHTML = '<p>Load a folder with QA file to begin</p>';
    startBtn.style.display = 'none';
    return;
  }

  if (annotationManager.canStartAnnotation()) {
    const progress = annotationManager.getProgress();
    statusEl.innerHTML = `<p>Ready to annotate! ${progress.completed}/${progress.total} segments completed</p>`;
    startBtn.style.display = 'block';
  } else if (annotationManager.hasQAFile) {
    statusEl.innerHTML = '<p>QA file found, but graph is empty. Complete Task 1 first.</p>';
    startBtn.style.display = 'none';
  } else {
    statusEl.innerHTML = '<p>No QA file found. This model doesn\'t require navigability annotation.</p>';
    startBtn.style.display = 'none';
  }
}

/**
 * Start annotation workflow
 */
async function startAnnotation() {
  if (!annotationManager) {
    alert('Unable to start annotation workflow');
    return;
  }

  const started = await annotationManager.startAnnotation();

  if (!started) {
    alert('Unable to start annotation workflow');
    return;
  }

  // Show annotation panel
  document.getElementById('annotationPanel').style.display = 'block';
  document.getElementById('startAnnotationBtn').style.display = 'none';

  // Update current task display
  updateCurrentTask();
}

/**
 * Update the current task display
 */
function updateCurrentTask() {
  // Always clear highlights first to reset from previous task/question
  clearSegmentHighlight();

  const task = annotationManager.getCurrentTask();

  if (!task) {
    // No more tasks
    document.getElementById('annotationPanel').style.display = 'none';
    document.getElementById('task2Status').innerHTML = '<p>✓ All annotations complete!</p>';
    return;
  }

  // Update progress
  document.getElementById('questionNum').textContent = task.questionIndex + 1;
  document.getElementById('questionTotal').textContent = task.questionTotal;
  document.getElementById('annotationProgress').textContent = `${task.progress.completed}/${task.progress.total}`;

  // Update question
  document.getElementById('questionText').textContent = task.question.question;

  // Update agent
  document.getElementById('agentName').textContent = task.agentName;

  // Update segment
  const segmentDetails = annotationManager.getSegmentDetails(task.segment);

  // Regular segment annotation
  const fromName = segmentDetails.from.name || segmentDetails.from.id;
  const toName = segmentDetails.to.name || segmentDetails.to.id;

  document.getElementById('segmentFrom').textContent = fromName;
  document.getElementById('segmentTo').textContent = toName;
  document.getElementById('segmentNum').textContent = task.segmentIndex + 1;
  document.getElementById('segmentTotal').textContent = task.segmentTotal;

  // Build question text with detail information if applicable
  let questionHTML = task.question.question;

  // Add start detail information if this segment starts from the start node
  if (segmentDetails.fromIsStart && segmentDetails.startDetail) {
    questionHTML += `<br><em style="color: #4CAF50;">🚀 Starting from: "${segmentDetails.startDetail}" in ${fromName}</em>`;
  }

  // Add end detail information if this segment goes to the end node
  if (segmentDetails.toIsEnd && segmentDetails.endDetail) {
    questionHTML += `<br><em style="color: #ff6b35;">🎯 Reaching: "${segmentDetails.endDetail}" in ${toName}</em>`;
  }

  document.getElementById('questionText').innerHTML = questionHTML;

  // Display existing annotation if available
  const existingAnnotationDiv = document.getElementById('existingAnnotation');
  const existingAnnotationContent = document.getElementById('existingAnnotationContent');

  if (task.existingAnnotation) {
    // Show existing annotation
    const annotation = task.existingAnnotation;
    const traversableText = annotation.traversable ? '✓ Yes (Traversable)' : '✗ No (Not Traversable)';
    const noteText = annotation.note ? `<br>Note: ${annotation.note}` : '';
    const timestamp = annotation.timestamp ? `<br><small>Annotated: ${new Date(annotation.timestamp).toLocaleString()}</small>` : '';

    existingAnnotationContent.innerHTML = `${traversableText}${noteText}${timestamp}`;
    existingAnnotationDiv.style.display = 'block';

    // Prefill with existing annotation data
    document.getElementById('annotationNote').value = annotation.note || '';
  } else {
    // No existing annotation
    existingAnnotationDiv.style.display = 'none';

    // Prefill note if available from previous "no" annotations
    if (task.prefillNote) {
      document.getElementById('annotationNote').value = task.prefillNote;
      console.log('Prefilled note from previous annotation:', task.prefillNote);
    } else {
      document.getElementById('annotationNote').value = '';
    }
  }

  // Highlight the segment nodes in yellow
  highlightSegmentNodes(task.segment);

  // Update spawn button text
  document.getElementById('spawnAgentBtn').textContent = 'Spawn Agent at Start';

  // Update annotation status indicator
  const statusSpan = document.getElementById('annotationStatus');
  if (task.existingAnnotation) {
    statusSpan.textContent = '[Already Annotated ✓]';
    statusSpan.style.color = '#4CAF50';
  } else {
    statusSpan.textContent = '[Not Yet Annotated]';
    statusSpan.style.color = '#ff6b35';
  }

  // Show/hide Next button: hide if current task is not yet annotated
  const nextBtn = document.getElementById('goNextBtn');
  if (task.existingAnnotation) {
    // Already annotated - show Next button to continue forward
    nextBtn.style.display = 'inline-block';
  } else {
    // Not yet annotated - this is the latest unannotated task, hide Next button
    nextBtn.style.display = 'none';
  }

  console.log('Current task:', task, 'Annotated:', !!task.existingAnnotation);
}

/**
 * Submit annotation (Yes/No)
 */
async function submitAnnotation(traversable) {
  const note = document.getElementById('annotationNote').value.trim();

  // Require note when annotating "No"
  if (!traversable && !note) {
    alert('Please provide a reason when marking a path as not traversable.');
    document.getElementById('annotationNote').focus();
    return;
  }

  // Clear highlights before moving to next task
  clearSegmentHighlight();

  const hasNext = await annotationManager.submitAnnotation(traversable, note);

  if (hasNext) {
    updateCurrentTask();
  } else {
    // All done
    updateCurrentTask();
  }

  // Remove agent if active
  removeAgent();
}

/**
 * Go forward to next annotation (without submitting)
 */
async function goNextAnnotation() {
  // Clear highlights before moving to next task
  clearSegmentHighlight();

  const hasNext = await annotationManager.goNext();

  if (hasNext) {
    updateCurrentTask();
  } else {
    alert('Already at the last task');
  }

  // Remove agent if active
  removeAgent();
}

/**
 * Go back to previous annotation
 */
async function goBackAnnotation() {
  // Clear highlights before moving to previous task
  clearSegmentHighlight();

  const hasPrevious = await annotationManager.goBack();

  if (hasPrevious) {
    updateCurrentTask();
  } else {
    alert('Already at the first task');
  }

  // Remove agent if active
  removeAgent();
}

/**
 * Mark current question as hallucinated and skip all its annotations
 */
async function markQuestionHallucinated() {
  if (!annotationManager || !annotationManager.isActive) {
    console.error('Cannot mark question as hallucinated: annotation manager not active');
    return;
  }

  const task = annotationManager.getCurrentTask();
  if (!task) {
    console.error('Cannot mark question as hallucinated: no current task');
    return;
  }

  // Confirm with user
  if (!confirm(`Mark this question as hallucinated?\n\n"${task.question.question}"\n\nThis will skip all annotation tasks for this question and mark it as invalid.`)) {
    return;
  }

  // Clear highlights before moving
  clearSegmentHighlight();

  // Mark question as hallucinated and skip to next question
  const hasNext = await annotationManager.markQuestionAsHallucinated();

  if (hasNext) {
    updateCurrentTask();
  } else {
    // All done
    updateCurrentTask();
  }

  // Remove agent if active
  removeAgent();

  console.log('✓ Marked question as hallucinated and skipped all its tasks');
}

/**
 * Spawn agent at start node of current segment
 */
function spawnAgentAtStart() {
  if (!annotationManager || !annotationManager.isActive) {
    console.error('Cannot spawn agent: annotation manager not active');
    return;
  }

  if (!agentManager || !agentManager.physicsEnabled) {
    console.error('Cannot spawn agent: physics not initialized yet');
    alert('Please wait for physics to initialize before spawning agents');
    return;
  }

  const task = annotationManager.getCurrentTask();
  if (!task) {
    console.error('Cannot spawn agent: no current task');
    return;
  }

  const segmentDetails = annotationManager.getSegmentDetails(task.segment);

  // Regular segment: spawn at "from" node
  if (!segmentDetails || !segmentDetails.from) {
    console.error('Cannot spawn agent: invalid segment details');
    return;
  }

  const spawnNode = segmentDetails.from;
  const nodeName = spawnNode.name || spawnNode.id;
  console.log('Spawning at from node:', nodeName);

  const position = {
    x: spawnNode.position.x,
    y: spawnNode.position.y,
    z: spawnNode.position.z,
  };

  console.log('Spawning agent at position:', position);
  console.log('Agent type:', task.agentType);

  // Spawn agent
  const agent = agentManager.spawnAgent(task.agentType, position);

  if (!agent) {
    console.error('Failed to spawn agent');
    return;
  }

  // Update UI
  document.getElementById('spawnAgentBtn').style.display = 'none';
  document.getElementById('removeAgentBtn').style.display = 'inline-block';

  console.log('✓ Spawned', task.agentName, 'at', nodeName);
  console.log('Agent mesh visible:', agent.mesh.isVisible);
  console.log('Agent position:', agent.mesh.position);
  console.log('Use WASD to control: W=forward, S=backward, A=turn left, D=turn right');
  console.log('Use Arrow Up/Down to adjust Y position: Up=+0.1, Down=-0.1');
}

/**
 * Remove active agent
 */
function removeAgent() {
  agentManager.removeAgent();

  // Update UI
  document.getElementById('spawnAgentBtn').style.display = 'inline-block';
  document.getElementById('removeAgentBtn').style.display = 'none';
}

/**
 * Handle keyboard controls for agent movement and annotation navigation
 */
function handleKeyboardControls(event) {
  const agent = agentManager.getActiveAgent();
  const agentActive = agent && agent.isActive;

  // Check if user is typing in an input field
  const activeElement = document.activeElement;
  const isTyping = activeElement && (
    activeElement.tagName === 'INPUT' || 
    activeElement.tagName === 'TEXTAREA' || 
    activeElement.contentEditable === 'true'
  );

  // Handle annotation navigation with Q/E keys (only if agent is not active and not typing)
  if (!agentActive && !isTyping && annotationManager && annotationManager.isActive) {
    if (event.key.toLowerCase() === 'q') {
      event.preventDefault();
      goBackAnnotation();
      return;
    } else if (event.key.toLowerCase() === 'e') {
      event.preventDefault();
      // Only go next if current task is already annotated
      const currentTask = annotationManager.getCurrentTask();
      if (currentTask && currentTask.existingAnnotation) {
        goNextAnnotation();
      }
      return;
    }
  }

  // If agent is not active, stop here
  if (!agentActive) {
    return;
  }

  // Prevent default for control keys
  if (['w', 's', 'a', 'd', 'W', 'S', 'A', 'D', 'ArrowUp', 'ArrowDown', 'q', 'e', 'Q', 'E'].includes(event.key)) {
    event.preventDefault();
  }

  switch (event.key.toLowerCase()) {
    case 'w':
      console.log('W pressed - Moving forward');
      agent.moveForward(1); // Forward
      break;
    case 's':
      console.log('S pressed - Moving backward');
      agent.moveForward(-1); // Backward
      break;
    case 'a':
      console.log('A pressed - Turning left');
      agent.turn(-1); // Turn left
      break;
    case 'd':
      console.log('D pressed - Turning right');
      agent.turn(1); // Turn right
      break;
  }

  // Handle arrow keys for Y position adjustment (only when agent is active)
  if (event.key === 'ArrowUp') {
    console.log('Arrow Up pressed - Increasing Y position');
    agent.mesh.position.y += 0.1;
    // Update physics body position if it exists
    if (agent.body) {
      const currentPos = agent.body.getPosition();
      agent.body.setPosition(currentPos.x, currentPos.y + 0.1, currentPos.z);
    }
    console.log('New Y position:', agent.mesh.position.y);
  } else if (event.key === 'ArrowDown') {
    console.log('Arrow Down pressed - Decreasing Y position');
    agent.mesh.position.y -= 0.1;
    // Update physics body position if it exists
    if (agent.body) {
      const currentPos = agent.body.getPosition();
      agent.body.setPosition(currentPos.x, currentPos.y - 0.1, currentPos.z);
    }
    console.log('New Y position:', agent.mesh.position.y);
  }
}

/**
 * Highlight nodes for current annotation with route context
 * Green: start and end nodes of route
 * Orange: intermediate nodes on the path
 * Yellow: current segment nodes (overrides other colors)
 */
function highlightSegmentNodes(segment) {
  // Get current task to find all nodes in the route
  const task = annotationManager.getCurrentTask();
  if (!task || !task.question || !task.question.nodes_involved) {
    console.warn('Cannot get route nodes for highlighting');
    return;
  }

  // Process the question to get the full node sequence
  const result = processQuestion(task.question, graph);
  if (!result || !result.nodeSequence) {
    console.warn('Cannot process question for highlighting');
    return;
  }

  const nodeSequence = result.nodeSequence;

  // Get start and end node IDs
  const startNodeId = nodeSequence.length > 0 ? nodeSequence[0].id : null;
  const endNodeId = nodeSequence.length > 0 ? nodeSequence[nodeSequence.length - 1].id : null;

  // Collect all node IDs on the path
  const pathNodeIds = new Set();
  result.segments.forEach(seg => {
    pathNodeIds.add(seg.from);
    pathNodeIds.add(seg.to);
  });

  // First pass: Color start/end nodes green, intermediate nodes orange
  pathNodeIds.forEach(nodeId => {
    const sphere = scene.getMeshByName(nodeId);
    if (sphere && sphere.material) {
      if (nodeId === startNodeId || nodeId === endNodeId) {
        // Start or end node - bright green
        sphere.material.diffuseColor = new BABYLON.Color3(0.2, 1, 0.2); // Bright green
        sphere.material.emissiveColor = new BABYLON.Color3(0.1, 0.5, 0.1); // Green glow
      } else {
        // Intermediate node - deep orange (red-orange)
        sphere.material.diffuseColor = new BABYLON.Color3(1, 0.35, 0); // Deep orange
        sphere.material.emissiveColor = new BABYLON.Color3(0.5, 0.15, 0); // Orange glow
      }
    }
  });

  // Second pass: Override with yellow for current segment (highest priority)
  const fromSphere = scene.getMeshByName(segment.from);
  const toSphere = scene.getMeshByName(segment.to);

  if (fromSphere && fromSphere.material) {
    fromSphere.material.diffuseColor = new BABYLON.Color3(1, 1, 0); // Yellow
    fromSphere.material.emissiveColor = new BABYLON.Color3(0.5, 0.5, 0); // Yellow glow
  }

  if (toSphere && toSphere.material) {
    toSphere.material.diffuseColor = new BABYLON.Color3(1, 1, 0); // Yellow
    toSphere.material.emissiveColor = new BABYLON.Color3(0.5, 0.5, 0); // Yellow glow
  }

  console.log('Highlighted route - Start:', startNodeId, 'End:', endNodeId, 'Current segment:', segment.from, '→', segment.to);
}

/**
 * Clear segment highlight and restore red color
 */
function clearSegmentHighlight() {
  // Reset all node spheres to red
  graph.nodes.forEach(node => {
    const sphere = scene.getMeshByName(node.id);
    if (sphere && sphere.material) {
      sphere.material.diffuseColor = new BABYLON.Color3(1, 0, 0); // Red
      sphere.material.emissiveColor = new BABYLON.Color3(0.3, 0, 0); // Red glow
    }
  });
}
