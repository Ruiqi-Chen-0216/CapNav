// ============================================================================
// AGENT SIMULATION MODULE
// ============================================================================
// Handles agent creation, physics, and movement using Ammo.js + Babylon.js

import * as BABYLON from '@babylonjs/core';
import { AGENT_PARAMS, PHYSICS_PARAMS } from './agentParams.js';

// ============================================================================
// AGENT CLASS
// ============================================================================
export class Agent {
  constructor(scene, agentType, position = { x: 0, y: 0, z: 0 }) {
    this.scene = scene;
    this.agentType = agentType;
    this.config = AGENT_PARAMS[agentType];

    // Agent state
    this.position = position;
    this.rotation = 0; // degrees
    this.isActive = false;

    // Create visual mesh first, then physics
    this.createVisualMesh();
    this.createPhysicsImpostor();

    // Create turning collider for wheelchair
    if (agentType === 'WHEELCHAIR') {
      this.createTurningCollider();
    }
  }

  createVisualMesh() {
    const config = this.config;
    const color = new BABYLON.Color3(...config.color);

    if (config.shape === 'cylinder') {
      this.mesh = BABYLON.MeshBuilder.CreateCylinder(
        `agent_${this.agentType}`,
        {
          height: config.height,
          diameter: config.radius * 2,
        },
        this.scene
      );
    } else if (config.shape === 'box') {
      this.mesh = BABYLON.MeshBuilder.CreateBox(
        `agent_${this.agentType}`,
        {
          width: config.width,
          height: config.height,
          depth: config.depth,
        },
        this.scene
      );
    }

    // Create material with bright, visible appearance
    const material = new BABYLON.StandardMaterial(`agent_mat_${this.agentType}`, this.scene);
    material.diffuseColor = color;
    material.emissiveColor = color.scale(0.5); // Brighter glow
    material.alpha = 0.9; // Less transparent
    material.backFaceCulling = false; // Visible from all angles
    this.mesh.material = material;

    // Create facing direction indicator (arrow/cone pointing forward)
    this.createDirectionIndicator();

    // Set rendering group to same as house mesh for proper depth testing
    // This allows intersections to be visible
    this.mesh.renderingGroupId = 0;

    // Initially hidden
    this.mesh.isVisible = false;
  }

  createPhysicsImpostor() {
    // Physics disabled for now - using free movement
    // Agents will move without collision detection
    console.log('Physics impostor creation skipped - using free movement mode');
  }

  createDirectionIndicator() {
    // Create a cone pointing in the forward direction (Z-axis in Babylon)
    this.directionArrow = BABYLON.MeshBuilder.CreateCylinder(
      `agent_arrow_${this.agentType}`,
      {
        height: 0.3,
        diameterTop: 0,
        diameterBottom: 0.15,
      },
      this.scene
    );

    // Position it in front of the agent
    this.directionArrow.position.z = this.config.radius || this.config.depth / 2 || 0.5;
    this.directionArrow.position.y = this.config.height / 2;
    this.directionArrow.rotation.x = Math.PI / 2; // Point forward

    // Bright yellow color for visibility
    const arrowMaterial = new BABYLON.StandardMaterial(`agent_arrow_mat_${this.agentType}`, this.scene);
    arrowMaterial.diffuseColor = new BABYLON.Color3(1, 1, 0); // Yellow
    arrowMaterial.emissiveColor = new BABYLON.Color3(0.8, 0.8, 0); // Bright yellow glow
    this.directionArrow.material = arrowMaterial;

    // Parent to main mesh so it follows
    this.directionArrow.parent = this.mesh;
    this.directionArrow.renderingGroupId = 0;
    this.directionArrow.isVisible = false;
  }

  createTurningCollider() {
    // Create larger collider for wheelchair turning
    const turningRadius = this.config.turningRadius;

    this.turningMesh = BABYLON.MeshBuilder.CreateCylinder(
      `agent_turning_${this.agentType}`,
      {
        height: 0.1,
        diameter: turningRadius * 2,
      },
      this.scene
    );

    const material = new BABYLON.StandardMaterial(`agent_turning_mat`, this.scene);
    material.diffuseColor = new BABYLON.Color3(...this.config.color);
    material.alpha = 0.2;
    material.wireframe = true;
    this.turningMesh.material = material;
    this.turningMesh.renderingGroupId = 0;
    this.turningMesh.isVisible = false;
  }

  // Activate agent and show in scene
  activate(position) {
    this.isActive = true;
    this.position = position;

    // Set mesh position
    this.mesh.position.set(
      position.x,
      position.y + this.config.height / 2,
      position.z
    );

    // Reset rotation
    this.rotation = 0;
    this.mesh.rotation.y = 0;

    // Show mesh and direction indicator
    this.mesh.isVisible = true;
    if (this.directionArrow) {
      this.directionArrow.isVisible = true;
    }
    if (this.turningMesh) {
      this.turningMesh.isVisible = true;
      this.turningMesh.position.set(
        position.x,
        0.05,
        position.z
      );
    }

    console.log(`Agent activated at position:`, position);
    console.log(`Agent type: ${this.agentType}, Config:`, this.config);
    console.log(`Mesh position after activation:`, {
      x: this.mesh.position.x,
      y: this.mesh.position.y,
      z: this.mesh.position.z
    });
    console.log('Free movement mode: WASD controls enabled');
  }

  // Deactivate agent and hide from scene
  deactivate() {
    this.isActive = false;
    this.mesh.isVisible = false;
    if (this.directionArrow) {
      this.directionArrow.isVisible = false;
    }
    if (this.turningMesh) {
      this.turningMesh.isVisible = false;
    }
  }

  // Move forward (positive) or backward (negative)
  moveForward(direction = 1) {
    if (!this.isActive) {
      console.warn('Agent not active, cannot move');
      return;
    }

    const distance = this.config.moveIncrement * direction;
    const radians = (this.rotation * Math.PI) / 180;

    // Calculate movement direction
    const dx = Math.sin(radians) * distance;
    const dz = Math.cos(radians) * distance;

    const oldPos = { x: this.mesh.position.x, z: this.mesh.position.z };

    // Move mesh
    this.mesh.position.x += dx;
    this.mesh.position.z += dz;

    // Update turning collider if exists
    if (this.turningMesh) {
      this.turningMesh.position.x = this.mesh.position.x;
      this.turningMesh.position.z = this.mesh.position.z;
    }

    console.log(`Moved ${direction > 0 ? 'forward' : 'backward'} by ${distance.toFixed(2)}m`);
    console.log(`Position: (${oldPos.x.toFixed(2)}, ${oldPos.z.toFixed(2)}) → (${this.mesh.position.x.toFixed(2)}, ${this.mesh.position.z.toFixed(2)})`);
  }

  // Turn left (negative) or right (positive)
  turn(direction = 1) {
    if (!this.isActive) {
      console.warn('Agent not active, cannot turn');
      return;
    }

    const oldRotation = this.rotation;
    this.rotation += this.config.turnIncrement * direction;
    this.rotation = this.rotation % 360; // Keep in 0-360 range

    // Update mesh rotation
    const radians = (this.rotation * Math.PI) / 180;
    this.mesh.rotation.y = radians;

    // Update turning collider if exists
    if (this.turningMesh) {
      this.turningMesh.rotation.y = radians;
    }

    console.log(`Turned ${direction < 0 ? 'left' : 'right'} by ${this.config.turnIncrement}°`);
    console.log(`Rotation: ${oldRotation.toFixed(1)}° → ${this.rotation.toFixed(1)}°`);
  }

  // Update is no longer needed since physics handles everything
  update() {
    if (!this.isActive) return;

    // Update turning collider position if exists
    if (this.turningMesh) {
      this.turningMesh.position.set(
        this.mesh.position.x,
        0.05,
        this.mesh.position.z
      );
      this.turningMesh.rotation.y = this.mesh.rotation.y;
    }
  }

  // Cleanup
  dispose() {
    if (this.mesh) {
      this.mesh.dispose();
    }
    if (this.directionArrow) this.directionArrow.dispose();
    if (this.turningMesh) this.turningMesh.dispose();
  }
}

// ============================================================================
// AGENT MANAGER
// ============================================================================
export class AgentManager {
  constructor(scene) {
    this.scene = scene;
    this.agents = {};
    this.activeAgent = null;
    this.physicsEnabled = false;

    // Physics will be initialized later when enablePhysics() is called
  }

  // Enable physics with Ammo.js (disabled for now - using free movement)
  async enablePhysics() {
    if (this.physicsEnabled) {
      console.warn('Physics already enabled');
      return;
    }

    console.log('Physics disabled - using free movement mode for agents');
    console.log('Agents will move freely without collision detection');

    // Skip Ammo.js loading, just create agents
    this.physicsEnabled = true;
    this.createAllAgents();
  }

  createAllAgents() {
    const agentTypes = Object.keys(AGENT_PARAMS);
    agentTypes.forEach(type => {
      this.agents[type] = new Agent(this.scene, type);
    });
    console.log('✓ Created agents for all types');
  }

  // Spawn an agent at a position
  spawnAgent(agentType, position) {
    // Check if physics is enabled
    if (!this.physicsEnabled) {
      console.error('Cannot spawn agent: physics not enabled');
      console.error('Call enablePhysics() first');
      return null;
    }

    // Deactivate current agent if any
    if (this.activeAgent) {
      this.activeAgent.deactivate();
    }

    // Activate new agent
    const agent = this.agents[agentType];
    if (!agent) {
      console.error('Agent type not found:', agentType);
      console.error('Available agents:', Object.keys(this.agents));
      console.error('Requested agent type:', agentType);
      return null;
    }

    agent.activate(position);
    this.activeAgent = agent;

    return agent;
  }

  // Remove active agent
  removeAgent() {
    if (this.activeAgent) {
      this.activeAgent.deactivate();
      this.activeAgent = null;
    }
  }

  // Get current active agent
  getActiveAgent() {
    return this.activeAgent;
  }

  // Update (mainly for turning collider sync)
  update(deltaTime) {
    if (this.activeAgent) {
      this.activeAgent.update();
    }
  }

  // Cleanup
  dispose() {
    Object.values(this.agents).forEach(agent => agent.dispose());
    this.agents = {};
    this.activeAgent = null;
  }
}
