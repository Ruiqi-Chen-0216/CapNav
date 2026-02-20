// ============================================================================
// AGENT CONFIGURATION PARAMETERS
// ============================================================================
// All agent dimensions, movement parameters, and physics constants
// Modify these values to adjust agent behavior

export const AGENT_PARAMS = {
  // Able-bodied human
  HUMAN: {
    name: 'Able-bodied human',
    shape: 'cylinder',
    height: 1.7, // meters
    radius: 0.3, // diameter 0.6m / 2
    verticalCrossAbility: 0.3, // can step over 30cm obstacles
    moveIncrement: 0.1, // meters per input
    turnIncrement: 5, // degrees per input
    mass: 70, // kg
    color: [0.2, 0.5, 1.0], // RGB for visualization
  },

  // Wheelchair user
  WHEELCHAIR: {
    name: 'Wheelchair user',
    shape: 'box',
    height: 1.2, // meters
    width: 0.7, // meters
    depth: 1.0, // meters
    verticalCrossAbility: 0.013, // can only go over 1.3cm
    turningRadius: 1, // meters
    moveIncrement: 0.1, // meters per input
    turnIncrement: 5, // degrees per input
    mass: 100, // kg (including wheelchair)
    color: [1.0, 0.5, 0.2], // RGB for visualization
  },

  // Humanoid robot
  ROBOT: {
    name: 'Humanoid robot',
    shape: 'cylinder',
    height: 1.5, // meters
    radius: 0.45, // diameter 0.9m
    verticalCrossAbility: 0.9, // can step over 20cm obstacles
    moveIncrement: 0.1, // meters per input
    turnIncrement: 5, // degrees per input
    mass: 50, // kg
    color: [0.5, 1.0, 0.2], // RGB for visualization
  },

  // Sweeping robot
  SWEEPER: {
    name: 'Sweeping robot',
    shape: 'cylinder',
    height: 0.15, // meters
    radius: 0.2, // diameter 0.4m
    verticalCrossAbility: 0.02, // can only go over 2cm
    moveIncrement: 0.1, // meters per input
    turnIncrement: 5, // degrees per input
    mass: 5, // kg
    color: [1.0, 0.2, 0.5], // RGB for visualization
  },
  // Sweeping robot
  QUADRUPEDAL: {
    name: 'Quadrupedal robot',
    shape: 'box',
    height: 0.7, // meters
    width: 0.5, // meters
    depth: 1.1, // meters
    verticalCrossAbility: 0.30, // can step over 30cm obstacles
    moveIncrement: 0.1, // meters per input
    turnIncrement: 5, // degrees per input
    mass: 5, // kg
    color: [1.0, 0.2, 0.5], // RGB for visualization
  }
};

// Physics simulation parameters
export const PHYSICS_PARAMS = {
  gravity: 0, // m/s^2
  timeStep: 1 / 60, // 60 FPS
  maxSubSteps: 3,
  friction: 0.3,
  restitution: 0.1, // bounciness
};

// Get list of all agent types
export function getAgentTypes() {
  return Object.keys(AGENT_PARAMS);
}

// Get agent config by type
export function getAgentConfig(agentType) {
  return AGENT_PARAMS[agentType];
}
