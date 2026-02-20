// ============================================================================
// NAVIGABILITY ANNOTATION MODULE
// ============================================================================
// Handles QA loading, annotation workflow, and traverse.json management

import { getAgentTypes, getAgentConfig } from './agentParams.js';
import { processQuestion } from './pathfinding.js';

/**
 * Annotation Manager
 * Manages the navigability annotation workflow for Task 2
 */
export class AnnotationManager {
  constructor(directoryHandle, modelFileName, graph) {
    this.directoryHandle = directoryHandle;
    this.modelFileName = modelFileName;
    this.graph = graph;

    // QA data
    this.questions = [];
    this.currentQuestionIndex = 0;
    this.currentAgentIndex = 0;
    this.currentSegmentIndex = 0;

    // Current question state
    this.currentQuestion = null;
    this.currentAgentType = null;
    this.currentSegment = null;
    this.currentSegments = [];

    // Traverse data (annotations)
    this.traverseData = {};
    // Format: {
    //   agentType: {
    //     "nodeA|nodeB": { traversable: true/false, note: "..." },
    //     "endTask:nodeId": { canPerformTask: true/false, note: "..." }
    //   },
    //   _hallucinatedQuestions: ["question text 1", "question text 2", ...]
    // }

    // State
    this.isActive = false;
    this.hasQAFile = false;
  }

  /**
   * Load QA file if it exists
   */
  async loadQAFile() {
    if (!this.directoryHandle || !this.modelFileName) return false;

    try {
      const qaFileName = this.modelFileName.replace('.glb', '_QA.json');
      const qaFileHandle = await this.directoryHandle.getFileHandle(qaFileName);
      const qaFile = await qaFileHandle.getFile();
      const qaText = await qaFile.text();
      const qaData = JSON.parse(qaText);

      this.questions = qaData.questions || qaData || [];
      this.hasQAFile = this.questions.length > 0;

      console.log(`Loaded ${this.questions.length} QA questions`);
      return true;
    } catch (error) {
      if (error.name === 'NotFoundError') {
        console.log('No QA file found');
        this.hasQAFile = false;
      } else {
        console.error('Error loading QA file:', error);
      }
      return false;
    }
  }

  /**
   * Load existing traverse annotations if they exist
   */
  async loadTraverseData() {
    if (!this.directoryHandle || !this.modelFileName) return;

    try {
      const traverseFileName = this.modelFileName.replace('.glb', '-traverse.json');
      const traverseFileHandle = await this.directoryHandle.getFileHandle(traverseFileName);
      const traverseFile = await traverseFileHandle.getFile();
      const traverseText = await traverseFile.text();
      this.traverseData = JSON.parse(traverseText);

      console.log('Loaded existing traverse data');

      // Log hallucinated questions if any
      if (this.traverseData._hallucinatedQuestions && this.traverseData._hallucinatedQuestions.length > 0) {
        console.log(`Found ${this.traverseData._hallucinatedQuestions.length} hallucinated questions:`, this.traverseData._hallucinatedQuestions);
      }
    } catch (error) {
      if (error.name === 'NotFoundError') {
        console.log('No existing traverse file found - starting fresh');
        this.traverseData = {};
      } else {
        console.error('Error loading traverse data:', error);
        this.traverseData = {};
      }
    }
  }

  /**
   * Save traverse annotations to file
   */
  async saveTraverseData() {
    if (!this.directoryHandle || !this.modelFileName) return;

    try {
      const traverseFileName = this.modelFileName.replace('.glb', '-traverse.json');
      const fileHandle = await this.directoryHandle.getFileHandle(traverseFileName, { create: true });
      const writable = await fileHandle.createWritable();

      const jsonData = JSON.stringify(this.traverseData, null, 2);
      await writable.write(jsonData);
      await writable.close();

      console.log('Saved traverse data');
    } catch (error) {
      console.error('Error saving traverse data:', error);
    }
  }

  /**
   * Check if annotation workflow can start
   */
  canStartAnnotation() {
    const canStart = this.hasQAFile && this.graph.nodes.length > 0 && this.graph.edges.length > 0;
    console.log(`canStartAnnotation: hasQAFile=${this.hasQAFile}, nodes=${this.graph.nodes.length}, edges=${this.graph.edges.length}, result=${canStart}`);
    return canStart;
  }

  /**
   * Start the annotation workflow
   */
  async startAnnotation() {
    if (!this.canStartAnnotation()) {
      console.log('Cannot start annotation - requirements not met');
      return false;
    }

    console.log(`Starting annotation with ${this.questions.length} questions`);

    this.isActive = true;
    this.currentQuestionIndex = 0;
    this.currentAgentIndex = 0;
    this.currentSegmentIndex = 0;

    // Find first unannotated task
    const found = await this.findNextTask();

    console.log(`First task found: ${found}`);

    return found;
  }

  /**
   * Find the next unannotated task (question, agent, segment)
   */
  async findNextTask() {
    const agentTypes = getAgentTypes();
    let needsSave = false;

    console.log(`[findNextTask] Searching from question ${this.currentQuestionIndex}/${this.questions.length}`);

    // Initialize hallucinated questions array if it doesn't exist
    if (!this.traverseData._hallucinatedQuestions) {
      this.traverseData._hallucinatedQuestions = [];
    }

    console.log(`[findNextTask] Hallucinated questions: ${this.traverseData._hallucinatedQuestions.length}`);

    // Iterate through questions
    for (let q = this.currentQuestionIndex; q < this.questions.length; q++) {
      const question = this.questions[q];

      // Skip hallucinated questions
      if (this.isQuestionHallucinated(question.question)) {
        console.log(`[findNextTask] Skipping hallucinated question ${q}: "${question.question}"`);
        continue;
      }

      console.log(`[findNextTask] Processing question ${q}: "${question.question}"`);

      const result = processQuestion(question, this.graph);

      if (result.error) {
        console.error(`[findNextTask] Question ${q} error: ${result.error}`);
        continue;
      }

      console.log(`[findNextTask] Question ${q} has ${result.segments.length} segments`);


      // Iterate through agents
      for (let a = (q === this.currentQuestionIndex ? this.currentAgentIndex : 0); a < agentTypes.length; a++) {
        const agentType = agentTypes[a];

        // Auto-skip HUMAN agent - automatically mark all as traversable
        if (agentType === 'HUMAN') {
          // Count how many segments we auto-approve
          let autoApproved = 0;

          // Annotate all segments for HUMAN as traversable
          for (const segment of result.segments) {
            const segmentKey = `${segment.from}|${segment.to}`;
            if (!this.isSegmentAnnotated(agentType, segmentKey)) {
              this.annotateSegment(agentType, segmentKey, true, 'Auto-approved: able-bodied human');
              autoApproved++;
              needsSave = true;
            }
          }

          if (autoApproved > 0) {
            console.log(`✓ Auto-approved ${autoApproved} segments for HUMAN (able-bodied) agent`);
          }

          // Skip to next agent
          continue;
        }

        // Iterate through segments
        for (let s = (q === this.currentQuestionIndex && a === this.currentAgentIndex ? this.currentSegmentIndex : 0); s < result.segments.length; s++) {
          const segment = result.segments[s];
          const segmentKey = `${segment.from}|${segment.to}`;

          // Check if this segment is already annotated for this agent
          if (!this.isSegmentAnnotated(agentType, segmentKey)) {
            // Found unannotated task
            console.log(`[findNextTask] ✓ Found task: Q${q} / Agent:${agentType} / Segment:${s}/${result.segments.length}`);

            this.currentQuestionIndex = q;
            this.currentAgentIndex = a;
            this.currentSegmentIndex = s;
            this.currentQuestion = question;
            this.currentAgentType = agentType;
            this.currentSegment = segment;
            this.currentSegments = result.segments;

            // Save auto-approved annotations before returning
            if (needsSave) {
              await this.saveTraverseData();
            }

            return true;
          }
        }
      }
    }

    // Save any remaining auto-approved annotations
    if (needsSave) {
      await this.saveTraverseData();
    }

    // No more tasks
    console.log('[findNextTask] ✗ No more tasks found');
    this.isActive = false;
    return false;
  }

  /**
   * Check if a segment is already annotated for an agent
   */
  isSegmentAnnotated(agentType, segmentKey) {
    return this.traverseData[agentType] &&
           this.traverseData[agentType][segmentKey] !== undefined;
  }

  /**
   * Get prefill note from previous "no" annotations for the same segment and agent
   * Returns the note from the most recent "no" annotation, or null if none found
   */
  getPrefillNote(agentType, segmentKey) {
    if (!this.traverseData[agentType]) {
      return null;
    }

    // Collect all "no" annotations for this segment
    const noAnnotations = [];

    for (const key in this.traverseData[agentType]) {
      // Check if it's the same segment (could be same direction or reversed)
      const [from1, to1] = segmentKey.split('|');
      const [from2, to2] = key.split('|');

      // Same segment if (from1→to1) or (to1→from1)
      const isSameSegment = (from1 === from2 && to1 === to2) ||
                            (from1 === to2 && to1 === from2);

      if (isSameSegment) {
        const annotation = this.traverseData[agentType][key];
        if (annotation.traversable === false && annotation.note) {
          noAnnotations.push({
            key: key,
            note: annotation.note,
            timestamp: annotation.timestamp || new Date(0).toISOString(),
          });
        }
      }
    }

    // If no "no" annotations found, return null
    if (noAnnotations.length === 0) {
      return null;
    }

    // Sort by timestamp (most recent first)
    noAnnotations.sort((a, b) => {
      return new Date(b.timestamp) - new Date(a.timestamp);
    });

    // Return the most recent note
    return noAnnotations[0].note;
  }

  /**
   * Get current annotation progress
   */
  getProgress() {
    const agentTypes = getAgentTypes();
    let totalTasks = 0;
    let completedTasks = 0;

    for (const question of this.questions) {
      // Skip hallucinated questions
      if (this.isQuestionHallucinated(question.question)) {
        continue;
      }

      const result = processQuestion(question, this.graph);
      if (result.error) continue;

      for (const agentType of agentTypes) {
        // Count segment annotations
        for (const segment of result.segments) {
          const segmentKey = `${segment.from}|${segment.to}`;
          totalTasks++;
          if (this.isSegmentAnnotated(agentType, segmentKey)) {
            completedTasks++;
          }
        }
      }
    }

    return { completed: completedTasks, total: totalTasks };
  }

  /**
   * Helper: Annotate a specific segment for an agent (no workflow advancement)
   */
  annotateSegment(agentType, segmentKey, traversable, note = '') {
    // Initialize agent entry if needed
    if (!this.traverseData[agentType]) {
      this.traverseData[agentType] = {};
    }

    // Save annotation
    this.traverseData[agentType][segmentKey] = {
      traversable: traversable,
      note: note,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Submit annotation for current task
   */
  async submitAnnotation(traversable, note = '') {
    if (!this.isActive || !this.currentSegment) return false;

    // Regular segment annotation
    const segmentKey = `${this.currentSegment.from}|${this.currentSegment.to}`;

    // Annotate the segment
    this.annotateSegment(this.currentAgentType, segmentKey, traversable, note);

    // Add question reference
    this.traverseData[this.currentAgentType][segmentKey].question = this.currentQuestion.question;

    // Save to file
    await this.saveTraverseData();

    // Move to next task in chronological order (not skipping to unannotated)
    const hasNext = await this.goNext();

    return hasNext;
  }

  /**
   * Check if a question is marked as hallucinated
   */
  isQuestionHallucinated(questionText) {
    if (!this.traverseData._hallucinatedQuestions) {
      this.traverseData._hallucinatedQuestions = [];
    }
    return this.traverseData._hallucinatedQuestions.includes(questionText);
  }

  /**
   * Mark current question as hallucinated and skip to next question
   */
  async markQuestionAsHallucinated() {
    if (!this.isActive || !this.currentQuestion) return false;

    // Initialize hallucinated questions array if it doesn't exist
    if (!this.traverseData._hallucinatedQuestions) {
      this.traverseData._hallucinatedQuestions = [];
    }

    // Add question to hallucinated list (avoid duplicates)
    const questionText = this.currentQuestion.question;
    if (!this.isQuestionHallucinated(questionText)) {
      this.traverseData._hallucinatedQuestions.push(questionText);
      console.log('Marked question as hallucinated:', questionText);
    }

    // Save to file
    await this.saveTraverseData();

    // Move to next question
    this.currentQuestionIndex++;
    this.currentAgentIndex = 0;
    this.currentSegmentIndex = 0;

    // Find next non-hallucinated task
    const hasNext = await this.findNextTask();

    return hasNext;
  }

  /**
   * Skip current task
   */
  async skipTask() {
    this.currentSegmentIndex++;
    return await this.findNextTask();
  }

  /**
   * Go back to previous task
   */
  async goBack() {
    if (!this.isActive) return false;

    const agentTypes = getAgentTypes();

    // Try to go back within current question and agent
    if (this.currentSegmentIndex > 0) {
      this.currentSegmentIndex--;
      await this.loadCurrentTask();
      return true;
    }

    // Try to go back to previous agent
    if (this.currentAgentIndex > 0) {
      this.currentAgentIndex--;
      const question = this.questions[this.currentQuestionIndex];
      const result = processQuestion(question, this.graph);

      // Go to last segment of previous agent
      this.currentSegmentIndex = result.segments.length - 1;
      await this.loadCurrentTask();
      return true;
    }

    // Try to go back to previous question
    if (this.currentQuestionIndex > 0) {
      this.currentQuestionIndex--;
      const question = this.questions[this.currentQuestionIndex];
      const result = processQuestion(question, this.graph);

      // Go to last agent and last segment
      this.currentAgentIndex = agentTypes.length - 1;
      this.currentSegmentIndex = result.segments.length - 1;
      await this.loadCurrentTask();
      return true;
    }

    // Already at the beginning
    console.log('Already at the first task');
    return false;
  }

  /**
   * Go forward to next task (without submitting)
   */
  async goNext() {
    if (!this.isActive) return false;

    const agentTypes = getAgentTypes();
    const currentQuestion = this.questions[this.currentQuestionIndex];
    const currentResult = processQuestion(currentQuestion, this.graph);

    // Try to go forward within current question and agent
    if (this.currentSegmentIndex < currentResult.segments.length - 1) {
      this.currentSegmentIndex++;
      await this.loadCurrentTask();
      return true;
    }

    // Try to go forward to next agent
    if (this.currentAgentIndex < agentTypes.length - 1) {
      this.currentAgentIndex++;
      this.currentSegmentIndex = 0;
      await this.loadCurrentTask();
      return true;
    }

    // Try to go forward to next question
    if (this.currentQuestionIndex < this.questions.length - 1) {
      this.currentQuestionIndex++;
      this.currentAgentIndex = 0;
      this.currentSegmentIndex = 0;
      await this.loadCurrentTask();
      return true;
    }

    // Already at the last task
    console.log('Already at the last task');
    return false;
  }

  /**
   * Check if we're on the latest task (no more tasks after this)
   */
  isOnLatestTask() {
    if (!this.isActive) return true;

    const agentTypes = getAgentTypes();
    const currentQuestion = this.questions[this.currentQuestionIndex];
    const currentResult = processQuestion(currentQuestion, this.graph);

    // Check if this is the last segment of the last agent of the last question
    const isLastQuestion = this.currentQuestionIndex === this.questions.length - 1;
    const isLastAgent = this.currentAgentIndex === agentTypes.length - 1;
    const isLastSegment = this.currentSegmentIndex === currentResult.segments.length - 1;

    return isLastQuestion && isLastAgent && isLastSegment;
  }

  /**
   * Load current task based on current indices
   */
  async loadCurrentTask() {
    const agentTypes = getAgentTypes();
    const question = this.questions[this.currentQuestionIndex];
    const result = processQuestion(question, this.graph);

    if (result.error) {
      console.error(`Question ${this.currentQuestionIndex}: ${result.error}`);
      return false;
    }

    const agentType = agentTypes[this.currentAgentIndex];
    const segment = result.segments[this.currentSegmentIndex];

    this.currentQuestion = question;
    this.currentAgentType = agentType;
    this.currentSegment = segment;
    this.currentSegments = result.segments;

    return true;
  }

  /**
   * Get existing annotation for current task (if any)
   */
  getExistingAnnotation() {
    if (!this.currentSegment) return null;

    const segmentKey = `${this.currentSegment.from}|${this.currentSegment.to}`;

    if (this.traverseData[this.currentAgentType] &&
        this.traverseData[this.currentAgentType][segmentKey]) {
      return this.traverseData[this.currentAgentType][segmentKey];
    }

    return null;
  }

  /**
   * Get current task info
   */
  getCurrentTask() {
    if (!this.isActive) return null;

    let prefillNote = null;

    // Get prefill for regular segments
    const segmentKey = `${this.currentSegment.from}|${this.currentSegment.to}`;
    prefillNote = this.getPrefillNote(this.currentAgentType, segmentKey);

    // Get existing annotation for this task
    const existingAnnotation = this.getExistingAnnotation();

    return {
      question: this.currentQuestion,
      questionIndex: this.currentQuestionIndex,
      questionTotal: this.questions.length,
      agentType: this.currentAgentType,
      agentName: getAgentConfig(this.currentAgentType).name,
      segment: this.currentSegment,
      segmentIndex: this.currentSegmentIndex,
      segmentTotal: this.currentSegments.length,
      progress: this.getProgress(),
      prefillNote: prefillNote, // Prefilled note from previous "no" annotations
      existingAnnotation: existingAnnotation, // Existing annotation for review
    };
  }

  /**
   * Get segment details (nodes, positions)
   * Includes start_detail/end_detail if the segment involves the start/end nodes
   */
  getSegmentDetails(segment, question = null) {
    // Use currentQuestion if question not provided
    const q = question || this.currentQuestion;

    // Get nodes
    const fromNode = this.graph.nodes.find(n => n.id === segment.from);
    const toNode = this.graph.nodes.find(n => n.id === segment.to);

    // Process the question to get the node sequence
    const result = processQuestion(q, this.graph);
    const startNodeId = result.nodeSequence && result.nodeSequence.length > 0 ? result.nodeSequence[0].id : null;
    const endNodeId = result.nodeSequence && result.nodeSequence.length > 0 ? result.nodeSequence[result.nodeSequence.length - 1].id : null;

    // Check if this segment involves the start or end nodes
    const fromIsStart = segment.from === startNodeId;
    const toIsEnd = segment.to === endNodeId;

    return {
      from: fromNode,
      to: toNode,
      edge: this.graph.edges.find(e => e.id === segment.id),
      // Include start_detail if "from" is the start node
      fromIsStart: fromIsStart,
      startDetail: fromIsStart && q.nodes_involved.start_detail ? q.nodes_involved.start_detail : null,
      // Include end_detail if "to" is the end node
      toIsEnd: toIsEnd,
      endDetail: toIsEnd && q.nodes_involved.end_detail ? q.nodes_involved.end_detail : null,
    };
  }
}
