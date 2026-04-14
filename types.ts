
export enum NeuralEngine {
  GEMINI_3_FLASH_LITE = 'gemini-3.1-flash-lite-preview',
  GEMINI_3_FLASH = 'gemini-3-flash-preview',
  GEMINI_3_PRO = 'gemini-3.1-pro-preview',
  GPT_4O = 'gpt-4o',
  GROK_3 = 'grok-3',
  DEEPSEEK_V3 = 'deepseek-chat'
}

export type AcademicLevel = 'Level 1' | 'Level 2' | 'Level 3' | 'Level 4' | 'Level 5' | 'Level 6' | 'Level 7' | 'Level 8' | 'Level 9' | 'Level 10' | 'Level 11' | 'Kid' | 'Beginner' | 'Pre-Elementary' | 'Elementary' | 'Higher Elementary' | 'Pre-Intermediate' | 'Intermediate' | 'Upper Intermediate' | 'Advanced' | 'TOEFL' | 'IELTS';
export type AnswerStrategy = 'TOPIC_FOCUSED' | 'GENERAL_MIXED' | 'GLOBAL_RANDOM';
export type WorksheetLength = 'Short' | 'Medium' | 'Long';
export type SettingsTab = 'COMMAND' | 'ACCOUNT' | 'ENGINE' | 'BACKBONE LOGIC' | 'DESIGN' | 'FORMAT_DESIGN' | 'LOGO';
export type PaperType = 'Plain' | 'Ruled' | 'Handwriting' | 'Dotted' | 'Grid';
export type LayoutStyle = 'Standard' | 'Full Columns' | 'Variable Layout';
export type Priority = 'High' | 'Medium' | 'Average' | 'Low';
export type RuleCategory = 'General' | 'Grammar' | 'Vocabulary' | 'Reading' | 'Generals' | 'Custom' | 'Mixed';

export interface ExternalKeys {
  [NeuralEngine.GPT_4O]?: string;
  [NeuralEngine.GROK_3]?: string;
  [NeuralEngine.DEEPSEEK_V3]?: string;
  [NeuralEngine.GEMINI_3_PRO]?: string;
  [NeuralEngine.GEMINI_3_FLASH]?: string;
}

export interface PromptConfig {
  id: string;
  label: string;
  unifiedLogic: string;
  selectedInstructionIds?: string[];
}

export interface StrictRule {
  id: string;
  label: string;
  description: string;
  promptInjection: string;
  active: boolean;
  priority: Priority;
  category: RuleCategory;
  isCustomized?: boolean;
}

export interface InstructionTemplate {
  id: string;
  label: string;
  professionalLabel?: string;
  prompt: string;
  category: 'GRAMMAR' | 'VOCABULARY' | 'READING' | 'GENERALS' | 'TABLES' | 'KIDS' | 'CUSTOM';
  displayLetter?: string;
  columnCount?: number;
  isCustomized?: boolean;
}

export interface ModuleLogic {
  module: string;
  prompts: PromptConfig[];
}

export interface HistoryItem {
  id: string;
  title: string;
  content: string;
  timestamp: number;
  promptId: string;
  logicSnapshot: string;
  module: string;
  level: AcademicLevel;
  topic: string;
  thought?: string;
}

export interface QuickSource {
  data: string;
  mimeType: string;
  name: string;
}

export interface Theme {
  id: string;
  name: string;
  color: string;
  bg: string;
  accent: string;
}

export interface BrandSettings {
  fontSize: number;
  fontWeight: string;
  letterSpacing: number;
  textTransform: 'uppercase' | 'none' | 'capitalize';
  schoolName: string;
  schoolAddress: string;
  footerText: string;
  studentLabel: string;
  idLabel: string;
  scoreLabel: string;
  dateLabel: string;
  classLabel: string;
  teacherLabel: string;
  headerStyle: number;
  logos: (string | undefined)[]; 
  logoWidth: number;
  logoData?: string;
  activeFont?: string;
  randomizeFont?: boolean;
  customHeaderText?: string;
  headerRulerStyle?: number;
}

export interface UserSession {
  name: string;
  code: string;
  loginTime: number;
  email?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'architect';
  text: string;
  timestamp: number;
}

// Added OutlineItem interface to resolve missing member errors in services and components
export interface OutlineItem {
  id: string;
  title: string;
  expanded: boolean;
  children?: OutlineItem[];
}

export interface CustomExerciseType {
  id: string;
  name: string;
  category: RuleCategory;
  styleId?: string;
  uid?: string;
}
