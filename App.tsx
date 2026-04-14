import React, { useState, useEffect, useRef } from 'react';
import {
  NeuralEngine,
  AcademicLevel,
  HistoryItem,
  QuickSource,
  StrictRule,
  SettingsTab,
  UserSession,
  BrandSettings,
  InstructionTemplate,
  Priority,
  RuleCategory,
  ExternalKeys,
  ChatMessage,
  AnswerStrategy,
  CustomExerciseType
} from './types';
import {
  INITIAL_MODULES,
  LANGUAGES,
  ACADEMIC_LEVELS,
  GLOBAL_STRICT_COMMAND,
  BORDER_FRAME_INSTRUCTION,
  PART_BACKGROUND_INSTRUCTION,
  INSTRUCTION_HEADER_BACKGROUND_INSTRUCTION,
  PAGE_STYLES,
  DEFAULT_STRICT_RULES,
  DEFAULT_MASTER_PROTOCOLS,
  INITIAL_TEMPLATES,
  THEMES,
  FONTS,
  SUBJECTS,
  PAPER_DESIGNS
} from './constants';

// --- THE NEW FIREBASE MAGIC ---
import { db, auth, googleProvider } from './firebase';
import { collection, addDoc, doc, getDoc, setDoc, updateDoc, query, where, getDocs, orderBy, limit, getDocFromServer, onSnapshot, Timestamp } from 'firebase/firestore';
import { onAuthStateChanged, signInWithPopup, signOut } from 'firebase/auth';

enum OperationType {
  CREATE = 'create',
  UPDATE = 'update',
  DELETE = 'delete',
  LIST = 'list',
  GET = 'get',
  WRITE = 'write',
}

interface FirestoreErrorInfo {
  error: string;
  operationType: OperationType;
  path: string | null;
  authInfo: {
    userId: string | undefined;
    email: string | null | undefined;
    emailVerified: boolean | undefined;
    isAnonymous: boolean | undefined;
    tenantId: string | null | undefined;
    providerInfo: {
      providerId: string;
      displayName: string | null;
      email: string | null;
      photoUrl: string | null;
    }[];
  }
}

function handleFirestoreError(error: unknown, operationType: OperationType, path: string | null) {
  const errInfo: FirestoreErrorInfo = {
    error: error instanceof Error ? error.message : String(error),
    authInfo: {
      userId: auth.currentUser?.uid,
      email: auth.currentUser?.email,
      emailVerified: auth.currentUser?.emailVerified,
      isAnonymous: auth.currentUser?.isAnonymous,
      tenantId: auth.currentUser?.tenantId,
      providerInfo: auth.currentUser?.providerData.map(provider => ({
        providerId: provider.providerId,
        displayName: provider.displayName,
        email: provider.email,
        photoUrl: provider.photoURL
      })) || []
    },
    operationType,
    path
  }
  console.error('Firestore Error: ', JSON.stringify(errInfo));
  throw new Error(JSON.stringify(errInfo));
}

export class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean, error: Error | null }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("ErrorBoundary caught an error", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      let errorMessage = "Something went wrong.";
      try {
        const parsed = JSON.parse(this.state.error?.message || "");
        if (parsed.error) errorMessage = `Firestore Error: ${parsed.error} (${parsed.operationType} on ${parsed.path})`;
      } catch {
        errorMessage = this.state.error?.message || errorMessage;
      }

      return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4">
          <div className="bg-white p-8 rounded-[32px] shadow-xl border border-slate-100 max-w-md w-full text-center space-y-6">
            <div className="h-16 w-16 bg-rose-100 text-rose-600 rounded-2xl flex items-center justify-center mx-auto text-2xl">
              <i className="fa-solid fa-circle-exclamation"></i>
            </div>
            <h2 className="text-xl font-black text-slate-900 uppercase">System Interruption</h2>
            <p className="text-slate-500 text-sm leading-relaxed">{errorMessage}</p>
            <button 
              onClick={() => window.location.reload()}
              className="w-full py-4 bg-slate-900 text-white rounded-2xl font-black uppercase tracking-widest hover:bg-slate-800 transition-all"
            >
              Restart Application
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
// ------------------------------

import { callNeuralEngine } from './services/neuralService';
import { exportToWord, exportToPDF, exportToHTML } from './services/wordExportService';
import Worksheet from './components/Worksheet';
import NeuralChatAssistant from './components/NeuralChatAssistant'
import Sidebar from './components/Sidebar';
import FormatDesignEditor from './components/FormatDesignEditor';
import { OnboardingTutorial } from './components/OnboardingTutorial';
const DEFAULT_BRAND_SETTINGS: BrandSettings = {
  fontSize: 12,
  fontWeight: '800',
  letterSpacing: 0,
  textTransform: 'none',
  schoolName: 'GLOBAL EDUCATION ACADEMY',
  schoolAddress: 'Developing Potential for Success School',
  footerText: 'This test is for educational purposes only. © 2026 DPSS.',
  studentLabel: 'STUDENT NAME',
  idLabel: 'STUDENT ID',
  scoreLabel: 'SCORE',
  dateLabel: 'DATE',
  classLabel: 'CLASS',
  teacherLabel: 'TEACHER',
  headerStyle: 10,
  logos: Array(30).fill(undefined),
  logoWidth: 300,
  logoData: undefined,
  activeFont: 'Garamond',
  randomizeFont: false
};

const DEFAULT_SESSION: UserSession = {
  name: 'Public User',
  email: 'public@dpss.edu',
  code: 'dpss',
  loginTime: Date.now()
};

const MASTER_PROTOCOLS_KEY = 'dp_master_v46';
const STRICT_RULES_KEY = 'dp_rules_v46';
const TEMPLATES_KEY = 'dp_templates_v46';
const HISTORY_KEY = 'dp_history_v46';
const BRAND_SETTINGS_KEY = 'dp_brand_v46';
const USER_SESSION_KEY = 'dp_session_v46';
const ENGINE_CONFIG_KEY = 'dp_engine_config_v46';
const ONBOARDING_KEY = 'dp_onboarding_v1';

const toTitleCase = (str: string) => {
  return str.toLowerCase().split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
};

const CollapsibleSection: React.FC<{
  title: string;
  subtitle?: string;
  icon: string;
  iconBg: string;
  iconColor: string;
  isCollapsed: boolean;
  onToggle: () => void;
  children: React.ReactNode;
  rightElement?: React.ReactNode;
}> = ({ title, subtitle, icon, iconBg, iconColor, isCollapsed, onToggle, children, rightElement }) => (
  <div className="space-y-6">
    <div className="flex items-center justify-between group cursor-pointer" onClick={onToggle}>
      <div className="flex items-center gap-4">
        <div className={`h-10 w-10 ${iconBg} rounded-2xl flex items-center justify-center ${iconColor} transition-transform duration-300 ${isCollapsed ? '' : 'rotate-12'}`}>
          <i className={`fa-solid ${icon}`}></i>
        </div>
        <div>
          <h4 className="text-[14px] font-black text-slate-900 uppercase tracking-tight">{title}</h4>
          {subtitle && <p className="text-[10px] font-medium text-slate-400 uppercase">{subtitle}</p>}
        </div>
      </div>
      <div className="flex items-center gap-4">
        {rightElement}
        <div className={`h-8 w-8 rounded-xl flex items-center justify-center text-slate-400 hover:bg-slate-100 transition-all ${isCollapsed ? '' : 'rotate-180'}`}>
          <i className="fa-solid fa-chevron-down text-[10px]"></i>
        </div>
      </div>
    </div>
    {!isCollapsed && (
      <div className="animate-in slide-in-from-top-4 duration-300">
        {children}
      </div>
    )}
  </div>
);

function App() {
  const [session, setSession] = useState<UserSession>({
    name: 'Public User',
    email: 'public@dpss.edu',
    code: 'dpss',
    loginTime: Date.now()
  });

  const [authLoading, setAuthLoading] = useState(true);
  const [isAuthReady, setIsAuthReady] = useState(false);
  const [firebaseError, setFirebaseError] = useState<string | null>(null);

  const [viewMode, setViewMode] = useState<'generator' | 'preview' | 'book_creation' | 'ielts_master' | 'dpss_studio' | 'grammar_iframe' | 'khmer_program' | 'design_test_style' | 'header_footer_design' | 'paper_style_design' | 'instruction_design' | 'mcq_grid'>('generator');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const [sidebarSide, setSidebarSide] = useState<'left' | 'right'>('left');
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);

  useEffect(() => {
    async function testConnection() {
      try {
        await getDocFromServer(doc(db, 'test', 'connection'));
      } catch (error) {
        if (error instanceof Error && error.message.includes('the client is offline')) {
          console.error("Please check your Firebase configuration. The client is offline.");
          setFirebaseError("Firebase configuration error: The client is offline. Please check your network or configuration.");
        }
      }
    }
    testConnection();

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        setSession({
          name: user.displayName || 'User',
          email: user.email || '',
          code: 'dpss',
          loginTime: Date.now()
        });
      } else {
        setSession(DEFAULT_SESSION);
      }
      setAuthLoading(false);
      setIsAuthReady(true);
    });

    return () => unsubscribe();
  }, []);

  // Sync settings with Firestore
  useEffect(() => {
    if (!isAuthReady || !auth.currentUser) return;

    const userSettingsRef = doc(db, 'users', auth.currentUser.uid);
    
    const unsubscribe = onSnapshot(userSettingsRef, (docSnap) => {
      if (docSnap.exists()) {
        const data = docSnap.data();
        if (data.brandSettings) setBrandSettings(data.brandSettings);
        if (data.paperStyles) setPaperStyles(data.paperStyles);
        if (data.paperDesign !== undefined) setPaperDesign(data.paperDesign);
      }
    }, (error) => {
      handleFirestoreError(error, OperationType.GET, `users/${auth.currentUser?.uid}`);
    });

    return () => unsubscribe();
  }, [isAuthReady]);

  // Sync custom designs with Firestore
  useEffect(() => {
    if (!isAuthReady || !auth.currentUser) return;

    const q = query(collection(db, 'customDesigns'), where('uid', '==', auth.currentUser.uid));
    
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const designs = snapshot.docs.map(doc => ({ ...doc.data(), id: doc.id } as any));
      setCustomDesigns(designs);
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, 'customDesigns');
    });

    return () => unsubscribe();
  }, [isAuthReady]);

  // Sync custom exercise types with Firestore
  useEffect(() => {
    if (!isAuthReady || !auth.currentUser) return;

    const q = query(collection(db, 'customExerciseTypes'), where('uid', '==', auth.currentUser.uid));
    
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const types = snapshot.docs.map(doc => ({ ...doc.data(), id: doc.id } as any));
      if (types.length > 0) {
        setCustomExerciseTypes(prev => {
          // Merge with defaults if needed, but usually we just want the cloud ones
          return types;
        });
      }
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, 'customExerciseTypes');
    });

    return () => unsubscribe();
  }, [isAuthReady]);

  // Sync history with Firestore
  useEffect(() => {
    if (!isAuthReady || !auth.currentUser) return;

    const q = query(
      collection(db, 'history'), 
      where('uid', '==', auth.currentUser.uid),
      orderBy('timestamp', 'desc'),
      limit(30)
    );
    
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const cloudHistory = snapshot.docs.map(doc => ({ ...doc.data(), id: doc.id } as HistoryItem));
      setHistory(cloudHistory);
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, 'history');
    });

    return () => unsubscribe();
  }, [isAuthReady]);

  const saveSettingsToFirestore = async () => {
    if (!auth.currentUser) return;
    try {
      const userSettingsRef = doc(db, 'users', auth.currentUser.uid);
      await setDoc(userSettingsRef, {
        uid: auth.currentUser.uid,
        brandSettings,
        paperStyles,
        paperDesign,
        updatedAt: Timestamp.now()
      }, { merge: true });
    } catch (error) {
      handleFirestoreError(error, OperationType.WRITE, `users/${auth.currentUser.uid}`);
    }
  };

  const handleAddCustomExerciseType = () => {
    setNewExerciseCategory(activeModule.charAt(0).toUpperCase() + activeModule.slice(1).toLowerCase());
    setNewExerciseName('');
    setShowAddExerciseModal(true);
  };

  const saveCustomExerciseType = async () => {
    if (newExerciseName.trim()) {
      const id = `custom_${Date.now()}`;
      const newType: CustomExerciseType = { 
        id, 
        name: newExerciseName.trim(), 
        category: newExerciseCategory as any,
        uid: auth.currentUser?.uid || 'anonymous'
      };

      // Update state
      setCustomExerciseTypes(prev => [...prev, newType]);

      // Save to Firestore if logged in
      if (auth.currentUser) {
        try {
          await setDoc(doc(db, 'customExerciseTypes', id), newType);
        } catch (error) {
          handleFirestoreError(error, OperationType.WRITE, `customExerciseTypes/${id}`);
        }
      }
      setShowAddExerciseModal(false);
    }
  };

  const handleDeleteCustomDesign = async (id: string) => {
    if (!confirm("Are you sure you want to delete this design?")) return;
    
    setCustomDesigns(prev => prev.filter(d => d.id !== id));
    
    if (auth.currentUser) {
      try {
        const { deleteDoc } = await import('firebase/firestore');
        await deleteDoc(doc(db, 'customDesigns', id));
      } catch (error) {
        handleFirestoreError(error, OperationType.DELETE, `customDesigns/${id}`);
      }
    }
  };
  const [isAssistantVisible, setIsAssistantVisible] = useState(false);
  const [activeModule, setActiveModule] = useState<string>('Grammar');
  const [activeLanguage, setActiveLanguage] = useState<string>('English');
  const [activeLevel, setActiveLevel] = useState<AcademicLevel>('Level 1');
  const [answerStrategy, setAnswerStrategy] = useState<AnswerStrategy>('GENERAL_MIXED');
  const [topic, setTopic] = useState<string>('English Grammar Test');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationStep, setGenerationStep] = useState<string>('');
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [worksheetContent, setWorksheetContent] = useState<string>('');
  const [showSettings, setShowSettings] = useState(false);
  const [showRenameView, setShowRenameView] = useState(false);
  const [activeModuleTitle, setActiveModuleTitle] = useState('');
  const [isSettingsFullScreen, setIsSettingsFullScreen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<SettingsTab>('COMMAND');
  const [isFrameEnabled, setIsFrameEnabled] = useState(true);
  const [enablePages, setEnablePages] = useState(false);
  const [isPartBackgroundEnabled, setIsPartBackgroundEnabled] = useState(false);
  const [isInstructionBackgroundEnabled, setIsInstructionBackgroundEnabled] = useState(true);
  const [isColorfulBackgroundEnabled, setIsColorfulBackgroundEnabled] = useState(true);
  const [instructionCase, setInstructionCase] = useState<'uppercase' | 'lowercase' | 'random'>('uppercase');
  const [activeSubject, setActiveSubject] = useState<string>('cambodia');
  const [isRandomSubject, setIsRandomSubject] = useState(false);
  const [showSubjectModal, setShowSubjectModal] = useState(false);
  const [showAddExerciseModal, setShowAddExerciseModal] = useState(false);
  const [newExerciseName, setNewExerciseName] = useState('');
  const [newExerciseCategory, setNewExerciseCategory] = useState<string>('Grammar');
  const [globalLayout, setGlobalLayout] = useState<number>(0); // 0-19: Paper Styles
  const [tableStyle, setTableStyle] = useState<string>('plain'); // plain, grid, list
  const [baseLayout, setBaseLayout] = useState<number>(() => {
    const saved = localStorage.getItem('dp_base_layout');
    return saved ? parseInt(saved) : 0;
  }); // 0: Clean, 1: Lined, 2: Grid, 3: Vertical Middle, 4: Rulers Left, 5-8: S1-S4
  const [instructionRulerStyle, setInstructionRulerStyle] = useState<number>(() => {
    const saved = localStorage.getItem('dp_instruction_ruler');
    return saved ? parseInt(saved) : 0;
  }); // 0: None, 1-6: S1-S6
  const [isBottomPanelHidden, setIsBottomPanelHidden] = useState(false);
  const [isCountriesHidden, setIsCountriesHidden] = useState(false);
  const [customArchitectSubTab, setCustomArchitectSubTab] = useState<string>('All');
  const [paperDesign, setPaperDesign] = useState<number>(() => {
    try {
      const saved = localStorage.getItem('paperDesign');
      return saved !== null ? parseInt(saved) : 18; // Default to Style 9: Modern Green
    } catch {
      return 18;
    }
  });
  const [instructionHeaderStyle, setInstructionHeaderStyle] = useState<number>(0); // 0: Default, 1-10: Styles
  const [defaultColumnCount, setDefaultColumnCount] = useState<number>(1); // 1-6 columns
  const [architectTab, setArchitectTab] = useState<'Grammar' | 'Vocabulary' | 'Reading' | 'Mixed' | 'Generals' | 'Custom'>('Grammar');
  const [collapsedSections, setCollapsedSections] = useState<Record<string, boolean>>({});
  const [mcqLayout, setMcqLayout] = useState<'single' | 'double' | 'quad'>(() => {
    const saved = localStorage.getItem('dp_mcq_layout');
    return saved as 'single' | 'double' | 'quad' || 'single';
  }); // A,B,C,D in 1, 2, or 4 lines
  const [mcqSpacing, setMcqSpacing] = useState<'none' | 'one'>(() => {
    const saved = localStorage.getItem('dp_mcq_spacing');
    return saved as 'none' | 'one' || 'none';
  }); // No space or one enter space

  useEffect(() => {
    localStorage.setItem('dp_base_layout', baseLayout.toString());
  }, [baseLayout]);
  useEffect(() => {
    localStorage.setItem('dp_instruction_ruler', instructionRulerStyle.toString());
  }, [instructionRulerStyle]);
  useEffect(() => {
    localStorage.setItem('dp_mcq_layout', mcqLayout);
  }, [mcqLayout]);
  useEffect(() => {
    localStorage.setItem('dp_mcq_spacing', mcqSpacing);
  }, [mcqSpacing]);
  const [mcqStyle, setMcqStyle] = useState<number>(() => {
    const saved = localStorage.getItem('dp_default_mcq_style');
    return saved ? parseInt(saved) : 0;
  });
  const [paperStyles, setPaperStyles] = useState<{
    mcq: number | string;
    tf: number | string;
    correctIncorrect: number | string;
    vocabulary: number | string;
    circle: number | string;
    sentenceCompletion: number | string;
    wordBox: number | string;
    readingPassage: number | string;
    matching: string;
    cloze: number | string;
    doubleMcq: number | string;
  }>(() => {
    try {
      const saved = localStorage.getItem('dp_paper_styles_v2');
      return saved ? JSON.parse(saved) : {
        mcq: 0,
        tf: 0,
        correctIncorrect: 3, // Default to Design 4
        vocabulary: 0,
        circle: 0,
        sentenceCompletion: 0,
        wordBox: 0,
        readingPassage: 0,
        matching: 'classic',
        cloze: 0,
        doubleMcq: 0
      };
    } catch {
      return {
        mcq: 0,
        tf: 0,
        correctIncorrect: 3, // Default to Design 4
        vocabulary: 0,
        circle: 0,
        sentenceCompletion: 0,
        wordBox: 0,
        readingPassage: 0,
        matching: 'classic',
        cloze: 0,
        doubleMcq: 0
      };
    }
  });

  useEffect(() => {
    localStorage.setItem('dp_paper_styles_v2', JSON.stringify(paperStyles));
  }, [paperStyles]);

  const [customDesigns, setCustomDesigns] = useState<{id: string, name: string, type: string, category: string, style: any, prompt?: string}[]>(() => {
    try {
      const saved = localStorage.getItem('dp_custom_designs_v46');
      const initial = saved ? JSON.parse(saved) : [];
      
      // Add requested custom designs if they don't exist
      const required = [
        {
          id: 'vocab-match-pro',
          name: 'Vocabulary Match Pro',
          type: 'vocabulary',
          category: 'Vocabulary',
          style: { vocabulary: 4 }, // Assuming 4 is a new style or we handle it in prompt
          prompt: 'Use a professional HTML table with zebra striping (alternate row backgrounds). Column 1: Number + Word. Column 2: Definition. Ensure definitions are perfectly aligned.'
        },
        {
          id: 'tf-checkbox-style',
          name: 'T/F Checkbox Style',
          type: 'tf',
          category: 'Reading',
          style: { tf: 3 }, // Assuming 3 is a new style or we handle it in prompt
          prompt: 'Use checkboxes before each statement (e.g., "1. [ ] Statement"). Apply a soft green background (#dcfce7) to the instruction header.'
        }
      ];

      const merged = Array.isArray(initial) ? [...initial] : [];
      required.forEach(req => {
        if (!merged.find(d => d.id === req.id)) {
          merged.push(req);
        }
      });
      return merged;
    } catch { return []; }
  });

  const [customExerciseTypes, setCustomExerciseTypes] = useState<CustomExerciseType[]>([
    { id: 'mcq', name: 'Multiple Choice (MCQ)', category: 'General' },
    { id: 'matching', name: 'Matching', category: 'General' },
    { id: 'true_false', name: 'True / False', category: 'General' },
    { id: 'correct_incorrect', name: 'Correct / Incorrect', category: 'General' },
    { id: 'rewrite', name: 'Rewrite the sentences', category: 'Grammar' }
  ]);
  const [designTargetTypeId, setDesignTargetTypeId] = useState<string | null>(null);
  const [editingCustomDesignId, setEditingCustomDesignId] = useState<string | null>(null);

  useEffect(() => {
    localStorage.setItem('dp_custom_designs_v46', JSON.stringify(customDesigns));
  }, [customDesigns]);
  
  const [isSingleReadingText, setIsSingleReadingText] = useState(false);
  const [isRelaxingBackgroundEnabled, setIsRelaxingBackgroundEnabled] = useState(true);
  const [currentBackground, setCurrentBackground] = useState("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?q=80&w=2000&auto=format&fit=crop"); // Ocean morning default

  const randomizeBackground = () => {
    const backgrounds = [
      "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?q=80&w=2000&auto=format&fit=crop", // Ocean morning
      "https://images.unsplash.com/photo-1471922694854-ff1b63b20054?q=80&w=2000&auto=format&fit=crop", // Ocean blue
      "https://images.unsplash.com/photo-1468413253725-0d5181091126?q=80&w=2000&auto=format&fit=crop", // Tropical beach
      "https://images.unsplash.com/photo-1505118380757-91f5f45d8de4?q=80&w=2000&auto=format&fit=crop", // Calm sea
      "https://images.unsplash.com/photo-1519046904884-53103b34b206?q=80&w=2000&auto=format&fit=crop"  // Beach sunset
    ];
    const randomBg = backgrounds[Math.floor(Math.random() * backgrounds.length)];
    setCurrentBackground(randomBg);
  };

  const [activeLogicCategory, setActiveLogicCategory] = useState<RuleCategory>('General');
  const [expandedRuleId, setExpandedRuleId] = useState<string | null>(null);
  const [activeProtocolCategory, setActiveProtocolCategory] = useState<RuleCategory>('General');
  const [expandedProtocolId, setExpandedProtocolId] = useState<string | null>(null);
  const [activeTemplateCategory, setActiveTemplateCategory] = useState<string>('GRAMMAR');
  const [expandedTemplateId, setExpandedTemplateId] = useState<string | null>(null);

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');

  const [activeThemeId, setActiveThemeId] = useState<string>(() => {
    try {
      const saved = localStorage.getItem('dp_theme_v30');
      return saved || 'default';
    } catch { return 'default'; }
  });

  const [activeEngine, setActiveEngine] = useState<NeuralEngine>(() => {
    try {
      const saved = localStorage.getItem(ENGINE_CONFIG_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        let active = parsed.active;
        // Migration for legacy model names
        if (
          active === 'gemini-3.1-flash-lite-preview' || 
          active === 'gemini-1.5-flash-lite' ||
          active === 'gemini-3-flash-lite'
        ) {
          active = NeuralEngine.GEMINI_3_FLASH_LITE;
        } else if (
          active === 'gemini-3-flash-preview' || 
          active === 'gemini-1.5-flash' ||
          active === 'gemini-3-flash'
        ) {
          active = NeuralEngine.GEMINI_3_FLASH;
        } else if (
          active === 'gemini-3.1-pro-preview' || 
          active === 'gemini-1.5-pro' ||
          active === 'gemini-3-pro'
        ) {
          active = NeuralEngine.GEMINI_3_PRO;
        }
        return active;
      }
      return NeuralEngine.GEMINI_3_FLASH_LITE;
    } catch { return NeuralEngine.GEMINI_3_FLASH_LITE; }
  });

  const [externalKeys, setExternalKeys] = useState<ExternalKeys>(() => {
    try {
      const saved = localStorage.getItem(ENGINE_CONFIG_KEY);
      return saved ? JSON.parse(saved).keys : {};
    } catch { return {}; }
  });
  
  const [brandSettings, setBrandSettings] = useState<BrandSettings>(() => {
    try {
      const saved = localStorage.getItem(BRAND_SETTINGS_KEY);
      const parsed = saved ? JSON.parse(saved) : DEFAULT_BRAND_SETTINGS;
      return { ...DEFAULT_BRAND_SETTINGS, ...parsed };
    } catch { return DEFAULT_BRAND_SETTINGS; }
  });

  const [isBrandLoaded, setIsBrandLoaded] = useState(false);
  const loadedEmailRef = useRef<string | null>(null);

  // Firebase Auth Listener
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        setSession({
          name: user.displayName || 'User',
          email: user.email || '',
          code: 'dpss',
          loginTime: Date.now()
        });
      } else {
        setSession(DEFAULT_SESSION);
      }
    });
    return () => unsubscribe();
  }, []);

  const handleGoogleLogin = async () => {
    try {
      setAuthLoading(true);
      await signInWithPopup(auth, googleProvider);
    } catch (error) {
      console.error("Login failed:", error);
      alert("Login failed. Please try again.");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await signOut(auth);
      setSession(DEFAULT_SESSION);
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  const [isFirebaseConnected, setIsFirebaseConnected] = useState(true);

  // Handle Module Defaults
  useEffect(() => {
    if (activeModule === 'Grammar') {
      setSelectedInstructionIds(['g_mcq', 'g_correct_incorrect', 'g_circle', 'g_complete_sentences', 'g_pair', 'g_best_rewrite']);
      setItemCountOverrides(prev => ({
        ...prev,
        'g_mcq': 10,
        'g_correct_incorrect': 20,
        'g_circle': 10,
        'g_complete_sentences': 10,
        'g_pair': 10,
        'g_best_rewrite': 10
      }));
      setColumnOverrides(prev => ({
        ...prev,
        'g_mcq': 1,
        'g_correct_incorrect': 2,
        'g_circle': 1,
        'g_complete_sentences': 1,
        'g_pair': 1,
        'g_best_rewrite': 1
      }));
    } else if (activeModule === 'Reading') {
      const readingIds = ['r_tf_stmt', 'r_mcq', 'r_short_answer', 'r_inferential', 'r_critical_thinking', 'r_tfng', 'r_mcq_expert', 'r_referential_qs', 'r_summary_cloze'];
      setSelectedInstructionIds(readingIds);
      setItemCountOverrides(prev => {
        const next = { ...prev };
        readingIds.forEach(id => next[id] = 10);
        return next;
      });
      setColumnOverrides(prev => {
        const next = { ...prev };
        readingIds.forEach(id => next[id] = 1);
        return next;
      });
    } else if (activeModule === 'Vocabulary') {
      const vocabIds = ['v_study_table', 'v_sentence_study', 'v_supply_terms', 'v_copy', 'v_synonym_writing', 'v_circle', 'v_mcq', 'v_speaking'];
      setSelectedInstructionIds(vocabIds);
      setItemCountOverrides(prev => {
        const next = { ...prev };
        vocabIds.forEach(id => next[id] = 15);
        return next;
      });
      setColumnOverrides(prev => {
        const next = { ...prev };
        vocabIds.forEach(id => next[id] = 1);
        return next;
      });
    } else if (activeModule === 'Mixed') {
      const mixedIds = ['g_mcq', 'g_correct_incorrect', 'r_tf_stmt', 'v_study_table'];
      setSelectedInstructionIds(mixedIds);
      setItemCountOverrides(prev => {
        const next = { ...prev };
        mixedIds.forEach(id => next[id] = 10);
        return next;
      });
      setColumnOverrides(prev => {
        const next = { ...prev };
        mixedIds.forEach(id => next[id] = 1);
        return next;
      });
    }
  }, [activeModule]);

  // Validate connection to Firestore
  useEffect(() => {
    const testConnection = async () => {
      try {
        await getDocFromServer(doc(db, 'test', 'connection'));
        setIsFirebaseConnected(true);
      } catch (error) {
        if (error instanceof Error && error.message.includes('the client is offline')) {
          console.error("Please check your Firebase configuration. The client is offline.");
          setIsFirebaseConnected(false);
        }
      }
    };
    testConnection();
  }, []);

  // Fetch brand settings from Firestore on login
  useEffect(() => {
    const fetchBrandSettings = async () => {
      // Reset load state when user changes
      setIsBrandLoaded(false);
      loadedEmailRef.current = null;

      if (session?.email) {
        try {
          const docRef = doc(db, 'user_settings', session.email);
          const docSnap = await getDoc(docRef);
          if (docSnap.exists() && docSnap.data().brandSettings) {
            setBrandSettings(docSnap.data().brandSettings);
          }
          // Mark this email as loaded
          loadedEmailRef.current = session.email;
        } catch (e) {
          handleFirestoreError(e, 'get' as any, `user_settings/${session.email}`);
        } finally {
          setIsBrandLoaded(true);
        }
      } else {
        setIsBrandLoaded(true);
      }
    };
    fetchBrandSettings();
  }, [session?.email]);
  
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      const saved = localStorage.getItem(HISTORY_KEY);
      const parsed = saved ? JSON.parse(saved) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      console.error("Failed to load history:", e);
      return [];
    }
  });

  const fetchCloudHistory = async (email: string) => {
    try {
      const q = query(
        collection(db, 'generatedTests'),
        where('authorEmail', '==', email)
      );
      const querySnapshot = await getDocs(q);
      const cloudHistory: HistoryItem[] = [];
      querySnapshot.forEach((doc) => {
        cloudHistory.push(doc.data() as HistoryItem);
      });
      
      // Sort in memory to avoid composite index requirement
      const sortedHistory = cloudHistory
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(0, 30);

      if (sortedHistory.length > 0) {
        setHistory(sortedHistory);
      }
    } catch (e) {
      handleFirestoreError(e, 'list' as any, 'generatedTests');
    }
  };

  useEffect(() => {
    if (session?.email) {
      fetchCloudHistory(session.email);
    }
  }, [session?.email]);

  const [masterProtocols, setMasterProtocols] = useState<StrictRule[]>(() => {
    try {
      const saved = localStorage.getItem(MASTER_PROTOCOLS_KEY);
      let parsed = saved ? JSON.parse(saved) : DEFAULT_MASTER_PROTOCOLS;
      if (!Array.isArray(parsed)) parsed = DEFAULT_MASTER_PROTOCOLS;
      
      // Force update existing defaults from constants.ts
      const updated = parsed.map((p: any) => {
        if (p.isCustomized) return p;
        const fresh = DEFAULT_MASTER_PROTOCOLS.find(f => f.id === p.id);
        return fresh ? { ...p, ...fresh } : p;
      });

      // Auto-merge missing defaults
      const existingIds = new Set(updated.map((p: any) => p.id));
      const missing = DEFAULT_MASTER_PROTOCOLS.filter(p => !existingIds.has(p.id));
      return [...updated, ...missing];
    } catch { return DEFAULT_MASTER_PROTOCOLS; }
  });
  const [strictRules, setStrictRules] = useState<StrictRule[]>(() => {
    try {
      const saved = localStorage.getItem(STRICT_RULES_KEY);
      let parsed = saved ? JSON.parse(saved) : DEFAULT_STRICT_RULES;
      if (!Array.isArray(parsed)) parsed = DEFAULT_STRICT_RULES;

      // Force update existing defaults from constants.ts
      const updated = parsed.map((r: any) => {
        if (r.isCustomized) return r;
        const fresh = DEFAULT_STRICT_RULES.find(f => f.id === r.id);
        return fresh ? { ...r, ...fresh } : r;
      });

      // Auto-merge missing defaults
      const existingIds = new Set(updated.map((r: any) => r.id));
      const missing = DEFAULT_STRICT_RULES.filter(r => !existingIds.has(r.id));
      return [...updated, ...missing];
    } catch { return DEFAULT_STRICT_RULES; }
  });
  const [instructionTemplates, setInstructionTemplates] = useState<InstructionTemplate[]>(() => {
    try {
      const saved = localStorage.getItem(TEMPLATES_KEY);
      let parsed = saved ? JSON.parse(saved) : INITIAL_TEMPLATES;
      if (!Array.isArray(parsed)) parsed = INITIAL_TEMPLATES;
      
      // Force update all fields from INITIAL_TEMPLATES for existing IDs
      const updated = parsed.map((t: any) => {
        if (t.isCustomized) return t;
        const fresh = INITIAL_TEMPLATES.find(f => f.id === t.id);
        if (fresh) {
          return {
            ...t,
            ...fresh
          };
        }
        return t;
      });

      // Auto-merge missing defaults
      const existingIds = new Set(updated.map((t: any) => t.id));
      const missing = INITIAL_TEMPLATES.filter(t => !existingIds.has(t.id));
      return [...updated, ...missing];
    } catch { return INITIAL_TEMPLATES; }
  });

  const [selectedInstructionIds, setSelectedInstructionIds] = useState<string[]>(['g_mcq', 'g_correct_incorrect', 'g_circle', 'g_complete_sentences', 'g_pair', 'g_spelling']);
  const [columnOverrides, setColumnOverrides] = useState<Record<string, number>>({
    'g_mcq': 1,
    'g_correct_incorrect': 2,
    'g_circle': 1,
    'g_complete_sentences': 1,
    'g_pair': 1,
    'g_spelling': 0
  });
  const [itemCountOverrides, setItemCountOverrides] = useState<Record<string, number>>({
    'g_mcq': 10,
    'g_correct_incorrect': 20,
    'g_circle': 10,
    'g_complete_sentences': 10,
    'g_pair': 10,
    'g_spelling': 30,
    'g_write_correct_form': 4,
    'g_rewrite_sentences': 5
  });
  
  const [sourceMaterial, setSourceMaterial] = useState<QuickSource | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const logoUploadRef = useRef<HTMLInputElement>(null);

  const [loginName, setLoginName] = useState('');
  const [loginCode, setLoginCode] = useState('');
  const [loginError, setLoginError] = useState('');

  const [showOnboarding, setShowOnboarding] = useState(() => {
    try {
      const saved = localStorage.getItem(ONBOARDING_KEY);
      return saved !== 'completed';
    } catch { return true; }
  });

  const [exportSettings, setExportSettings] = useState({
    filename: '',
    title: '',
    showModal: false
  });

  useEffect(() => { 
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); 
    } catch (e) {
      console.warn("History storage limit reached. Oldest items may be lost.", e);
      // Optional: Try to save a smaller subset if full save fails
      try {
        const smallerHistory = history.slice(0, 10);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(smallerHistory));
      } catch (innerE) {
        console.error("Critical storage failure for history", innerE);
      }
    }
  }, [history]);

  useEffect(() => { 
    try {
      localStorage.setItem(TEMPLATES_KEY, JSON.stringify(instructionTemplates)); 
    } catch (e) { console.warn("Templates storage limit reached", e); }
  }, [instructionTemplates]);

  useEffect(() => { 
    try {
      localStorage.setItem(STRICT_RULES_KEY, JSON.stringify(strictRules)); 
    } catch (e) { console.warn("Rules storage limit reached", e); }
  }, [strictRules]);

  useEffect(() => { 
    try {
      localStorage.setItem(MASTER_PROTOCOLS_KEY, JSON.stringify(masterProtocols)); 
    } catch (e) { console.warn("Protocols storage limit reached", e); }
  }, [masterProtocols]);
  useEffect(() => { 
    try {
      localStorage.setItem(BRAND_SETTINGS_KEY, JSON.stringify(brandSettings)); 
    } catch (e) {
      console.warn("Storage quota exceeded. Some branding settings might not persist locally.", e);
    }
    
    // Persist brand settings to Firestore
    const persistBrandSettings = async () => {
      // Only save if we are logged in AND the current user's data has been loaded
      if (session?.email && isBrandLoaded && loadedEmailRef.current === session.email) {
        try {
          const docRef = doc(db, 'user_settings', session.email);
          await setDoc(docRef, { brandSettings }, { merge: true });
        } catch (e) {
          handleFirestoreError(e, 'write' as any, `user_settings/${session.email}`);
          // Alert user if save fails, likely due to size
          if (e instanceof Error && e.message.includes('too large')) {
             alert("CRITICAL: Your logo collection is too large to save to the cloud. Please delete some logos or use smaller images.");
          }
        }
      }
    };
    persistBrandSettings();
  }, [brandSettings, session?.email, isBrandLoaded]);
  useEffect(() => { 
    localStorage.setItem('dp_theme_v30', activeThemeId); 
    const theme = THEMES.find(t => t.id === activeThemeId) || THEMES[0];
    
    // Core Colors
    document.documentElement.style.setProperty('--primary-orange', theme.color);
    document.documentElement.style.setProperty('--accent-orange-light', theme.accent);
    document.documentElement.style.setProperty('--accent-orange-dark', theme.color);
    
    // Body Background
    const body = document.body;
    if (theme.bg.startsWith('linear-gradient') || theme.bg.startsWith('radial-gradient')) {
      body.style.background = theme.bg;
    } else {
      body.style.background = theme.bg;
      body.style.backgroundImage = 'none';
    }

    // Handle text contrast (simple heuristic)
    const isDark = theme.id === 'midnight' || theme.id === 'nebula';
    body.style.color = isDark ? '#f8fafc' : '#1e293b';
    
    // Update sidebar/main backgrounds if they are too dark
    const main = document.querySelector('main');
    const aside = document.querySelector('aside');
    if (main) {
      main.style.backgroundColor = isDark ? '#0b1221' : 'rgba(255, 255, 255, 0.4)';
      main.style.backdropFilter = 'blur(20px)';
    }
    if (aside) {
      aside.style.backgroundColor = isDark ? '#0b1221' : 'rgba(255, 255, 255, 0.6)';
      aside.style.backdropFilter = 'blur(20px)';
    }

  }, [activeThemeId]);
  useEffect(() => { 
    localStorage.setItem(ENGINE_CONFIG_KEY, JSON.stringify({ active: activeEngine, keys: externalKeys }));
  }, [activeEngine, externalKeys]);

  useEffect(() => {
    const interval = setInterval(() => {
      const randomTheme = THEMES[Math.floor(Math.random() * THEMES.length)];
      setActiveThemeId(randomTheme.id);
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const cyclePriority = (current: Priority): Priority => {
    const priorities: Priority[] = ['Low', 'Average', 'Medium', 'High'];
    const currentIndex = priorities.indexOf(current);
    return priorities[(currentIndex + 1) % priorities.length];
  };

  const handleOnboardingComplete = () => {
    setShowOnboarding(false);
    localStorage.setItem(ONBOARDING_KEY, 'completed');
  };

  const toggleInstruction = (id: string) => setSelectedInstructionIds(prev => prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]);
  const setItemCount = (id: string, count: number) => setItemCountOverrides(prev => ({ ...prev, [id]: count }));
  const adjustColumns = (id: string, delta: number) => {
    setColumnOverrides(prev => ({ ...prev, [id]: Math.max(0, Math.min(6, (prev[id] || 0) + delta)) }));
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => setSourceMaterial({ data: (event.target?.result as string).split(',')[1], mimeType: file.type, name: file.name });
      reader.readAsDataURL(file);
    }
  };

  const handleLogoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // 1. Initial size check
      if (file.size > 10 * 1024 * 1024) {
        alert("Image is too large. Please use a file smaller than 10MB.");
        if (logoUploadRef.current) logoUploadRef.current.value = '';
        return;
      }

      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          // 2. Resize & Compress Logic
          const canvas = document.createElement('canvas');
          let width = img.width;
          let height = img.height;
          
          // Max dimension 600px is sufficient for A4 header logos
          // This keeps file size very low (~50-100KB)
          const MAX_DIM = 600; 
          
          if (width > height) {
            if (width > MAX_DIM) {
              height *= MAX_DIM / width;
              width = MAX_DIM;
            }
          } else {
            if (height > MAX_DIM) {
              width *= MAX_DIM / height;
              height = MAX_DIM;
            }
          }
          
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          if (ctx) {
             ctx.drawImage(img, 0, 0, width, height);
             // Compress to JPEG 0.7 quality
             const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
             
             setBrandSettings(prev => {
               const newLogos = [...prev.logos];
               const firstEmpty = newLogos.findIndex(l => !l);
               if (firstEmpty !== -1) {
                 newLogos[firstEmpty] = dataUrl;
               } else {
                 newLogos.push(dataUrl);
               }
               return { ...prev, logos: newLogos, logoData: dataUrl };
             });
          }
          
          if (logoUploadRef.current) logoUploadRef.current.value = '';
        };
        img.src = event.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
  };

  const removeLogo = (index: number) => {
    setBrandSettings(prev => {
      const newLogos = [...prev.logos];
      newLogos[index] = undefined;
      return { ...prev, logos: newLogos };
    });
  };

  const generateNeuralBlueprint = (count: number) => {
    const keys = ['A', 'B', 'C', 'D'];
    let blueprint: string[] = [];
    
    // Bucket logic: For every 10 items, pre-select a bucket
    const numBuckets = Math.ceil(count / 10);
    
    for (let b = 0; b < numBuckets; b++) {
      const bucketSize = Math.min(10, count - b * 10);
      const bucket: string[] = [];
      
      if (bucketSize === 10) {
        // Specific distribution for 10: e.g. 3A, 2B, 2C, 3D
        const dist = ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'D'];
        bucket.push(...dist);
      } else {
        // Mandatory presence for smaller buckets
        const mandatoryKeys = bucketSize >= 4 ? [...keys] : keys.slice(0, bucketSize);
        bucket.push(...mandatoryKeys);
        while (bucket.length < bucketSize) {
          bucket.push(keys[Math.floor(Math.random() * keys.length)]);
        }
      }
      
      // Shuffle the bucket
      for (let i = bucket.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [bucket[i], bucket[j]] = [bucket[j], bucket[i]];
      }
      
      blueprint.push(...bucket);
    }

    // Enforce Streak Limit: Max 2 identical
    for (let i = 2; i < blueprint.length; i++) {
      if (blueprint[i] === blueprint[i-1] && blueprint[i] === blueprint[i-2]) {
        for (let j = i + 1; j < blueprint.length; j++) {
          if (blueprint[j] !== blueprint[i]) {
            [blueprint[i], blueprint[j]] = [blueprint[j], blueprint[i]];
            break;
          }
        }
      }
    }

    return blueprint;
  };

  const applyCustomDesign = (design: any) => {
    setPaperStyles(prev => {
      const next = { ...prev };
      if (design.type === 'mcq') next.mcq = design.id;
      else if (design.type === 'matching') next.matching = design.id;
      else if (design.type === 'true_false' || design.type === 'tf') next.tf = design.id;
      else if (design.type === 'correct_incorrect') next.correctIncorrect = design.id;
      else if (design.type === 'vocabulary') next.vocabulary = design.id;
      else if (design.type === 'reading' || design.type === 'readingPassage') next.readingPassage = design.id;
      else if (design.type === 'circle') next.circle = design.id;
      else if (design.type === 'sentenceCompletion') next.sentenceCompletion = design.id;
      else if (design.type === 'wordBox') next.wordBox = design.id;
      return next;
    });
    if (design.style?.mcqStyle !== undefined) {
      setMcqStyle(design.style.mcqStyle);
    }
    alert(`Applied custom design: ${design.name}`);
  };

  const getStyleInstruction = (type: string, value: any, defaultLogic: string) => {
    if (typeof value === 'string' && value.startsWith('custom_')) {
      const design = customDesigns.find(d => d.id === value);
      if (design) {
        const customHtml = design.style.editableContent?.mainContent || design.style.mainContent || design.prompt || '';
        return `[CUSTOM DESIGN ENFORCED - ABSOLUTE MANDATORY]: You MUST use this EXACT HTML structure as a template for each item in the ${type} section:
\`\`\`html
${customHtml}
\`\`\`
[STRICT RULES]:
1. ELEMENT ORDER: You MUST maintain the EXACT order of elements (tables, text, blanks, etc.) as shown in the template.
2. TABLE STRUCTURE: If the template uses a table for questions or options, you MUST use a table.
3. PLACEHOLDER REPLACEMENT: Replace the sample text with REAL questions about {{TOPIC}}.
4. NO DEVIATION: Do NOT add extra spacing or change the layout. The user expects the generated test to look IDENTICAL to their design.`;
      }
    }
    return defaultLogic;
  };

  const handleGenerate = async () => {
    console.log("🚀 Starting Neural Synthesis...");
    
    const paperStylesInstruction = `
[PAPER STYLE ARCHITECT]:
[CRITICAL OVERRIDE]: The styles defined in this PAPER STYLE ARCHITECT section MUST OVERRIDE any conflicting formatting instructions found in the individual exercise prompts. The Paper Style is the absolute final authority on formatting. For example, if the exercise prompt asks for "___" but the Paper Style asks for "[ ]", you MUST use "[ ]". If the exercise prompt asks to write "C or I" but the Paper Style asks to choose "( C / I )", you MUST follow the Paper Style.
- MCQ Style: ${getStyleInstruction('mcq', paperStyles.mcq, paperStyles.mcq === 0 ? "Standard A, B, C, D with period." : 
                paperStyles.mcq === 1 ? "Underscore prefix before number (e.g., ___ 1. Question)." :
                paperStyles.mcq === 2 ? "Boxed letters [A] [B] [C] [D]." :
                paperStyles.mcq === 3 ? "Circled letters with random colors. MANDATORY: Wrap each letter (A, B, C, D) in a span with a random, distinct background color (e.g., <span style=\"background-color: #ff9999; color: black; border-radius: 50%; padding: 4px 8px; display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; font-weight: bold;\">A</span>). Use different colors for A, B, C, and D. Ensure text is clearly visible and centered." : 
                "Custom layout style " + (typeof paperStyles.mcq === 'number' ? paperStyles.mcq + 1 : paperStyles.mcq))}
- MCQ Spacing: ${mcqSpacing === 'none' ? "No extra vertical space between questions." : "Add exactly ONE empty line (Enter) between each question for better spacing."}
- True/False Style: ${getStyleInstruction('tf', paperStyles.tf, paperStyles.tf === 0 ? "Put ( T / F ) at the VERY END of each statement, right-aligned if possible." : 
                      paperStyles.tf === 1 ? "Put an underscore line before the number: ____ 1. Statement." :
                      paperStyles.tf === 2 ? "Put a checkbox before the number: [ ] 1. Statement." :
                      paperStyles.tf === 3 ? "Put 'True / False' labels on a new line under the statement." :
                      "Custom T/F style " + (typeof paperStyles.tf === 'number' ? paperStyles.tf + 1 : paperStyles.tf))}
- Correct/Incorrect Style: ${getStyleInstruction('correctIncorrect', paperStyles.correctIncorrect, paperStyles.correctIncorrect === 0 ? "Put a checkbox [ ] before the number (e.g., [ ] 1. Sentence). DO NOT use underscores." : 
                             paperStyles.correctIncorrect === 1 ? "Put ( C / I ) at the VERY END of each statement (e.g., 1. Sentence ( C / I ))." :
                             paperStyles.correctIncorrect === 2 ? "Put 'Correct / Incorrect' labels on a new line under the statement." :
                             "Custom C/I style " + (typeof paperStyles.correctIncorrect === 'number' ? paperStyles.correctIncorrect + 1 : paperStyles.correctIncorrect))}
- Vocabulary Style: ${getStyleInstruction('vocabulary', paperStyles.vocabulary, paperStyles.vocabulary === 0 ? "Classic Fill-in-the-blank (e.g., 1. Word   __: Definition). Use a 2-column borderless table to align the blanks and definitions." : 
                       paperStyles.vocabulary === 1 ? "Alternating Rows (Italicized words on left, definitions on right). Use a 2-column borderless table." :
                       paperStyles.vocabulary === 2 ? "Standard Alternating Rows (Words on left, definitions on right). Use a 2-column borderless table." :
                       "Bordered Table (Reversed: Create an HTML table with 2 columns. Column 1 = Definition, Column 2 = Word/Key Term).")}
- Vocabulary Indentation: For ALL standard lists (not bordered tables), you MUST use a 2-column borderless HTML table to align the definitions perfectly. Column 1: Number and Word. Column 2: Definition. Do NOT just use spaces.
- Vocabulary Introduction: Always include a clear introduction sentence before the vocabulary list (e.g., "PART A: Study the following vocabulary words and their corresponding definitions.").
- Global Indentation: For ALL question types, ensure the question numbers (1., 2., 3.) are perfectly aligned vertically. Use a table structure if necessary to ensure the text starts at the same horizontal position.
- Instruction Clarity: ALL "PART X: ..." headers and instruction sentences MUST have a light background color (e.g., background: #f1f5f9) and dark text (color: #1e293b) to ensure they are perfectly clear and visible regardless of the page background. Use dark text for light backgrounds.
- Circle Style: ${getStyleInstruction('circle', paperStyles.circle, paperStyles.circle === 0 ? "Standard bold text to circle." : 
                  paperStyles.circle === 1 ? "Underlined text to circle." :
                  paperStyles.circle === 2 ? "Italicized text to circle." :
                  "Custom Circle style " + (typeof paperStyles.circle === 'number' ? paperStyles.circle + 1 : paperStyles.circle))}
- Sentence Completion Style: ${getStyleInstruction('sentenceCompletion', paperStyles.sentenceCompletion, paperStyles.sentenceCompletion === 0 ? "Standard blank line at the end (e.g., 1. The cat is ____.)." : 
                                 paperStyles.sentenceCompletion === 1 ? "Blank line with base word in parentheses (e.g., 1. The cat is ____ (sleep).)." :
                                 "Custom Sentence Completion style " + (typeof paperStyles.sentenceCompletion === 'number' ? paperStyles.sentenceCompletion + 1 : paperStyles.sentenceCompletion))}
- Word Box Style: ${getStyleInstruction('wordBox', paperStyles.wordBox, paperStyles.wordBox === 0 ? "Standard comma-separated list in a box." : 
                   paperStyles.wordBox === 1 ? "Bulleted list in a box." :
                   "Custom Word Box style " + (typeof paperStyles.wordBox === 'number' ? paperStyles.wordBox + 1 : paperStyles.wordBox))}
- Reading Passage Style: ${getStyleInstruction('readingPassage', paperStyles.readingPassage, paperStyles.readingPassage === 0 ? "Standard single column text." : 
                           paperStyles.readingPassage === 1 ? "Two-column text layout." :
                           paperStyles.readingPassage === 2 ? "Text enclosed in a bordered box." :
                           "Custom Reading Passage style " + (typeof paperStyles.readingPassage === 'number' ? paperStyles.readingPassage + 1 : paperStyles.readingPassage))}
- Matching Style: ${getStyleInstruction('matching', paperStyles.matching, paperStyles.matching === 'classic' ? "Classic A/B columns." : "Custom Matching style " + paperStyles.matching)}
- Cloze Style: ${getStyleInstruction('cloze', paperStyles.cloze, "Standard cloze passage with blanks.")}
- Double MCQ Style: ${getStyleInstruction('doubleMcq', paperStyles.doubleMcq, "Standard double-gap MCQ with 4 options per item.")}
`;

    if (selectedInstructionIds.length === 0) { 
      console.warn("⚠️ No components selected.");
      alert("Please select at least one component (e.g., MCQ, True/False) from the list below."); 
      return; 
    }
    
    setIsGenerating(true);
    setGenerationError(null);
    setGenerationStep('Initializing Neural Core...');
    
    const selectedTemps = instructionTemplates.filter(t => selectedInstructionIds.includes(t.id));

    // Filter Master Protocols and Strict Rules by category
    const filterByCategory = (rules: StrictRule[]) => 
      rules.filter(r => r.active && (r.category === 'General' || r.category.toLowerCase() === activeModule.toLowerCase()));

    const filteredProtocols = filterByCategory(masterProtocols);
    const filteredRules = filterByCategory(strictRules);

    let pageStyleInstruction = '';
    if (enablePages) {
      const randomStyle = PAGE_STYLES[Math.floor(Math.random() * PAGE_STYLES.length)];
      pageStyleInstruction = `\n[PAGE STYLE - CRITICAL]: Wrap the ENTIRE assessment content in a single <div> with the following style: "${randomStyle.style}". This creates a unique beautiful page border/frame.`;
    }

    let partBackgroundInstruction = '';
    if (isPartBackgroundEnabled) {
      partBackgroundInstruction = `\n${PART_BACKGROUND_INSTRUCTION}`;
    }

    let instructionBackgroundInstruction = '';
    if (isInstructionBackgroundEnabled) {
      instructionBackgroundInstruction = `\n${INSTRUCTION_HEADER_BACKGROUND_INSTRUCTION}`;
    }

    // Randomize blank style between underscores and dots
    const blankStyles = ['____________________', '........................'];
    const selectedBlankStyle = blankStyles[Math.floor(Math.random() * blankStyles.length)];

    let currentSubject = SUBJECTS.find(s => s.id === activeSubject) || SUBJECTS[0];
    if (isRandomSubject) {
      currentSubject = SUBJECTS[Math.floor(Math.random() * SUBJECTS.length)];
    }
    const subjectInstruction = `\n[LOCALIZATION - CRITICAL]: Use names and places from the following lists to make the test culturally relevant. 
    NAMES: ${currentSubject.names.join(', ')}
    PLACES: ${currentSubject.places.join(', ')}
    Ensure these names and places are used naturally within the questions and reading texts.`;

    const caseInstruction = instructionCase === 'uppercase' 
      ? "CRITICAL: ALL instructions, introductions, and headers (e.g., 'PART A: Study the following...', 'Introduction: ...') MUST be in ALL CAPS (UPPERCASE). This is a strict requirement. Exercises and questions should use normal sentence case."
      : instructionCase === 'lowercase' 
      ? "CRITICAL: ALL instructions, introductions, and headers MUST follow 'Title Case' (only the first letter of each word is capitalized, e.g., 'Part A: Fill In The Blanks'). This is a strict requirement. Exercises and questions should use normal sentence case."
      : instructionCase === 'random'
      ? "CRITICAL: Randomly choose between ALL CAPS or Title Case for the instructions and introductions. This is a strict requirement. Exercises and questions should use normal sentence case."
      : "";

    const protocolsPrompt = filteredProtocols.map(p => `[PROTOCOL - ${p.priority}]: ${p.promptInjection.replace(/{{BLANK}}/g, selectedBlankStyle)}`).join('\n');
    const rulesPrompt = filteredRules.map(r => `[STRICT RULE - ${r.priority}]: ${r.promptInjection.replace(/{{BLANK}}/g, selectedBlankStyle)}`).join('\n');
    
    const strategyInstruction = answerStrategy === 'GENERAL_MIXED' 
      ? `[STRATEGY]: GENERAL-MIXED (Horizontal Logic). The context is {{TOPIC}}, but distractors should test high-frequency "general" errors (Gerunds, Prepositions, Agreement).`
      : `[STRATEGY]: TOPIC-FOCUSED (Vertical Logic). Every item and distractor must focus strictly on the rules of {{TOPIC}}.`;

    const generationIntegrityInstruction = `
[GENERATION INTEGRITY - CRITICAL]:
1. ALL SELECTED TYPES: You MUST generate content for EVERY SINGLE exercise type selected in the list below. Do NOT skip any.
2. UNIQUE READING PASSAGES: For Reading exercises, you MUST generate a COMPLETELY UNIQUE and DIFFERENT reading passage for EACH exercise type. 
   - Example: If "Reading MCQ" and "Reading T/F" are both selected, you MUST generate TWO different stories/texts. 
   - NEVER use the same text for multiple exercise types unless the user explicitly asks for "One Reading Text for All Parts".
3. ITEM COUNTS: Strictly follow the item count overrides if provided.
4. VARIETY: Ensure high variety in scenarios and sentence structures.
`;

    const rulerInstruction = `\n[RULER STYLE - CRITICAL]: After EVERY instruction header (e.g., PART A: ...), you MUST insert a <div class="instruction-ruler-5"></div>. This is a visual separator.`;

    const headerDesigns = [
      `<div style="border-bottom: 2pt solid black; padding-bottom: 10pt; margin-bottom: 20pt;">
        <div style="display: flex; align-items: center; gap: 15pt; margin-bottom: 15pt;">
          ${brandSettings.logoData ? `<img src="${brandSettings.logoData}" style="max-height: 60pt; width: auto;" />` : ''}
          <h1 style="text-align: ${brandSettings.logoData ? 'left' : 'center'}; flex: 1; margin: 0; font-size: 24pt; text-transform: uppercase;">${brandSettings.schoolName || 'Worksheet'}</h1>
        </div>
        <div style="display: flex; justify-content: space-between; font-weight: bold; font-size: 10pt; margin-bottom: 5pt;">
          <span>${brandSettings.studentLabel || 'Name'}: _________________________________</span>
          <span>${brandSettings.dateLabel || 'Date'}: ____ / ____ / ____</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-weight: bold; font-size: 10pt;">
          <span>${brandSettings.classLabel || 'Class'}: _________________________________</span>
          <span>${brandSettings.teacherLabel || 'Teacher'}: _________________________________</span>
        </div>
      </div>`,
      `<div style="border: 2pt solid black; padding: 15pt; margin-bottom: 20pt; text-align: center;">
        <h1 style="font-size: 22pt; margin-bottom: 5pt;">${brandSettings.schoolName || 'Worksheet'}</h1>
        <div style="border-top: 1pt solid black; padding-top: 10pt; display: grid; grid-template-columns: 1fr 1fr; gap: 10pt; text-align: left; font-size: 9pt;">
          <div>${brandSettings.studentLabel || 'NAME'}: __________________________</div>
          <div>${brandSettings.dateLabel || 'DATE'}: __________________________</div>
          <div>${brandSettings.classLabel || 'CLASS'}: _________________________</div>
          <div>${brandSettings.scoreLabel || 'SCORE'}: ________ / ________</div>
        </div>
      </div>`,
      `<div style="margin-bottom: 30pt;">
        <div style="font-size: 8pt; border-bottom: 1pt solid #ccc; padding-bottom: 5pt; margin-bottom: 10pt; display: flex; justify-content: space-between;">
          <span>${brandSettings.schoolName}</span>
          <span>Academic Year: 2025-2026</span>
        </div>
        <h1 style="font-size: 28pt; font-weight: 900; letter-spacing: -1pt; margin-bottom: 10pt;">${topic.toUpperCase()}</h1>
        <div style="background: #f1f5f9; padding: 10pt; border-radius: 4pt; display: flex; gap: 20pt; font-size: 9pt;">
          <span><b>${brandSettings.studentLabel || 'STUDENT'}:</b> ____________________</span>
          <span><b>${brandSettings.idLabel || 'ID'}:</b> __________</span>
          <span><b>${brandSettings.dateLabel || 'DATE'}:</b> __________</span>
        </div>
      </div>`,
      `<div style="border-left: 5pt solid #2563eb; padding-left: 15pt; margin-bottom: 25pt;">
        <h1 style="font-size: 20pt; color: #1e40af; margin-bottom: 5pt; font-weight: 800;">${brandSettings.schoolName || 'Worksheet'}</h1>
        <div style="font-size: 10pt; color: #64748b; margin-bottom: 10pt;">Topic: ${topic || 'General Assessment'}</div>
        <div style="display: flex; gap: 15pt; font-size: 9pt; border-top: 1pt dashed #cbd5e1; padding-top: 8pt;">
          <span>${brandSettings.studentLabel || 'Name'}: _________________</span>
          <span>${brandSettings.classLabel || 'Class'}: _________</span>
          <span>${brandSettings.dateLabel || 'Date'}: _________</span>
        </div>
      </div>`,
      `<div style="background: #1e293b; color: white; padding: 20pt; border-radius: 8pt; margin-bottom: 25pt; position: relative; overflow: hidden;">
        <div style="position: absolute; right: -20pt; top: -20pt; width: 100pt; height: 100pt; background: rgba(255,255,255,0.05); border-radius: 50%;"></div>
        <h1 style="font-size: 24pt; font-weight: 900; margin-bottom: 10pt; position: relative;">${brandSettings.schoolName || 'Worksheet'}</h1>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10pt; font-size: 9pt; opacity: 0.9;">
          <div style="border-bottom: 1pt solid rgba(255,255,255,0.3); padding-bottom: 2pt;">${brandSettings.studentLabel || 'Student'}: ________________</div>
          <div style="border-bottom: 1pt solid rgba(255,255,255,0.3); padding-bottom: 2pt;">${brandSettings.idLabel || 'ID'}: ________________</div>
          <div style="border-bottom: 1pt solid rgba(255,255,255,0.3); padding-bottom: 2pt;">${brandSettings.scoreLabel || 'Score'}: ________</div>
        </div>
      </div>`,
      `<div style="border: 4pt solid #16a34a; padding: 15pt; margin-bottom: 25pt; border-radius: 12pt; background: #f0fdf4;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2pt solid #16a34a; padding-bottom: 10pt; margin-bottom: 10pt;">
          <h1 style="font-size: 18pt; color: #166534; font-weight: 900; margin: 0;">${brandSettings.schoolName || 'Worksheet'}</h1>
          <div style="text-align: right; font-size: 8pt; color: #15803d;">
            <div>${brandSettings.dateLabel || 'DATE'}: ____/____/____</div>
            <div>${brandSettings.classLabel || 'CLASS'}: ___________</div>
          </div>
        </div>
        <div style="font-size: 10pt; color: #14532d; font-weight: bold;">${brandSettings.studentLabel || 'NAME'}: ______________________________________________________</div>
      </div>`,
      `<div style="background: linear-gradient(135deg, #065f46 0%, #064e3b 100%); color: white; padding: 25pt; margin-bottom: 30pt; border-radius: 4pt; box-shadow: 0 10pt 20pt rgba(6, 95, 70, 0.2);">
        <div style="border: 1pt solid rgba(255,255,255,0.2); padding: 15pt;">
          <h1 style="font-size: 22pt; font-weight: 200; letter-spacing: 5pt; text-align: center; margin-bottom: 15pt; text-transform: uppercase;">${brandSettings.schoolName || 'Assessment'}</h1>
          <div style="display: flex; justify-content: space-around; font-size: 9pt; font-family: monospace;">
            <span>[ ${brandSettings.studentLabel || 'STUDENT'}: ____________ ]</span>
            <span>[ ${brandSettings.dateLabel || 'DATE'}: __/__/__ ]</span>
            <span>[ ${brandSettings.scoreLabel || 'SCORE'}: ____ ]</span>
          </div>
        </div>
      </div>`,
      `<div style="border: 1pt solid #e2e8f0; padding: 0; margin-bottom: 25pt; border-radius: 8pt; overflow: hidden; box-shadow: 0 4pt 6pt -1pt rgba(0,0,0,0.1);">
        <div style="background: #facc15; padding: 15pt; display: flex; justify-content: space-between; align-items: center;">
          <h1 style="font-size: 16pt; font-weight: 900; color: #854d0e; margin: 0;">${brandSettings.schoolName || 'Worksheet'}</h1>
          <div style="background: white; padding: 4pt 12pt; border-radius: 20pt; font-size: 9pt; font-weight: bold; color: #854d0e;">${brandSettings.scoreLabel || 'SCORE'}: ____ / ____</div>
        </div>
        <div style="padding: 12pt; display: grid; grid-template-columns: 2fr 1fr; gap: 10pt; font-size: 9pt; background: white;">
          <div style="border-bottom: 1pt solid #e2e8f0;">${brandSettings.studentLabel || 'NAME'}: _________________________</div>
          <div style="border-bottom: 1pt solid #e2e8f0;">${brandSettings.dateLabel || 'DATE'}: ____________</div>
        </div>
      </div>`,
      `<div style="border-top: 8pt solid #dc2626; padding-top: 15pt; margin-bottom: 25pt;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
          <div>
            <h1 style="font-size: 26pt; font-weight: 900; color: #991b1b; line-height: 1;">${brandSettings.schoolName || 'TEST'}</h1>
            <div style="font-size: 10pt; color: #ef4444; font-weight: bold; margin-top: 5pt;">ACADEMIC EVALUATION</div>
          </div>
          <div style="text-align: right; font-family: serif; font-style: italic; font-size: 10pt;">
            <div>${brandSettings.studentLabel || 'Name'}: _________________</div>
            <div>${brandSettings.classLabel || 'Class'}: ________________</div>
            <div>${brandSettings.dateLabel || 'Date'}: _________________</div>
          </div>
        </div>
      </div>`,
      `<div style="display: flex; flex-direction: column; gap: 10pt; margin-bottom: 30pt;">
        <h1 style="font-size: 14pt; font-weight: 400; color: #64748b; border-left: 2pt solid #cbd5e1; padding-left: 10pt;">${brandSettings.schoolName || 'Worksheet'}</h1>
        <div style="font-size: 32pt; font-weight: 900; color: #0f172a; line-height: 1;">${topic.toUpperCase()}</div>
        <div style="height: 1pt; background: #e2e8f0; width: 100%;"></div>
        <div style="display: flex; gap: 30pt; font-size: 9pt; color: #94a3b8; font-weight: bold;">
          <span>${brandSettings.studentLabel || 'STUDENT'}: ____________________</span>
          <span>${brandSettings.dateLabel || 'DATE'}: ____________________</span>
        </div>
      </div>`,
      `<div style="display: flex; align-items: center; gap: 20pt; border-bottom: 3pt solid black; padding-bottom: 15pt; margin-bottom: 25pt;">
        ${brandSettings.logoData ? `<img src="${brandSettings.logoData}" style="max-height: 80pt; width: auto;" />` : '<div style="width: 80pt; height: 80pt; border: 2pt dashed #cbd5e1; display: flex; align-items: center; justify-content: center; color: #94a3b8; font-size: 8pt; text-align: center;">School Logo</div>'}
        <div style="flex: 1;">
          <h1 style="font-size: 24pt; font-weight: 900; margin: 0; text-transform: uppercase;">${brandSettings.schoolName || 'Worksheet'}</h1>
          <div style="font-size: 10pt; font-weight: bold; color: #64748b; margin-top: 5pt;">${brandSettings.schoolAddress}</div>
          <div style="display: flex; gap: 15pt; font-size: 9pt; margin-top: 10pt; border-top: 1pt solid #e2e8f0; padding-top: 5pt;">
            <span>${brandSettings.studentLabel || 'Name'}: _________________</span>
            <span>${brandSettings.classLabel || 'Class'}: _________</span>
            <span>${brandSettings.dateLabel || 'Date'}: _________</span>
          </div>
        </div>
      </div>`
    ];

    const selectedHeader = headerDesigns[paperDesign % headerDesigns.length];

    const mcqLayoutInstruction = `[MCQ LAYOUT - ABSOLUTE MANDATORY]: For Multiple Choice Questions, you MUST format the options (A, B, C, D) using an HTML <table> with the class "options-table". 
    THIS RULE OVERRIDES ANY OTHER MCQ FORMATTING RULE.
    ${mcqLayout === 'single' ? '- ONE LINE: Create a <table> with 1 row and 4 columns. Row 1: [A, B, C, D].' : 
      mcqLayout === 'double' ? '- TWO LINES: Create a <table> with 2 rows and 2 columns. Row 1: [A, B], Row 2: [C, D]. (This ensures A and C are vertically aligned in the first column).' : 
      '- FOUR LINES: Create a <table> with 4 rows and 1 column. Row 1: [A], Row 2: [B], Row 3: [C], Row 4: [D].'}
    ${mcqStyle > 0 ? '- [STYLE]: Wrap the option letters (A, B, C, D) in <b> tags (e.g., <td><b>A</b> Option text</td>). MANDATORY: DO NOT use brackets like (A) or [A]. Just the plain letter inside the <b> tag. These will be styled as professional badges.' : '- [STYLE]: Use plain letters (A, B, C, D) followed by a period.'}
    
    [MCQ DESIGN VARIETY]:
    - Randomly choose between these styles for the question line:
      1. Standard: "1. What is the capital of France?"
      2. Answer Blank at Start: "____ 1. What is the capital of France?" (Use 5 underscores for the blank).
      3. Letter on Line: "<u>&nbsp;&nbsp;A&nbsp;&nbsp;</u> 1. What is the capital of France?" (Where A is the correct letter, placed on a underlined blank before the number).
    
    [TEACHER ANSWER KEY - MANDATORY]: 
    - At the very end of the document, add a section wrapped in <div class="answer-key-section">.
    - Inside, add a <h2> titled "Teacher Answer Key".
    - Format the answers compactly: "Part A: 1:A, 2:C, 3:D..." 
    - If "MCQ Design" is active, wrap the letters in <b> tags here too (e.g., "1:<b>A</b>").
    - Ensure the answer key is clearly separated from the test content.
    
    [HEADER DESIGN]:
    - Use the following HTML for the top of the worksheet:
    ${selectedHeader}
    
    [FOOTER DESIGN]:
    - At the very bottom of the document (after the answer key), add a footer wrapped in <div class="worksheet-footer">.
    - Use the following text for the footer: "${brandSettings.footerText}"
    - Style it with small font size (8pt), centered, and slightly faded color.
    
    [EXERCISE DESIGN - CREATIVE]: 
    - For "Writing Definition" or "Supply Key Terms", use professional HTML tables with <thead> and <tbody>.
    - Use zebra striping for tables.
    - For "Matching", randomly choose between these styles:
      1. Word Bank at Top: List words in a box at the top, then definitions below with a blank line on the left (e.g., "__________ 1. A large animal with a trunk").
      2. Word Bank at Bottom: Definitions on the right, blank lines on the left (e.g., "__________ 1. ..."), and a word bank box at the very bottom of the section.
      3. Two-Column Match: Column 1 (1-10) with terms, Column 2 (A-G) with definitions.
      4. Letter Matching: "____ 1. Definition..." with a list of options (A, B, C...) below or in a side box.
    - For "Fill in the blank", vary the blank position: sometimes at the beginning ("__________ is the capital of France"), middle, or end. Use <span class="blank-line"></span> for blanks.
    - For "True or False", you MUST use this style:
        ${paperStyles.tf === 0 ? 'a. ( T / F ) at the end: "1. The sky is blue. ( T / F )"' :
          paperStyles.tf === 1 ? 'b. Blank at start: "<span class="blank-line"></span> 1. The sky is blue."' :
          paperStyles.tf === 2 ? 'c. Checkbox at start: "<span class="checkbox-box"></span> 1. The sky is blue."' :
          'd. True / False labels at the end.'}
    - For "Correct or Incorrect", you MUST use this style:
        ${paperStyles.correctIncorrect === 0 ? 'a. Checkbox at start: "<span class="checkbox-box"></span> 1. Sentence with error."' :
          paperStyles.correctIncorrect === 1 ? 'b. ( C / I ) at the end: "1. Sentence with error. ( C / I )"' :
          'c. "Correct / Incorrect" labels at the end.'}
    - For "Sentence Rewrite", provide the original sentence, then a blank line below it for the student to write the new version.
    - Make the designs look interesting and varied, like a high-quality printed textbook.
    - Ensure all tables have the class "professional-table" for consistent styling.`;

    const alignments = ['left', 'center', 'right'];
    const randomAlignment = alignments[Math.floor(Math.random() * alignments.length)];
    
    const headerStyle = `class="header-row", background-color: #334155, color: white, text-align: left, padding-left: 15pt, font-weight: bold`;
    
    const componentLogic = selectedTemps.map((t, idx) => {
      const overrideCol = columnOverrides[t.id] !== undefined ? columnOverrides[t.id] : (t.columnCount !== undefined ? t.columnCount : defaultColumnCount);
      const overrideItems = itemCountOverrides[t.id] || 10;
      
      let blueprintStr = '';
      
      // Determine the type of question to generate the correct blueprint
      const isTF = t.id.includes('tf') || t.label.includes('True/False');
      const isMCQ = t.id.includes('mcq') || t.label.includes('MCQ');
      const isOpenEnded = t.id.includes('short_answer') || t.id.includes('inferential') || t.id.includes('critical_thinking') || t.id.includes('rewrite') || t.id.includes('speaking');

      if (isTF) {
        // Generate T/F/NG blueprint
        const tfKeys = ['T', 'F', 'NG'];
        const blueprint: string[] = [];
        for (let i = 0; i < overrideItems; i++) {
          blueprint.push(tfKeys[Math.floor(Math.random() * tfKeys.length)]);
        }
        blueprintStr = `(USE THIS ANSWER KEY: ${blueprint.map((key, i) => `${i + 1}:${key}`).join(', ')})`;
      } else if (isMCQ || (!isOpenEnded && !isTF)) {
        // Generate standard A, B, C, D blueprint for MCQs and other closed-ended questions
        const blueprint = generateNeuralBlueprint(overrideItems);
        blueprintStr = `(USE THIS ANSWER KEY: ${blueprint.map((key, i) => `${i + 1}:${key}`).join(', ')})`;
      } else {
        // Open-ended questions do not get a forced blueprint
        blueprintStr = `(DO NOT USE A PRE-ASSIGNED ANSWER KEY. Generate natural, accurate answers for the Teacher Answer Key based on the text.)`;
      }

      let formatInstruction = '';
      
      const effectiveCols = overrideCol > 0 ? overrideCol : 1;
      const effectiveTableStyle = overrideCol === 0 ? 'list' : tableStyle;

      if (effectiveTableStyle === 'plain') {
        if (effectiveCols > 1) {
          formatInstruction = `(MANDATORY FORMAT: Use a real HTML <table> with ${effectiveCols} columns. 
            ${isPartBackgroundEnabled ? 'MANDATORY: Apply a unique background style class from the PART BACKGROUND PROTOCOL to this <table> tag.' : ''}
            - Row 1: Header row spanning all ${effectiveCols} columns (colspan="${effectiveCols}"), with ${headerStyle}. Title: "PART ${String.fromCharCode(65 + idx)}: ${t.professionalLabel || t.label}".
            - Row 2: Distribute the ${overrideItems} items STRICTLY EVENLY across ${effectiveCols} columns. (e.g. if 10 items, put 5 in Col 1 and 5 in Col 2).
            - MANDATORY: Every numbered item (1., 2., 3., etc.) MUST start on a NEW LINE using an HTML <p> or <br> tag. DO NOT bunch them together.
            - The table MUST have an outer border: 1.5pt solid #334155.
            - DO NOT put borders between the items inside the cells. This is the "Plain" layout with ${effectiveCols} columns.)`;
        } else {
          formatInstruction = `(MANDATORY FORMAT: Use a real HTML <table> with 1 column and EXACTLY 2 rows. 
            ${isPartBackgroundEnabled ? 'MANDATORY: Apply a unique background style class from the PART BACKGROUND PROTOCOL to this <table> tag.' : ''}
            - Row 1: Header row with ${headerStyle}. Title: "PART ${String.fromCharCode(65 + idx)}: ${t.professionalLabel || t.label}".
            - Row 2: A single <td> containing ALL ${overrideItems} items. MANDATORY: Every numbered item (1., 2., 3., etc.) MUST start on a NEW LINE using an HTML <p> or <br> tag. DO NOT bunch them together in a single paragraph.
            - The table MUST have an outer border: 1.5pt solid #334155.
            - DO NOT put borders between the items inside the second row. This is the "Plain" layout.)`;
        }
      } else if (effectiveTableStyle === 'grid') {
        formatInstruction = `(MANDATORY FORMAT: Use a real HTML <table> with ${effectiveCols} columns. 
            ${isPartBackgroundEnabled ? 'MANDATORY: Apply a unique background style class from the PART BACKGROUND PROTOCOL to this <table> tag.' : ''}
            - Row 1: Header row spanning all ${effectiveCols} columns (colspan="${effectiveCols}"), with ${headerStyle}. Title: "PART ${String.fromCharCode(65 + idx)}: ${t.professionalLabel || t.label}".
            - Subsequent rows: Distribute the ${overrideItems} items STRICTLY EVENLY across ${effectiveCols} columns (one item per cell).
            - Every <td> MUST have a border: 1pt solid #334155; padding: 10px; vertical-align: top;
            - This creates a professional worksheet grid with ${effectiveCols} columns.)`;
      } else if (effectiveTableStyle === 'list') {
        if (effectiveCols > 1) {
          formatInstruction = `(MANDATORY FORMAT: Use a real HTML <table> with ${effectiveCols} columns. 
            ${isPartBackgroundEnabled ? 'MANDATORY: Apply a unique background style class from the PART BACKGROUND PROTOCOL to this <table> tag.' : ''}
            - Row 1: Header row spanning all ${effectiveCols} columns (colspan="${effectiveCols}"), with ${headerStyle}. Title: "PART ${String.fromCharCode(65 + idx)}: ${t.professionalLabel || t.label}".
            - Subsequent rows: Distribute the ${overrideItems} items STRICTLY EVENLY across ${effectiveCols} columns.
            - Every <td> MUST have a bottom border: 1pt solid #334155; padding: 10px;
            - This creates a lined list with ${effectiveCols} columns.)`;
        } else {
          formatInstruction = `(MANDATORY FORMAT: Use a real HTML <table> with 1 column. 
            ${isPartBackgroundEnabled ? 'MANDATORY: Apply a unique background style class from the PART BACKGROUND PROTOCOL to this <table> tag.' : ''}
            - Row 1: Header row with ${headerStyle}. Title: "PART ${String.fromCharCode(65 + idx)}: ${t.professionalLabel || t.label}".
            - Subsequent rows: Each row contains EXACTLY ONE item.
            - Every <td> MUST have a bottom border: 1pt solid #334155; padding: 10px;
            - This creates horizontal lines between every question.)`;
        }
      } else {
        const customTable = customDesigns.find(d => d.id === effectiveTableStyle);
        if (customTable && customTable.prompt) {
          formatInstruction = `(MANDATORY FORMAT: ${customTable.prompt}
            ${isPartBackgroundEnabled ? 'MANDATORY: Apply a unique background style class from the PART BACKGROUND PROTOCOL to this <table> tag.' : ''}
            - Row 1: Header row spanning all columns, with ${headerStyle}. Title: "PART ${String.fromCharCode(65 + idx)}: ${t.professionalLabel || t.label}".
            - Distribute the ${overrideItems} items according to the custom table style.)`;
        }
      }
        
      const rawHeader = t.professionalLabel || t.label;
      const formattedHeader = instructionCase === 'uppercase' ? rawHeader.toUpperCase() : instructionCase === 'lowercase' ? toTitleCase(rawHeader) : rawHeader;
        
      return `PART ${String.fromCharCode(65 + idx)} [MANDATORY INSTRUCTION HEADER - YOU MUST USE THIS EXACT TEXT: "${formattedHeader}"]: ${t.prompt.replace(/{{BLANK}}/g, selectedBlankStyle)} (GENERATE EXACTLY ${overrideItems} ITEMS) ${blueprintStr} ${formatInstruction} ${paperStylesInstruction}`;
    }).join('\n\n');

    const moduleSafetyGuard = activeModule === 'Grammar'
      ? `[MODULE SAFETY GUARD - CRITICAL]: You are generating a GRAMMAR assessment. You are strictly FORBIDDEN from including reading passages or vocabulary-only definitions. Focus 100% on grammar rules, situational logic, and positional word order. Ensure NO LEAKAGE from Reading or Vocabulary modules.`
      : activeModule === 'Vocabulary'
      ? `[MODULE SAFETY GUARD - CRITICAL]: You are generating a VOCABULARY assessment. You are strictly FORBIDDEN from testing grammar rules, injecting grammar errors, or including reading passages. 
         - NO READING LOGIC: Do NOT include "Not Mentioned" or "Unknown" options. 
         - NO GRAMMAR LOGIC: Protocol 21 (Cross-Topic Injection) and Rule 1 (No-Free-Verb) are DISABLED. 
         - NO GRAMMAR TOPICS: Avoid using sentences that test "Must/Have to", "Should", or other modal verbs. Focus on the meaning of the word itself.
         - PURE SEMANTICS: Focus 100% on word meanings. All distractors must be grammatically identical to the correct answer.`
      : activeModule === 'Reading'
      ? `[MODULE SAFETY GUARD - CRITICAL]: You are generating a READING assessment. You are strictly FORBIDDEN from testing grammar rules or injecting grammar errors. Focus 100% on comprehension and inference logic.
         - PASSAGE DIVERSITY: ${isSingleReadingText ? 'Use ONE SINGLE reading passage for the ENTIRE test. All parts must refer to this single passage.' : 'Use a DIFFERENT reading passage for EACH part of the test.'}
         - LEVEL ADAPTATION: The length and level of thinking must strictly match the selected Academic Level (${activeLevel}).`
      : '';

    const readingPassageLength = (activeLevel === 'Kid' || activeLevel === 'Beginner') ? '50-80 words' : '300-500 words';
    const readingPassageInstruction = isSingleReadingText 
      ? `1. GENERATE ONE SINGLE PASSAGE (~${readingPassageLength}) about "${topic}" at the top of the test.` 
      : `1. GENERATE A UNIQUE, SEPARATE PASSAGE (~${readingPassageLength}) FOR EVERY SINGLE PART of the test. Each part MUST have its own distinct text.`;

    const mandatorySequence = activeModule === 'Grammar' 
      ? `1. GENERATE ALL ${selectedInstructionIds.length} REQUESTED PARTS. ADAPT TITLES TO MATCH "${topic}".\n2. ENFORCE "NO FREE VERB" & "SITUATIONAL EVIDENCE" rules for all grammar stems. MANDATORY: Every question MUST have a full context sentence. DO NOT generate just blanks.\n3. [SOURCE PRIORITY]: If source material is provided, strictly use ALL grammar rules and examples from it. If there are 6 rules, use all 6.\n4. [VARIETY]: If no specific topic is provided, generate a variety of grammar topics (e.g., Tenses, Conditionals, Relative Clauses, Passive Voice, etc.). DO NOT default to just one topic.\n5. [COMPLETENESS]: You are FORBIDDEN from stopping early. You MUST generate ALL parts requested.`
      : activeModule === 'Reading'
      ? `${readingPassageInstruction}\n2. APPLY [NATURAL PARAPHRASE] logic to all questions (No keyword matching).\n3. ENFORCE [READING LOGIC FIREWALL] (Strictly forbidden from testing grammar).\n4. ENSURE all distractors are grammatically identical to the correct answer.\n5. [UNIQUE PASSAGES]: Each exercise type MUST have its own unique reading passage unless "Single Reading Text" is active.\n6. [COMPLETENESS]: You MUST generate ALL ${selectedInstructionIds.length} requested parts.`
      : `1. GENERATE ALL ${selectedInstructionIds.length} REQUESTED PARTS. ADAPT TITLES TO MATCH "${topic}".\n2. ENFORCE [VOCABULARY FIREWALL] (No grammar clues).\n3. [COMPLETENESS]: You MUST generate ALL parts requested.`;

    const instructionRulerPrompt = instructionRulerStyle > 0 
      ? `[INSTRUCTION RULER - MANDATORY]: After EVERY instruction header (e.g., PART A: ...), you MUST insert a <div class="instruction-ruler-${instructionRulerStyle}"></div>. This is a visual separator that MUST be visible.
         - S1: Simple 1pt solid line.
         - S2: 2pt dashed line.
         - S3: 3pt double line.
         - S4: Thick 4pt solid line with gradient.
         - S5: Decorative line with stars (★ ★ ★).
         - S6: Decorative line with hearts (♥ ♥ ♥).`
      : '';

    const finalLogic = `
${moduleSafetyGuard}
${generationIntegrityInstruction}
${subjectInstruction}
${caseInstruction}
${GLOBAL_STRICT_COMMAND.replace(/{{TOPIC}}/g, topic || "General English").replace(/{{BLANK}}/g, selectedBlankStyle)}
${isFrameEnabled ? BORDER_FRAME_INSTRUCTION : ''}
${pageStyleInstruction}
${partBackgroundInstruction}
${instructionBackgroundInstruction}
${instructionRulerPrompt}
${protocolsPrompt}
${strategyInstruction.replace(/{{TOPIC}}/g, topic || "General English")}
${mcqLayoutInstruction}
${paperStylesInstruction}
${rulesPrompt}

[SYSTEM OBJECTIVE]: Generate a COMPLETE assessment based on the requested components.
[MANDATORY]: You MUST generate ALL ${selectedInstructionIds.length} requested parts. DO NOT skip any parts. If you hit a length limit, prioritize completing all parts with fewer items rather than skipping entire parts.
[TARGET TOPIC]: "${topic || "General English"}"
[TARGET LEVEL]: ${activeLevel}
[LANGUAGE]: ${activeLanguage}

### MANDATORY SEQUENCE ###
${mandatorySequence}

${componentLogic}
    `;
    
    try {
      setGenerationStep('Applying Master Protocols...');
      // Randomize logo from available logos
      const availableLogos = (brandSettings.logos || []).filter(l => !!l);
      if (availableLogos.length > 0) {
        const randomLogo = availableLogos[Math.floor(Math.random() * availableLogos.length)];
        setBrandSettings(prev => ({ ...prev, logoData: randomLogo }));
      }

      // Randomize Font if enabled
      if (brandSettings.randomizeFont) {
        const randomFont = FONTS[Math.floor(Math.random() * FONTS.length)];
        setBrandSettings(prev => ({ ...prev, activeFont: randomFont.name }));
      }

      setGenerationStep('Synthesizing Test Items...');
      // FIREBASE CLOUD SAVE IMPLEMENTATION
      // ==================================================
      // 1. Call the AI Brain
      const result = await callNeuralEngine(activeEngine, finalLogic, protocolsPrompt, sourceMaterial, externalKeys);
      
      if (result.text.includes('Error:')) {
        setGenerationError(result.text);
        setIsGenerating(false);
        setGenerationStep('');
        return;
      }

      setGenerationStep('Finalizing Layout...');
      setWorksheetContent(result.text);
      setIsGenerating(false);
      setGenerationStep('');
      setGenerationError(null);
      setViewMode('preview');

      // 2. Create the data package
      const newTestItem = {
        id: `hist-${Date.now()}`,
        title: `${activeLanguage} ${activeModule}: ${activeLevel} - ${topic || "Synthesis"}`,
        content: result.text,
        timestamp: Date.now(),
        promptId: 'manual',
        logicSnapshot: finalLogic,
        module: activeModule,
        level: activeLevel,
        topic: topic,
        // Add who created it
        authorName: auth.currentUser?.displayName || session?.name || 'Anonymous',
        authorEmail: auth.currentUser?.email || session?.email || 'N/A',
        uid: auth.currentUser?.uid || 'anonymous'
      };

      // 3. Update Local History (so you see it on screen)
      setHistory(prev => {
        const current = Array.isArray(prev) ? prev : [];
        return [newTestItem, ...current].slice(0, 30);
      });

      // 4. SEND TO THE CLOUD (The Magic Step!)
      if (auth.currentUser) {
        try {
             await setDoc(doc(db, 'history', newTestItem.id), newTestItem);
             console.log("✅☁️ Test successfully saved to the Firebase Cloud Notebook!");
        } catch (e) {
             handleFirestoreError(e, OperationType.WRITE, `history/${newTestItem.id}`);
        }
      }
    } catch (error: any) {
      console.error("Generation failed:", error);
      setGenerationError(error.message || "Neural synthesis failed. Please check your connection or API keys.");
      setIsGenerating(false);
      setGenerationStep('');
    }
  };

  const handleAssistantMessage = async (msg: string, file?: QuickSource) => {
    const userMsg: ChatMessage = { id: `msg-${Date.now()}`, role: 'user', text: msg, timestamp: Date.now() };
    setChatMessages(prev => [...prev, userMsg]);
    setIsGenerating(true);
    setGenerationStep('Assistant Processing...');
    try {
      const context = `Assistant Mode. Worksheet: ${worksheetContent.slice(0, 1000)}. Edit based on: ${msg}`;
      const result = await callNeuralEngine(activeEngine, msg, context, file || sourceMaterial, externalKeys);
      
      if (result.text.includes('Error:')) {
        setGenerationError(result.text);
        setIsGenerating(false);
        setGenerationStep('');
        return;
      }

      setChatMessages(prev => [...prev, { id: `msg-bot-${Date.now()}`, role: 'architect', text: "Synthesis updated.", timestamp: Date.now() }]);
      setWorksheetContent(result.text);
      setIsGenerating(false);
      setGenerationStep('');
    } catch (error: any) {
      console.error("Assistant failed:", error);
      setGenerationError(error.message || "Assistant synthesis failed.");
      setIsGenerating(false);
      setGenerationStep('');
    }
  };

  const handlePrint = () => {
    // Ensure the window is focused before printing
    window.focus();
    window.print();
  };

  const [previewZoom, setPreviewZoom] = useState(100);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const handleExportWord = () => {
    if (!worksheetContent) return;
    
    // Create a cleaner filename from the topic
    const cleanTopic = (topic || 'Assessment').trim().replace(/[^a-z0-9]/gi, '_');
    const timestamp = new Date().toISOString().split('T')[0];
    
    setExportSettings(prev => ({
      ...prev,
      filename: `DPSS_${activeLanguage}_${activeLevel}_${cleanTopic}_${timestamp}`,
      title: `${activeModule} Assessment: ${topic || 'General'}`,
      showModal: true
    }));
  };

  const confirmExportWord = () => {
    const { filename, title } = exportSettings;
    const logoHtml = brandSettings.logoData ? `<table style="width: 100%; border: none; margin-bottom: 2pt;"><tr><td style="border: none; text-align: center;"><img src="${brandSettings.logoData}" width="624" style="width: 6.5in;" /></td></tr></table>` : '';
    const activeFontObj = FONTS.find(f => f.name === brandSettings.activeFont);
    const activeFontFamily = activeFontObj ? activeFontObj.family : "'Times New Roman', serif";
    const headerText = brandSettings.customHeaderText || `${activeLevel}: ${activeModule}: ${topic || 'Assessment'}`;
    const headerRuler = (brandSettings.headerRulerStyle || 0) > 0 ? `<div class="instruction-ruler-${brandSettings.headerRulerStyle}" style="margin-top: 4pt; margin-bottom: 4pt;"></div>` : '';
    const header = `${logoHtml}<table style="width: 100%; border-bottom: ${(brandSettings.headerRulerStyle || 0) > 0 ? 'none' : '2pt solid black'}; margin-bottom: 2pt; font-family: ${activeFontFamily};"><tr><td style="border: none; width: 100%; text-align: center;"><b>${headerText}</b></td></tr></table>${headerRuler}`;
    
    // Use the headerHtml argument correctly
    exportToWord(
      worksheetContent, 
      filename || `DPSS_Test_${activeLanguage}_${activeLevel}`,
      header,
      '0.4in 0.6in 0.4in 0.6in',
      brandSettings.activeFont || 'Times New Roman',
      '1.15',
      undefined,
      isFrameEnabled,
      PAPER_DESIGNS[paperDesign],
      paperStyles,
      mcqStyle,
      globalLayout,
      baseLayout,
      instructionRulerStyle,
      instructionHeaderStyle
    );
    
    setExportSettings(prev => ({ ...prev, showModal: false }));
  };

  const updateRule = (id: string, updates: Partial<StrictRule>) => setStrictRules(prev => prev.map(r => r.id === id ? { ...r, ...updates, isCustomized: true } : r));
  const updateProtocol = (id: string, updates: Partial<StrictRule>) => setMasterProtocols(prev => prev.map(p => p.id === id ? { ...p, ...updates, isCustomized: true } : p));
  const updateTemplate = (id: string, updates: Partial<InstructionTemplate>) => setInstructionTemplates(prev => prev.map(t => t.id === id ? { ...t, ...updates, isCustomized: true } : t));
  const deleteTemplate = (id: string) => setInstructionTemplates(prev => prev.filter(t => t.id !== id));
  const deleteRule = (id: string) => setStrictRules(prev => prev.filter(r => r.id !== id));
  const deleteProtocol = (id: string) => setMasterProtocols(prev => prev.filter(p => p.id !== id));
  
  const handleModuleChange = (m: string) => {
    setActiveModule(m);
    setActiveModuleTitle(m);
    setShowRenameView(true);
  };

  const handleRenameConfirm = () => {
    setShowRenameView(false);
  };

  const syncWithDefaults = () => {
    try {
      setMasterProtocols(prev => {
        const updated = prev.map(p => {
          if (p.isCustomized) return p;
          const defaultProtocol = DEFAULT_MASTER_PROTOCOLS.find(dp => dp.id === p.id);
          return defaultProtocol ? { ...p, ...defaultProtocol } : p;
        });
        const existingIds = new Set(prev.map(p => p.id));
        const newItems = DEFAULT_MASTER_PROTOCOLS.filter(p => !existingIds.has(p.id));
        return [...updated, ...newItems];
      });
      setStrictRules(prev => {
        const updated = prev.map(r => {
          if (r.isCustomized) return r;
          const defaultRule = DEFAULT_STRICT_RULES.find(dr => dr.id === r.id);
          return defaultRule ? { ...r, ...defaultRule } : r;
        });
        const existingIds = new Set(prev.map(r => r.id));
        const newItems = DEFAULT_STRICT_RULES.filter(r => !existingIds.has(r.id));
        return [...updated, ...newItems];
      });
      setInstructionTemplates(prev => {
        const updated = prev.map(t => {
          if (t.isCustomized) return t;
          const defaultTemp = INITIAL_TEMPLATES.find(dt => dt.id === t.id);
          return defaultTemp ? { ...t, ...defaultTemp } : t;
        });
        const existingIds = new Set(prev.map(t => t.id));
        const newItems = INITIAL_TEMPLATES.filter(t => !existingIds.has(t.id));
        return [...updated, ...newItems];
      });
      alert("Neural protocols and templates synchronized with latest definitions. Custom edits were preserved.");
    } catch (error) {
      console.error("Sync failed:", error);
      alert("Neural Circuit Interrupted during sync. Please try again.");
    }
  };

  const hardReset = () => {
    if (confirm("WARNING: This will delete all custom rules, protocols, and templates. Are you sure?")) {
      localStorage.removeItem(MASTER_PROTOCOLS_KEY);
      localStorage.removeItem(STRICT_RULES_KEY);
      localStorage.removeItem(TEMPLATES_KEY);
      window.location.reload();
    }
  };

  const addRule = () => {
    const newRule: StrictRule = { id: `rule-${Date.now()}`, label: 'NEW LOGIC NODE', description: '', promptInjection: '', active: true, priority: 'Medium', category: activeLogicCategory };
    setStrictRules([...strictRules, newRule]); setExpandedRuleId(newRule.id);
  };
  const addProtocol = () => {
    const newProtocol: StrictRule = { id: `mp-${Date.now()}`, label: 'NEW PROTOCOL', description: '', promptInjection: '', active: true, priority: 'Medium', category: activeProtocolCategory };
    setMasterProtocols([...masterProtocols, newProtocol]); setExpandedProtocolId(newProtocol.id);
  };
  const addTemplate = () => {
    const newId = `temp-${Date.now()}`;
    setInstructionTemplates(prev => [...prev, { id: newId, label: `NEW PART`, prompt: `Detail logic for {{TOPIC}}...`, category: activeTemplateCategory as any, columnCount: 0 }]);
    setExpandedTemplateId(newId);
  };

  const handleFirestoreError = (error: any, operationType: 'create' | 'update' | 'delete' | 'list' | 'get' | 'write', path: string | null) => {
    const errInfo = {
      error: error instanceof Error ? error.message : String(error),
      authInfo: {
        userId: auth.currentUser?.uid,
        email: auth.currentUser?.email,
        emailVerified: auth.currentUser?.emailVerified,
        isAnonymous: auth.currentUser?.isAnonymous,
        tenantId: auth.currentUser?.tenantId,
        providerInfo: auth.currentUser?.providerData.map(provider => ({
          providerId: provider.providerId,
          displayName: provider.displayName,
          email: provider.email,
          photoUrl: provider.photoURL
        })) || []
      },
      operationType,
      path
    };
    console.error('Firestore Error: ', JSON.stringify(errInfo));
    // We don't necessarily want to throw and crash the app, but we want the agent to see it
  };

  return (
    <ErrorBoundary>
      <div className="flex h-screen overflow-hidden text-slate-300 relative transition-all duration-500 bg-slate-900">
        {firebaseError && (
          <div className="fixed top-0 left-0 right-0 z-[1000] bg-rose-600 text-white px-4 py-2 text-center text-xs font-bold animate-in slide-in-from-top duration-500">
            <i className="fa-solid fa-circle-exclamation mr-2"></i>
            {firebaseError}
            <button onClick={() => setFirebaseError(null)} className="ml-4 underline">Dismiss</button>
          </div>
        )}
        {showRenameView && (
          <div className="fixed inset-0 z-[300] bg-slate-950/90 backdrop-blur-xl flex items-center justify-center p-6">
            <div className="bg-white rounded-[48px] shadow-2xl w-full max-w-md overflow-hidden animate-in zoom-in-95 duration-300">
              <div className="p-10 space-y-8">
                <div className="flex items-center gap-4">
                  <div className="h-12 w-12 bg-orange-100 text-orange-600 rounded-2xl flex items-center justify-center">
                    <i className="fa-solid fa-pen-to-square text-xl"></i>
                  </div>
                  <div>
                    <h3 className="text-lg font-black uppercase tracking-tight text-slate-900">Configure Module</h3>
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Set your custom title for {activeModule}</p>
                  </div>
                </div>

                <div className="space-y-4">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Display Name</label>
                  <input 
                    autoFocus
                    value={activeModuleTitle}
                    onChange={e => setActiveModuleTitle(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleRenameConfirm()}
                    className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-orange-500 font-bold text-slate-700"
                    placeholder="e.g. Mid-Term Grammar Review"
                  />
                </div>

                <div className="flex gap-4 pt-4">
                  <button 
                    onClick={() => setShowRenameView(false)}
                    className="flex-1 py-5 bg-slate-100 text-slate-500 rounded-3xl text-[11px] font-black uppercase tracking-widest hover:bg-slate-200 transition-all"
                  >
                    Cancel
                  </button>
                  <button 
                    onClick={handleRenameConfirm}
                    className="flex-1 py-5 bg-orange-600 text-white rounded-3xl text-[11px] font-black uppercase tracking-widest hover:brightness-110 shadow-xl shadow-orange-600/20 transition-all"
                  >
                    Confirm
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
        {showOnboarding && <OnboardingTutorial onComplete={handleOnboardingComplete} />}
      {viewMode === 'generator' && (
        <>
          <Sidebar 
            isOpen={isSidebarOpen}
            onClose={() => setIsSidebarOpen(false)}
            curriculum={INITIAL_MODULES}
            activeModule={activeModule}
            onModuleChange={handleModuleChange}
            activeLevel={activeLevel}
            onLevelChange={setActiveLevel}
            topic={topic}
            onTopicChange={setTopic}
            onClearCanvas={() => { setWorksheetContent(''); setTopic(''); setSelectedInstructionIds([]); }}
            onToggleSettings={(tab) => { if(tab) setSettingsTab(tab as SettingsTab); setShowSettings(true); }}
            history={history}
            onLoadHistory={(item) => { setWorksheetContent(item.content); setViewMode('preview'); }}
            onDeleteHistory={async (id) => {
              try {
                const newHistory = history.filter(h => h.id !== id);
                setHistory(newHistory);
                localStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));
                if (session?.email) {
                  const docRef = doc(db, 'user_history', session.email);
                  await setDoc(docRef, { history: newHistory });
                }
              } catch (e) { console.error(e); }
            }}
            onRenameHistory={async (id, newTitle) => {
              try {
                const newHistory = history.map(h => h.id === id ? { ...h, title: newTitle } : h);
                setHistory(newHistory);
                localStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));
                if (session?.email) {
                  const docRef = doc(db, 'user_history', session.email);
                  await setDoc(docRef, { history: newHistory });
                }
              } catch (e) { console.error(e); }
            }}
            brandSettings={brandSettings}
            templates={instructionTemplates.filter(t => t.category?.toUpperCase() === activeModule.toUpperCase() || t.category?.toUpperCase() === 'ALL')}
            activeTemplate={null}
            onTemplateSelect={(t) => toggleInstruction(t.id)}
            isSingleReadingText={isSingleReadingText}
            onSingleReadingTextChange={setIsSingleReadingText}
            isRelaxingBackgroundEnabled={isColorfulBackgroundEnabled}
            onRelaxingBackgroundChange={setIsColorfulBackgroundEnabled}
            isPartBackgroundEnabled={isPartBackgroundEnabled}
            onPartBackgroundChange={setIsPartBackgroundEnabled}
            isInstructionBackgroundEnabled={isInstructionBackgroundEnabled}
            onInstructionBackgroundChange={setIsInstructionBackgroundEnabled}
            onRandomizeBackground={randomizeBackground}
            paperDesign={paperDesign}
            onPaperDesignChange={setPaperDesign}
            onDesignPaperClick={() => setViewMode('design_test_style')}
            onPaperStyleClick={() => setViewMode('paper_style_design')}
            mcqLayout={mcqLayout}
            onMCQLayoutChange={setMcqLayout}
            onInstructionDesignClick={() => setViewMode('instruction_design')}
            onHeaderFooterDesignClick={() => setViewMode('header_footer_design')}
            onFormatDesignClick={() => { setSettingsTab('FORMAT_DESIGN'); setShowSettings(true); }}
            onSubjectsClick={() => setShowSubjectModal(true)}
            instructionCase={instructionCase}
            onInstructionCaseChange={setInstructionCase}
            width={sidebarWidth}
            onWidthChange={setSidebarWidth}
            side={sidebarSide}
            onSideChange={setSidebarSide}
            user={auth.currentUser}
            onLogin={handleGoogleLogin}
            onLogout={handleLogout}
          />

          <main 
            style={{ 
              marginLeft: isSidebarOpen && sidebarSide === 'left' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px',
              marginRight: isSidebarOpen && sidebarSide === 'right' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px'
            }}
            className="flex-1 flex flex-col overflow-hidden transition-all duration-500 relative"
          >
            {isRelaxingBackgroundEnabled && (
               <div 
                 className="absolute inset-0 z-0 bg-cover bg-center transition-all duration-1000"
                 style={{ backgroundImage: `url('${currentBackground}')` }}
               >
                 <div className="absolute inset-0 bg-white/5 backdrop-blur-[2px]"></div>
               </div>
            )}
            {/* Mobile Overlay */}
            {isSidebarOpen && (
              <div 
                className="fixed inset-0 bg-slate-900/20 z-[100] lg:hidden"
                onClick={() => setIsSidebarOpen(false)}
              />
            )}
            {/* Top Navigation Bar */}
            <header className="h-20 bg-slate-900/80 backdrop-blur-md border-b border-slate-800 px-4 lg:px-8 flex items-center justify-between shrink-0 relative z-50 overflow-x-auto no-scrollbar">
              <div className="flex items-center gap-4 lg:gap-6 min-w-max">
                {!isSidebarOpen && (
                  <button onClick={() => setIsSidebarOpen(true)} className="h-10 w-10 bg-white/20 backdrop-blur-md rounded-xl flex items-center justify-center text-white hover:text-orange-400 transition-all border border-white/30 cursor-pointer">
                    <i className="fa-solid fa-bars"></i>
                  </button>
                )}
                
                <div className="flex items-center gap-4">
                  <button 
                    onClick={handleGenerate}
                    disabled={isGenerating}
                    className="px-6 lg:px-8 py-3 bg-orange-600 text-white rounded-xl text-[11px] font-bold uppercase tracking-widest flex items-center gap-3 hover:bg-orange-700 transition-all shadow-lg shadow-orange-200/50 active:scale-95 disabled:opacity-50 whitespace-nowrap"
                  >
                    {isGenerating ? <i className="fa-solid fa-spinner animate-spin"></i> : <i className="fa-solid fa-wand-magic-sparkles"></i>}
                    Build Test
                  </button>

                  <div className="flex bg-white/20 backdrop-blur-md p-1 rounded-xl gap-1 border border-white/30 min-w-max relative z-[50]">
                    <button 
                      onClick={() => {
                        const next = (baseLayout + 1) % 9;
                        setBaseLayout(next);
                        if (next === 4) setMcqLayout('quad');
                      }}
                      className={`px-4 lg:px-6 py-2 rounded-lg text-[11px] font-bold flex items-center gap-2 transition-all whitespace-nowrap cursor-pointer ${baseLayout > 0 ? 'bg-blue-600 text-white shadow-md' : 'text-white hover:bg-white/40'}`}
                    >
                      <i className={`fa-solid ${baseLayout === 4 ? 'fa-columns' : baseLayout === 3 ? 'fa-arrows-left-right' : baseLayout === 2 ? 'fa-table-columns' : baseLayout === 1 ? 'fa-grip-lines' : 'fa-list'} text-[10px]`}></i> 
                      {baseLayout === 0 ? 'Option 1' : baseLayout === 1 ? 'Option 2' : baseLayout === 2 ? 'Option 3' : baseLayout === 3 ? 'Option 4' : baseLayout === 4 ? 'Option 5' : `Option ${baseLayout + 1}`}
                    </button>
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        const next = (instructionRulerStyle + 1) % 7;
                        setInstructionRulerStyle(next);
                      }}
                      className={`px-4 lg:px-6 py-2 rounded-lg text-[11px] font-bold flex items-center gap-2 transition-all whitespace-nowrap cursor-pointer ${instructionRulerStyle > 0 ? 'bg-indigo-600 text-white shadow-md' : 'text-white hover:bg-white/40'}`}
                    >
                      <i className="fa-solid fa-ruler-horizontal text-[10px]"></i> 
                      {instructionRulerStyle === 0 ? 'Ruler: None' : 
                       instructionRulerStyle === 1 ? 'Ruler: S1' :
                       instructionRulerStyle === 2 ? 'Ruler: S2' :
                       instructionRulerStyle === 3 ? 'Ruler: S3' :
                       instructionRulerStyle === 4 ? 'Ruler: S4' :
                       instructionRulerStyle === 5 ? 'Ruler: S5' :
                       'Ruler: S6'}
                    </button>
                    <button 
                      onClick={() => setShowSettings(true)}
                      className="px-4 lg:px-6 py-2 text-white hover:bg-white/40 rounded-lg text-[11px] font-bold flex items-center gap-2 transition-all whitespace-nowrap cursor-pointer"
                    >
                      <i className="fa-solid fa-eye text-[10px]"></i> Workspace
                    </button>
                    <button 
                      onClick={() => fileInputRef.current?.click()}
                      className="px-4 lg:px-6 py-2 text-white hover:bg-white/40 rounded-lg text-[11px] font-bold flex items-center gap-2 transition-all whitespace-nowrap cursor-pointer"
                    >
                      <i className="fa-solid fa-file-import text-[10px]"></i> Source
                    </button>
                    <button 
                      onClick={() => setIsFrameEnabled(!isFrameEnabled)}
                      className={`px-4 lg:px-6 py-2 rounded-lg text-[11px] font-bold flex items-center gap-2 transition-all whitespace-nowrap cursor-pointer ${isFrameEnabled ? 'bg-orange-600 text-white shadow-md' : 'text-white hover:bg-white/40'}`}
                    >
                      <i className={`fa-solid ${isFrameEnabled ? 'fa-square-check' : 'fa-square'} text-[10px]`}></i> Frame
                    </button>
                    <button 
                      onClick={() => setEnablePages(!enablePages)}
                      className={`px-4 lg:px-6 py-2 rounded-lg text-[11px] font-bold flex items-center gap-2 transition-all whitespace-nowrap cursor-pointer ${enablePages ? 'bg-purple-600 text-white shadow-md' : 'text-white hover:bg-white/40'}`}
                    >
                      <i className={`fa-solid ${enablePages ? 'fa-square-check' : 'fa-square'} text-[10px]`}></i> Pages
                    </button>
                    <button 
                      onClick={() => setIsPartBackgroundEnabled(!isPartBackgroundEnabled)}
                      className={`px-4 lg:px-6 py-2 rounded-lg text-[11px] font-bold flex items-center gap-2 transition-all whitespace-nowrap ${isPartBackgroundEnabled ? 'bg-emerald-600 text-white shadow-md' : 'text-slate-700 hover:bg-white/40'}`}
                    >
                      <i className={`fa-solid ${isPartBackgroundEnabled ? 'fa-square-check' : 'fa-square'} text-[10px]`}></i> Part BG
                    </button>
                    <button 
                      onClick={() => setIsInstructionBackgroundEnabled(!isInstructionBackgroundEnabled)}
                      className={`px-4 lg:px-6 py-2 rounded-lg text-[11px] font-bold flex items-center gap-2 transition-all whitespace-nowrap ${isInstructionBackgroundEnabled ? 'bg-amber-600 text-white shadow-md' : 'text-slate-700 hover:bg-white/40'}`}
                    >
                      <i className={`fa-solid ${isInstructionBackgroundEnabled ? 'fa-square-check' : 'fa-square'} text-[10px]`}></i> Instruction BG
                    </button>
                    <button 
                      onClick={() => setIsColorfulBackgroundEnabled(!isColorfulBackgroundEnabled)}
                      className={`px-4 lg:px-6 py-2 rounded-lg text-[11px] font-bold flex items-center gap-2 transition-all whitespace-nowrap ${isColorfulBackgroundEnabled ? 'bg-cyan-600 text-white shadow-md' : 'text-slate-700 hover:bg-white/40'}`}
                    >
                      <i className={`fa-solid ${isColorfulBackgroundEnabled ? 'fa-square-check' : 'fa-square'} text-[10px]`}></i> Color BG
                    </button>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-4 shrink-0 px-4">
                <button className="h-10 w-10 text-slate-700 hover:text-orange-600 transition-colors bg-white/20 backdrop-blur-md rounded-xl border border-white/30 shrink-0">
                  <i className="fa-solid fa-palette text-lg"></i>
                </button>
              </div>
            </header>

            <div className="flex-1 overflow-y-auto p-8 no-scrollbar relative z-10">
              <div className="max-w-6xl mx-auto space-y-8">
                {/* 3-Column Layout: Templates Left | Global Config | Templates Right */}
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-start">
                  {/* Templates Left (Half) */}
                  <div className="lg:col-span-1 space-y-4">
                    <div className="flex items-center justify-between px-2 mb-4">
                      <div className="flex items-center gap-2">
                        <div className="h-1 w-4 bg-orange-500 rounded-full"></div>
                        <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Templates (A-M)</h3>
                      </div>
                    </div>
                    <div className="space-y-3">
                      {instructionTemplates
                        .filter(t => t.category?.toUpperCase() === activeModule.toUpperCase() || t.category?.toUpperCase() === 'ALL')
                        .sort((a, b) => {
                          const order = ['g_mcq', 'g_correct_incorrect', 'g_circle', 'g_best_rewrite', 'g_complete_sentences', 'g_pair', 'g_spelling'];
                          const aIdx = order.indexOf(a.id);
                          const bIdx = order.indexOf(b.id);
                          if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx;
                          if (aIdx !== -1) return -1;
                          if (bIdx !== -1) return 1;
                          return 0;
                        })
                        .slice(0, Math.ceil(instructionTemplates.filter(t => t.category?.toUpperCase() === activeModule.toUpperCase() || t.category?.toUpperCase() === 'ALL').length / 2))
                        .map((t, idx) => {
                          const isSelected = selectedInstructionIds.includes(t.id);
                          const cat = t.category?.toUpperCase();
                          const colorClass = cat === 'VOCABULARY' ? 'emerald' : cat === 'READING' ? 'blue' : 'orange';
                          
                          return (
                            <div
                              key={idx}
                              className={`group bg-white border rounded-2xl p-4 flex items-center justify-between hover:border-${colorClass}-200 hover:shadow-md transition-all cursor-pointer ${isSelected ? `border-${colorClass}-500 bg-${colorClass}-50/30` : 'border-slate-100'}`}
                              onClick={() => toggleInstruction(t.id)}
                            >
                              <div className="flex items-center gap-3">
                                <div className={`h-9 w-9 rounded-xl flex items-center justify-center transition-colors ${isSelected ? `bg-${colorClass}-600 text-white` : `bg-slate-50 text-slate-400 group-hover:bg-${colorClass}-50 group-hover:text-${colorClass}-500`}`}>
                                  <i className="fa-solid fa-book text-sm"></i>
                                </div>
                                <span className={`text-[10px] font-bold uppercase tracking-tight ${isSelected ? 'text-slate-900' : 'text-slate-600 group-hover:text-slate-900'}`}>{t.label}</span>
                              </div>
                              <div className={`h-6 w-6 rounded-lg border flex items-center justify-center transition-all ${isSelected ? `bg-${colorClass}-600 border-${colorClass}-600 text-white` : `border-slate-100 text-slate-300 group-hover:border-${colorClass}-500 group-hover:text-${colorClass}-500`}`}>
                                <i className={`fa-solid ${isSelected ? 'fa-check' : 'fa-plus'} text-[10px]`}></i>
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  </div>

                  <div className="lg:col-span-3 space-y-8">
                    <div className="bg-white rounded-[32px] p-10 border border-slate-100 shadow-sm space-y-8">
                      {sourceMaterial && (
                        <div className="flex items-center gap-3 bg-emerald-50 px-4 py-2 rounded-xl border border-emerald-100 w-fit">
                          <i className="fa-solid fa-file-circle-check text-emerald-500"></i>
                          <span className="text-[10px] font-bold text-emerald-600 uppercase tracking-widest">{sourceMaterial.name} attached</span>
                          <button onClick={() => setSourceMaterial(null)} className="text-emerald-400 hover:text-emerald-600 ml-2">
                            <i className="fa-solid fa-xmark"></i>
                          </button>
                        </div>
                      )}

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        <div className="space-y-3">
                          <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Language</label>
                          <select 
                            value={activeLanguage}
                            onChange={(e) => setActiveLanguage(e.target.value)}
                            className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-5 py-4 text-slate-700 font-bold text-sm outline-none focus:border-orange-200 transition-all appearance-none cursor-pointer"
                          >
                            {LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
                          </select>
                        </div>

                        <div className="space-y-3">
                          <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Academic Level</label>
                          <select 
                            value={activeLevel}
                            onChange={(e) => setActiveLevel(e.target.value as AcademicLevel)}
                            className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-5 py-4 text-slate-700 font-bold text-sm outline-none focus:border-orange-200 transition-all appearance-none cursor-pointer"
                          >
                            {ACADEMIC_LEVELS.map(l => <option key={l} value={l}>{l}</option>)}
                          </select>
                        </div>

                        <div className="space-y-3">
                          <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Universal Topic</label>
                          <input 
                            type="text"
                            value={topic}
                            onChange={(e) => setTopic(e.target.value)}
                            placeholder="e.g. Present Simple, My Family..."
                            className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-5 py-4 text-slate-700 font-bold text-sm outline-none focus:border-orange-200 transition-all placeholder:text-slate-300"
                          />
                        </div>
                      </div>
                    </div>

                    {/* Test Structure Section (Moved here for better flow) */}
                    <div className="space-y-6">
                      <div className="flex items-center justify-between px-2">
                        <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">Selected Exercises ({selectedInstructionIds.length})</h3>
                        {selectedInstructionIds.length > 0 && (
                          <button 
                            onClick={() => setSelectedInstructionIds([])}
                            className="text-[10px] font-bold text-red-400 hover:text-red-600 uppercase tracking-widest transition-colors"
                          >
                            Clear All
                          </button>
                        )}
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {instructionTemplates.filter(t => (t.category?.toUpperCase() === activeModule.toUpperCase() || t.category?.toUpperCase() === 'ALL') && selectedInstructionIds.includes(t.id)).map((t, idx) => {
                          const curItems = itemCountOverrides[t.id] || 10;
                          const curCols = columnOverrides[t.id] !== undefined ? columnOverrides[t.id] : (t.columnCount !== undefined ? t.columnCount : defaultColumnCount);
                          
                          // Diverse color mapping based on index
                          const colors = ['orange', 'blue', 'emerald', 'rose', 'violet', 'amber', 'indigo', 'cyan'];
                          const colorClass = colors[idx % colors.length];
                          
                          // Relaxing backgrounds
                          const backgrounds = [
                            'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&q=80&w=400&h=200', // Forest
                            'https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&q=80&w=400&h=200', // Mountain
                            'https://images.unsplash.com/photo-1505118380757-91f5f5632de0?auto=format&fit=crop&q=80&w=400&h=200', // Ocean
                            'https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&q=80&w=400&h=200', // Lake
                            'https://images.unsplash.com/photo-1470770841072-f978cf4d019e?auto=format&fit=crop&q=80&w=400&h=200', // Meadow
                          ];
                          const bgUrl = backgrounds[idx % backgrounds.length];

                          return (
                            <div key={t.id} className={`card-gradient-${colorClass} rounded-2xl p-3 border border-${colorClass}-100 compact-shadow group hover:shadow-md transition-all relative overflow-hidden`}>
                              {/* Relaxing Background Overlay */}
                              <div 
                                className="absolute inset-0 opacity-[0.08] pointer-events-none bg-cover bg-center mix-blend-multiply"
                                style={{ backgroundImage: `url(${bgUrl})` }}
                              />
                              
                              <div className="absolute top-0 right-0 p-1 opacity-0 group-hover:opacity-100 transition-opacity z-10">
                                <button onClick={() => toggleInstruction(t.id)} className="h-6 w-6 bg-red-50 text-red-500 rounded-lg flex items-center justify-center hover:bg-red-500 hover:text-white transition-all shadow-sm">
                                  <i className="fa-solid fa-trash-can text-[9px]"></i>
                                </button>
                              </div>
                              
                              <div className="flex items-center gap-2 mb-2 relative z-10">
                                <div className={`h-7 w-7 bg-white text-${colorClass}-600 rounded-lg flex items-center justify-center shadow-sm border border-${colorClass}-100 flex-shrink-0`}>
                                  <i className="fa-solid fa-star text-[10px]"></i>
                                </div>
                                <span className="text-[10px] font-black text-slate-800 uppercase tracking-tight leading-tight truncate">{t.label}</span>
                              </div>
                              
                              <div className="flex flex-col gap-2 relative z-10">
                                <div className="space-y-1">
                                  <div className="flex justify-between items-center px-1">
                                    <span className="text-[8px] font-bold text-slate-400 uppercase tracking-widest">Item</span>
                                    <span className={`text-[9px] font-black text-${colorClass}-600`}>{curItems}</span>
                                  </div>
                                  <div className="flex bg-white/60 backdrop-blur-sm rounded-lg p-0.5 gap-0.5 border border-slate-100 shadow-inner">
                                    {[5, 10, 15, 20, 25, 30].map(num => (
                                      <button 
                                        key={num} 
                                        onClick={() => setItemCount(t.id, num)} 
                                        className={`flex-1 h-5 rounded-md text-[8px] font-bold transition-all ${curItems === num ? `bg-white text-${colorClass}-600 shadow-sm border border-${colorClass}-50` : 'text-slate-400 hover:text-slate-600'}`}
                                      >
                                        {num}
                                      </button>
                                    ))}
                                  </div>
                                </div>
                                <div className="space-y-1">
                                  <div className="flex justify-between items-center px-1">
                                    <span className="text-[8px] font-bold text-slate-400 uppercase tracking-widest">Column</span>
                                    <span className="text-[9px] font-black text-slate-600">{curCols || 'L'}</span>
                                  </div>
                                  <div className="flex bg-white/60 backdrop-blur-sm rounded-lg p-0.5 gap-0.5 border border-slate-100 shadow-inner">
                                    {[0, 1, 2, 3, 4, 6].map(num => (
                                      <button 
                                        key={num} 
                                        onClick={() => setColumnOverrides(prev => ({ ...prev, [t.id]: num }))} 
                                        className={`flex-1 h-5 rounded-md text-[8px] font-bold transition-all ${curCols === num ? `bg-white text-${colorClass}-600 shadow-sm border border-${colorClass}-50` : 'text-slate-400 hover:text-slate-600'}`}
                                      >
                                        {num === 0 ? 'L' : num}
                                      </button>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            </div>
                          );
                        })}

                        {selectedInstructionIds.length === 0 && (
                          <div className="md:col-span-2 h-40 border-2 border-dashed border-slate-200 rounded-[32px] bg-slate-50/50 flex flex-col items-center justify-center text-center p-6">
                            <div className="h-10 w-10 bg-white rounded-full flex items-center justify-center text-slate-200 mb-3 shadow-sm">
                              <i className="fa-solid fa-plus text-lg"></i>
                            </div>
                            <h4 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">No Exercises Selected</h4>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Templates Right (Half) */}
                  <div className="lg:col-span-1 space-y-4">
                    <div className="flex items-center justify-between px-2 mb-4">
                      <div className="flex items-center gap-2">
                        <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Templates (N-Z)</h3>
                        <div className="h-1 w-4 bg-orange-500 rounded-full"></div>
                      </div>
                    </div>
                    <div className="space-y-3">
                      {instructionTemplates
                        .filter(t => t.category?.toUpperCase() === activeModule.toUpperCase() || t.category?.toUpperCase() === 'ALL')
                        .sort((a, b) => {
                          const order = ['g_mcq', 'g_correct_incorrect', 'g_circle', 'g_best_rewrite', 'g_complete_sentences', 'g_pair', 'g_spelling'];
                          const aIdx = order.indexOf(a.id);
                          const bIdx = order.indexOf(b.id);
                          if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx;
                          if (aIdx !== -1) return -1;
                          if (bIdx !== -1) return 1;
                          return 0;
                        })
                        .slice(Math.ceil(instructionTemplates.filter(t => t.category?.toUpperCase() === activeModule.toUpperCase() || t.category?.toUpperCase() === 'ALL').length / 2))
                        .map((t, idx) => {
                          const isSelected = selectedInstructionIds.includes(t.id);
                          const cat = t.category?.toUpperCase();
                          const colorClass = cat === 'VOCABULARY' ? 'emerald' : cat === 'READING' ? 'blue' : 'orange';
                          
                          return (
                            <div
                              key={idx}
                              className={`group bg-white border rounded-2xl p-4 flex items-center justify-between hover:border-${colorClass}-200 hover:shadow-md transition-all cursor-pointer ${isSelected ? `border-${colorClass}-500 bg-${colorClass}-50/30` : 'border-slate-100'}`}
                              onClick={() => toggleInstruction(t.id)}
                            >
                              <div className="flex items-center gap-3">
                                <div className={`h-9 w-9 rounded-xl flex items-center justify-center transition-colors ${isSelected ? `bg-${colorClass}-600 text-white` : `bg-slate-50 text-slate-400 group-hover:bg-${colorClass}-50 group-hover:text-${colorClass}-500`}`}>
                                  <i className="fa-solid fa-book text-sm"></i>
                                </div>
                                <span className={`text-[10px] font-bold uppercase tracking-tight ${isSelected ? 'text-slate-900' : 'text-slate-600 group-hover:text-slate-900'}`}>{t.label}</span>
                              </div>
                              <div className={`h-6 w-6 rounded-lg border flex items-center justify-center transition-all ${isSelected ? `bg-${colorClass}-600 border-${colorClass}-600 text-white` : `border-slate-100 text-slate-300 group-hover:border-${colorClass}-500 group-hover:text-${colorClass}-500`}`}>
                                <i className={`fa-solid ${isSelected ? 'fa-check' : 'fa-plus'} text-[10px]`}></i>
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                </div>

                {/* Main Content Grid (Now just Live Output) */}
                <div className="grid grid-cols-1 gap-8">
                  {/* Live Output Section */}
                  <div className="space-y-6">
                    <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest px-2">Live Output</h3>
                    <div className="h-[400px] bg-white rounded-[32px] border border-slate-100 shadow-sm flex flex-col items-center justify-center text-center p-10 relative overflow-hidden">
                      <div className="absolute inset-0 bg-gradient-to-b from-slate-50/50 to-transparent pointer-events-none"></div>
                      <div className="h-24 w-24 bg-slate-50 rounded-full flex items-center justify-center text-slate-100 mb-8">
                        <i className="fa-solid fa-sparkles text-4xl"></i>
                      </div>
                      <h4 className="text-base font-bold text-slate-800 mb-3">Ready to Build</h4>
                      <p className="text-xs text-slate-400 max-w-[240px] leading-relaxed">Configure your test and click "Build Test" to generate your assessment.</p>
                      
                      {isGenerating && (
                        <div className="fixed bottom-8 right-8 bg-white border border-slate-200 rounded-[32px] p-6 shadow-2xl z-[9999] animate-in slide-in-from-bottom-10 duration-500 max-w-sm w-full border-b-4 border-b-orange-500">
                          <div className="flex items-center gap-6">
                            <div className="relative flex-shrink-0">
                              <div className="h-16 w-16 border-[4px] border-slate-100 border-t-orange-600 rounded-full animate-spin"></div>
                              <div className="absolute inset-0 flex items-center justify-center">
                                <i className="fa-solid fa-brain-circuit text-xl text-orange-600 animate-pulse"></i>
                              </div>
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center justify-between mb-1">
                                <h3 className="text-[11px] font-black text-slate-900 uppercase tracking-widest flex items-center gap-2">
                                  <i className="fa-solid fa-microchip text-orange-600"></i>
                                  Neural Synthesis
                                </h3>
                                <div className="flex items-center gap-1.5">
                                  <div className="h-1.5 w-1.5 bg-orange-600 rounded-full animate-ping"></div>
                                  <span className="text-[9px] font-black text-orange-600 uppercase tracking-widest">{generationStep || 'Processing...'}</span>
                                </div>
                              </div>
                              <p className="text-[10px] text-slate-400 leading-relaxed mb-3">
                                Crafting your professional assessment with deep semantic analysis...
                              </p>
                              <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden shadow-inner">
                                <div className="h-full bg-gradient-to-r from-orange-500 to-orange-700 animate-progress w-full"></div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {generationError && (
                        <div className="absolute inset-0 bg-white/95 backdrop-blur-md flex flex-col items-center justify-center z-30 p-10">
                          <div className="h-16 w-16 bg-red-50 rounded-full flex items-center justify-center text-red-600 mb-6 shadow-xl shadow-red-600/10">
                            <i className="fa-solid fa-triangle-exclamation text-2xl"></i>
                          </div>
                          <h4 className="text-sm font-bold text-slate-800 mb-2">Neural Synthesis Failed</h4>
                          <p className="text-xs text-slate-400 max-w-[280px] leading-relaxed mb-8">The AI engine encountered an issue. This could be due to a complex prompt or temporary service interruption.</p>
                          <div className="flex gap-4">
                            <button onClick={() => setGenerationError(null)} className="px-6 py-3 bg-slate-100 text-slate-600 rounded-xl text-[10px] font-bold uppercase tracking-widest hover:bg-slate-200 transition-all">Dismiss</button>
                            <button onClick={handleGenerate} className="px-6 py-3 bg-orange-600 text-white rounded-xl text-[10px] font-bold uppercase tracking-widest hover:bg-orange-700 shadow-lg shadow-orange-600/20 transition-all flex items-center gap-2">
                              <i className="fa-solid fa-rotate-right"></i>
                              Retry Synthesis
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </main>
        </>
      )}

      {viewMode === 'preview' && (
        <section 
          style={{ 
            marginLeft: isSidebarOpen && !isFullscreen && sidebarSide === 'left' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px',
            marginRight: isSidebarOpen && !isFullscreen && sidebarSide === 'right' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px'
          }}
          className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-500 bg-slate-50 transition-all duration-300"
        >
          {!isFullscreen && (
            <div className="p-4 lg:p-6 bg-white border-b border-slate-200 flex flex-wrap gap-4 justify-between items-center z-10 no-print shadow-sm">
              <button onClick={() => setViewMode('generator')} className="border border-slate-200 text-slate-600 px-6 lg:px-8 py-3 rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-slate-50 flex items-center gap-4 group transition-all">
                <i className="fa-solid fa-arrow-left group-hover:-translate-x-1 transition-transform"></i> WORKSPACE
              </button>
              
              <div className="flex items-center gap-4 bg-slate-50 px-4 py-2 rounded-xl border border-slate-100">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Zoom</span>
                <input 
                  type="range" 
                  min="50" 
                  max="150" 
                  value={previewZoom} 
                  onChange={(e) => setPreviewZoom(parseInt(e.target.value))}
                  className="w-24 h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-orange-600"
                />
                <span className="text-[10px] font-bold text-slate-600 w-8">{previewZoom}%</span>
              </div>

              <div className="flex gap-2 lg:gap-3 ml-auto">
                <button onClick={() => setIsFullscreen(true)} className="h-10 w-10 lg:h-12 lg:w-12 bg-white border border-slate-200 rounded-xl flex items-center justify-center text-slate-400 hover:text-orange-600 transition-all shadow-sm" title="Fullscreen">
                  <i className="fa-solid fa-expand"></i>
                </button>
                <button 
                  onClick={() => exportToPDF('worksheet-container', `Test_${new Date().getTime()}`)}
                  className="px-4 py-3 bg-red-600 text-white rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-red-700 shadow-sm flex items-center gap-2 transition-all"
                >
                  <i className="fa-solid fa-file-pdf"></i> PDF
                </button>
                <button 
                  onClick={() => exportToHTML(worksheetContent, `Test_${new Date().getTime()}`)}
                  className="px-4 py-3 bg-blue-600 text-white rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-blue-700 shadow-sm flex items-center gap-2 transition-all"
                >
                  <i className="fa-solid fa-code"></i> HTML
                </button>
                <button 
                  onClick={handleExportWord}
                  className="px-4 py-3 bg-indigo-600 text-white rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-indigo-700 shadow-sm flex items-center gap-2 transition-all"
                >
                  <i className="fa-solid fa-file-word"></i> Word
                </button>
                <button onClick={handlePrint} className="h-10 w-10 lg:h-12 lg:w-12 bg-white border border-slate-200 rounded-xl flex items-center justify-center text-slate-400 hover:text-orange-600 transition-all shadow-sm">
                  <i className="fa-solid fa-print"></i>
                </button>
              </div>
            </div>
          )}
          
          {isFullscreen && (
            <button 
              onClick={() => setIsFullscreen(false)}
              className="fixed top-6 right-6 h-12 w-12 bg-slate-900/80 backdrop-blur-md text-white rounded-full flex items-center justify-center z-[200] hover:bg-slate-900 transition-all shadow-2xl no-print"
            >
              <i className="fa-solid fa-compress"></i>
            </button>
          )}

          <div className="flex-1 overflow-auto no-scrollbar">
            <Worksheet 
              content={worksheetContent} 
              onContentChange={setWorksheetContent} 
              isGenerating={isGenerating} 
              theme={THEMES.find(t => t.id === activeThemeId) || THEMES[0]} 
              paperType="Plain" 
              brandSettings={brandSettings} 
              level={activeLevel} 
              module={activeModule} 
              topic={topic} 
              paperDesign={paperDesign}
              mcqStyle={mcqStyle}
              isColorfulBackgroundEnabled={isColorfulBackgroundEnabled}
              isInstructionBackgroundEnabled={isInstructionBackgroundEnabled}
              globalLayout={globalLayout}
              baseLayout={baseLayout}
              instructionRulerStyle={instructionRulerStyle}
              zoom={previewZoom}
            />
          </div>
        </section>
      )}

      {viewMode === 'grammar_iframe' && (
        <section 
          style={{ 
            marginLeft: isSidebarOpen && sidebarSide === 'left' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px',
            marginRight: isSidebarOpen && sidebarSide === 'right' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px'
          }}
          className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-500 bg-slate-50 transition-all duration-300"
        >
          <div className="p-4 lg:p-6 bg-white border-b border-slate-200 flex flex-wrap gap-4 justify-between items-center z-10 no-print shadow-sm">
            <button onClick={() => setViewMode('generator')} className="border border-slate-200 text-slate-600 px-6 lg:px-8 py-3 rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-slate-50 flex items-center gap-4 group transition-all">
              <i className="fa-solid fa-arrow-left group-hover:-translate-x-1 transition-transform"></i> WORKSPACE
            </button>
            <div className="flex-1 text-center">
              <h2 className="text-slate-800 font-bold uppercase tracking-widest text-[12px]">Neural Grammar Engine</h2>
            </div>
            <div className="flex gap-2">
              <a 
                href="https://aistudio.google.com/apps/f6448ec0-06de-44f2-93d6-13cd43bceb87?showPreview=true&showAssistant=true" 
                target="_blank" 
                rel="noopener noreferrer"
                className="px-6 py-3 bg-orange-600 text-white rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-orange-700 shadow-sm flex items-center gap-2 transition-all"
              >
                <i className="fa-solid fa-arrow-up-right-from-square"></i> Launch Tool
              </a>
            </div>
          </div>
          <div className="flex-1 bg-white overflow-hidden relative">
            <div className="absolute inset-0 flex flex-col items-center justify-center p-10 text-center bg-slate-50 -z-10">
              <i className="fa-solid fa-circle-exclamation text-4xl text-slate-300 mb-4"></i>
              <p className="text-slate-500 font-bold text-sm">If the tool refuses to connect, please use the "Launch Tool" button above.</p>
            </div>
            <iframe 
              src="https://aistudio.google.com/apps/f6448ec0-06de-44f2-93d6-13cd43bceb87?showPreview=true&showAssistant=true"
              className="w-full h-full min-h-[800px] border-none relative z-10"
              title="Grammar Tool"
              sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads allow-modals allow-top-navigation-by-user-activation"
            />
          </div>
        </section>
      )}

      {viewMode === 'khmer_program' && (
        <section 
          style={{ 
            marginLeft: isSidebarOpen && sidebarSide === 'left' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px',
            marginRight: isSidebarOpen && sidebarSide === 'right' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px'
          }}
          className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-500 bg-slate-50 transition-all duration-300"
        >
          <div className="p-4 lg:p-6 bg-white border-b border-slate-200 flex flex-wrap gap-4 justify-between items-center z-10 no-print shadow-sm">
            <button onClick={() => setViewMode('generator')} className="border border-slate-200 text-slate-600 px-6 lg:px-8 py-3 rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-slate-50 flex items-center gap-4 group transition-all">
              <i className="fa-solid fa-arrow-left group-hover:-translate-x-1 transition-transform"></i> WORKSPACE
            </button>
            <div className="flex-1 text-center">
              <h2 className="text-slate-800 font-bold uppercase tracking-widest text-[12px]">Khmer Program Test Builder</h2>
            </div>
            <div className="flex gap-2">
              <a 
                href="https://chanthy-master-engine-gbcdawq79gtmzdw7cqfh7f.streamlit.app/?tool=khmer_program&embed=true" 
                target="_blank" 
                rel="noopener noreferrer"
                className="px-6 py-3 bg-orange-600 text-white rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-orange-700 shadow-sm flex items-center gap-2 transition-all"
              >
                <i className="fa-solid fa-arrow-up-right-from-square"></i> Launch Tool
              </a>
            </div>
          </div>
          <div className="flex-1 bg-white overflow-hidden relative">
            <div className="absolute inset-0 flex flex-col items-center justify-center p-10 text-center bg-slate-50 -z-10">
              <i className="fa-solid fa-circle-exclamation text-4xl text-slate-300 mb-4"></i>
              <p className="text-slate-500 font-bold text-sm">If the tool refuses to connect, please use the "Launch Tool" button above.</p>
            </div>
            <iframe 
              src="https://chanthy-master-engine-gbcdawq79gtmzdw7cqfh7f.streamlit.app/?tool=khmer_program&embed=true"
              className="w-full h-full min-h-[800px] border-none relative z-10"
              title="Khmer Program Tool"
              sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads allow-modals allow-top-navigation-by-user-activation"
            />
          </div>
        </section>
      )}

      {viewMode === 'book_creation' && (
        <section 
          style={{ 
            marginLeft: isSidebarOpen && sidebarSide === 'left' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px',
            marginRight: isSidebarOpen && sidebarSide === 'right' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px'
          }}
          className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-500 bg-slate-50 transition-all duration-300"
        >
          <div className="p-4 lg:p-6 bg-white border-b border-slate-200 flex flex-wrap gap-4 justify-between items-center z-10 no-print shadow-sm">
            <button onClick={() => setViewMode('generator')} className="border border-slate-200 text-slate-600 px-6 lg:px-8 py-3 rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-slate-50 flex items-center gap-4 group transition-all">
              <i className="fa-solid fa-arrow-left group-hover:-translate-x-1 transition-transform"></i> WORKSPACE
            </button>
            <div className="flex-1 text-center">
              <h2 className="text-slate-800 font-bold uppercase tracking-widest text-[12px]">Neural Book Engine</h2>
            </div>
            <div className="flex gap-2">
              <a 
                href="https://chanthy-master-engine-gbcdawq79gtmzdw7cqfh7f.streamlit.app/?tool=book_creation&embed=true" 
                target="_blank" 
                rel="noopener noreferrer"
                className="px-6 py-3 bg-orange-600 text-white rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-orange-700 shadow-sm flex items-center gap-2 transition-all"
              >
                <i className="fa-solid fa-arrow-up-right-from-square"></i> Launch Tool
              </a>
            </div>
          </div>
          <div className="flex-1 bg-white overflow-hidden relative">
            <div className="absolute inset-0 flex flex-col items-center justify-center p-10 text-center bg-slate-50 -z-10">
              <i className="fa-solid fa-circle-exclamation text-4xl text-slate-300 mb-4"></i>
              <p className="text-slate-500 font-bold text-sm">If the tool refuses to connect, please use the "Launch Tool" button above.</p>
            </div>
            <iframe 
              src="https://chanthy-master-engine-gbcdawq79gtmzdw7cqfh7f.streamlit.app/?tool=book_creation&embed=true"
              className="w-full h-full min-h-[800px] border-none relative z-10"
              title="Book Creation Tool"
              sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads allow-modals allow-top-navigation-by-user-activation"
            />
          </div>
        </section>
      )}

      {viewMode === 'paper_style_design' && (
        <section 
          style={{ 
            marginLeft: isSidebarOpen && sidebarSide === 'left' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px',
            marginRight: isSidebarOpen && sidebarSide === 'right' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px'
          }}
          className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-500 bg-slate-50 transition-all duration-300"
        >
          <div className="p-4 lg:p-6 bg-white border-b border-slate-200 flex flex-wrap gap-4 justify-between items-center z-10 no-print shadow-sm">
            <button onClick={() => setViewMode('generator')} className="border border-slate-200 text-slate-600 px-6 lg:px-8 py-3 rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-slate-50 flex items-center gap-4 group transition-all">
              <i className="fa-solid fa-arrow-left group-hover:-translate-x-1 transition-transform"></i> WORKSPACE
            </button>
            <div className="flex-1 text-center">
              <h2 className="text-slate-800 font-bold uppercase tracking-widest text-[12px]">Paper Style Selection</h2>
            </div>
            <div className="flex gap-2">
              <button 
                onClick={() => setViewMode('generator')}
                className="px-6 py-3 bg-emerald-600 text-white rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-emerald-700 shadow-sm flex items-center gap-2 transition-all"
              >
                <i className="fa-solid fa-check"></i> Save & Set Default
              </button>
            </div>
          </div>
          <div className="flex-1 bg-slate-50 overflow-y-auto p-8 no-scrollbar">
            <div className="max-w-4xl mx-auto space-y-10">
              <div className="bg-white rounded-[32px] p-10 border border-slate-100 shadow-sm">
                <h3 className="text-xl font-black text-slate-900 mb-2 uppercase tracking-tight">Popular Paper Styles</h3>
                <p className="text-sm text-slate-500 mb-8">Choose a base layout for your test paper. Selecting a style sets it as your default.</p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
                  {[
                    { id: 0, name: 'Option 1: Clean White', desc: 'Pure white paper with subtle border.', icon: 'fa-file-lines' },
                    { id: 1, name: 'Option 2: Orange Mix', desc: 'White paper with orange accents.', icon: 'fa-palette' },
                    { id: 2, name: 'Option 3: Modern Emerald', desc: 'Professional green theme.', icon: 'fa-gem' },
                    { id: 3, name: 'Option 4: Soft Lavender', desc: 'Elegant purple theme.', icon: 'fa-feather' },
                    { id: 4, name: 'Option 5: Mint', desc: 'Very light green paper.', icon: 'fa-leaf' },
                    { id: 5, name: 'Option 6: Peach', desc: 'Very light orange paper.', icon: 'fa-sun' },
                    { id: 6, name: 'Option 7: Sky', desc: 'Very light blue paper.', icon: 'fa-cloud' },
                    { id: 7, name: 'Option 8: Lavender', desc: 'Very light purple paper.', icon: 'fa-moon' },
                    { id: 8, name: 'Option 9: Citrus', desc: 'Light Green & Orange mix.', icon: 'fa-lemon' },
                    { id: 9, name: 'Option 10: Rose', desc: 'Very light pink paper.', icon: 'fa-heart' },
                    { id: 10, name: 'Option 11: Stars', desc: 'Decorative stars background.', icon: 'fa-star' },
                    { id: 11, name: 'Option 12: Flowers', desc: 'Decorative flowers background.', icon: 'fa-flower' },
                    { id: 12, name: 'Option 13: Hearts', desc: 'Decorative hearts background.', icon: 'fa-heart' },
                    { id: 13, name: 'Option 14: Bubbles', desc: 'Decorative bubbles background.', icon: 'fa-soap' },
                    { id: 14, name: 'Option 15: Leaves', desc: 'Decorative leaves background.', icon: 'fa-leaf' },
                    { id: 15, name: 'Option 16: Rainbow', desc: 'Subtle rainbow gradient.', icon: 'fa-rainbow' },
                    { id: 16, name: 'Option 17: Galaxy', desc: 'Dark galaxy themed paper.', icon: 'fa-user-astronaut' },
                    { id: 17, name: 'Option 18: Notebook', desc: 'Classic spiral notebook style.', icon: 'fa-book-open' },
                    { id: 18, name: 'Option 19: Vintage', desc: 'Aged parchment style.', icon: 'fa-scroll' },
                    { id: 19, name: 'Option 20: Modern', desc: 'Geometric modern art style.', icon: 'fa-shapes' },
                  ].map((style) => (
                    <div 
                      key={style.id}
                      onClick={() => {
                        setGlobalLayout(style.id);
                        if (style.id === 3) {
                          setMcqLayout('quad');
                        }
                      }}
                      className={`p-8 rounded-[40px] border-2 cursor-pointer transition-all ${globalLayout === style.id ? 'border-emerald-500 bg-emerald-50/30 shadow-xl scale-[1.02]' : 'border-slate-100 bg-white hover:border-emerald-200 shadow-sm'}`}
                    >
                      <div className="flex justify-between items-start mb-6">
                        <div className={`h-14 w-14 rounded-2xl flex items-center justify-center text-2xl ${globalLayout === style.id ? 'bg-emerald-500 text-white' : 'bg-slate-100 text-slate-400'}`}>
                          <i className={`fa-solid ${style.icon}`}></i>
                        </div>
                        {globalLayout === style.id && <div className="h-8 w-8 bg-emerald-500 text-white rounded-full flex items-center justify-center shadow-lg animate-in zoom-in"><i className="fa-solid fa-check"></i></div>}
                      </div>
                      <h4 className="text-lg font-black text-slate-900 uppercase tracking-tight mb-2">{style.name}</h4>
                      <p className="text-xs font-medium text-slate-500 leading-relaxed">{style.desc}</p>
                    </div>
                  ))}
                </div>

                {/* Preview Section - Moved up for better visibility */}
                <div className="bg-slate-900 rounded-[40px] p-12 border border-slate-800 shadow-2xl overflow-hidden relative group">
                  <div className="absolute top-0 right-0 p-8">
                    <div className="px-4 py-2 bg-emerald-500/20 border border-emerald-500/30 rounded-full text-[10px] font-black text-emerald-400 uppercase tracking-widest animate-pulse">Live Preview Active</div>
                  </div>
                  <h3 className="text-2xl font-black text-white mb-2 uppercase tracking-tight">Style Visualization</h3>
                  <p className="text-sm text-slate-400 mb-12">See how your selected paper style will look in the final output.</p>
                  
                  <div className="relative flex justify-center">
                    <div className={`w-full max-w-[500px] aspect-[1/1.414] bg-white shadow-[0_50px_100px_-20px_rgba(0,0,0,0.5)] rounded-sm overflow-hidden transition-all duration-700 ${
                      tableStyle === 'list' ? 'layout-lined' : 
                      tableStyle === 'grid' ? 'layout-grid' : ''
                    } ${
                      globalLayout === 0 ? 'layout-clean-white' :
                      globalLayout === 1 ? 'layout-orange-mix' :
                      globalLayout === 2 ? 'layout-modern-emerald' :
                      globalLayout === 3 ? 'layout-soft-lavender' :
                      globalLayout === 4 ? 'layout-mint' :
                      globalLayout === 5 ? 'layout-peach' :
                      globalLayout === 6 ? 'layout-sky' :
                      globalLayout === 7 ? 'layout-lavender' :
                      globalLayout === 8 ? 'layout-citrus' :
                      globalLayout === 9 ? 'layout-rose' : 
                      globalLayout === 10 ? 'layout-stars' :
                      globalLayout === 11 ? 'layout-flowers' :
                      globalLayout === 12 ? 'layout-hearts' :
                      globalLayout === 13 ? 'layout-bubbles' :
                      globalLayout === 14 ? 'layout-leaves' :
                      globalLayout === 15 ? 'layout-rainbow' :
                      globalLayout === 16 ? 'layout-galaxy' :
                      globalLayout === 17 ? 'layout-notebook' :
                      globalLayout === 18 ? 'layout-vintage' :
                      globalLayout === 19 ? 'layout-modern' : ''
                    }`} style={{ transform: 'scale(0.9)' }}>
                      <div className="p-12 space-y-8">
                        <div className="h-6 w-3/4 bg-slate-200 rounded-full"></div>
                        <div className="h-6 w-1/2 bg-slate-200 rounded-full"></div>
                        <div className="h-6 w-full bg-slate-100 rounded-full"></div>
                        <div className="pt-12 space-y-6">
                          <div className="h-4 w-1/3 bg-slate-300 rounded-full"></div>
                          <div className="grid grid-cols-2 gap-6">
                            <div className="h-4 bg-slate-100 rounded-full"></div>
                            <div className="h-4 bg-slate-100 rounded-full"></div>
                          </div>
                        </div>
                        <div className="space-y-6">
                          <div className="h-4 w-1/3 bg-slate-300 rounded-full"></div>
                          <div className="grid grid-cols-2 gap-6">
                            <div className="h-4 bg-slate-100 rounded-full"></div>
                            <div className="h-4 bg-slate-100 rounded-full"></div>
                          </div>
                        </div>
                      </div>
                      {/* Ruler Simulation for Preview */}
                      {baseLayout === 4 && (
                        <div className="absolute left-0 top-0 bottom-0 w-[40px] bg-slate-50 border-r-2 border-red-300 flex items-center justify-center">
                          <div className="h-full w-[1px] bg-red-200"></div>
                        </div>
                      )}
                      {baseLayout === 3 && (
                        <div className="absolute left-1/2 top-0 bottom-0 w-[2px] bg-red-500 opacity-50"></div>
                      )}
                      {instructionRulerStyle > 0 && (
                        <div className="absolute top-[100px] left-1/2 -translate-x-1/2 w-3/4 h-[2px] bg-slate-800"></div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {viewMode === 'design_test_style' && (
        <section 
          style={{ 
            marginLeft: isSidebarOpen && sidebarSide === 'left' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px',
            marginRight: isSidebarOpen && sidebarSide === 'right' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px'
          }}
          className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-500 bg-slate-50 transition-all duration-300"
        >
          <div className="p-4 lg:p-6 bg-white border-b border-slate-200 flex flex-wrap gap-4 justify-between items-center z-10 no-print shadow-sm">
            <button onClick={() => setViewMode('generator')} className="border border-slate-200 text-slate-600 px-6 lg:px-8 py-3 rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-slate-50 flex items-center gap-4 group transition-all">
              <i className="fa-solid fa-arrow-left group-hover:-translate-x-1 transition-transform"></i> WORKSPACE
            </button>
            <div className="flex-1 text-center">
              <h2 className="text-slate-800 font-bold uppercase tracking-widest text-[12px]">Design Test Style</h2>
            </div>
            <div className="flex gap-2">
              <button 
                onClick={() => setViewMode('generator')}
                className="px-6 py-3 bg-orange-600 text-white rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-orange-700 shadow-sm flex items-center gap-2 transition-all"
              >
                <i className="fa-solid fa-check"></i> Save & Set Default
              </button>
            </div>
          </div>
          <div className="flex-1 bg-slate-50 overflow-y-auto p-8 no-scrollbar">
            <div className="max-w-4xl mx-auto space-y-12">
              <div className="bg-white rounded-[32px] p-10 border border-slate-100 shadow-sm">
                <div className="flex justify-between items-center mb-8">
                  <div>
                    <h3 className="text-xl font-black text-slate-900 mb-2 uppercase tracking-tight">Design Test Style Library</h3>
                    <p className="text-sm text-slate-500">Customize the visual structure and default formatting for each question type.</p>
                  </div>
                  <div className="flex items-center gap-4">
                    <button 
                      onClick={handleAddCustomExerciseType}
                      className="px-6 py-3 bg-slate-100 text-slate-700 rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-slate-200 shadow-sm flex items-center gap-2 transition-all"
                    >
                      <i className="fa-solid fa-plus"></i> Add New Exercise Type
                    </button>
                    <button 
                      onClick={() => {
                        setSettingsTab('FORMAT_DESIGN');
                        setShowSettings(true);
                      }}
                      className="px-6 py-3 bg-blue-600 text-white rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-blue-700 shadow-lg shadow-blue-100 flex items-center gap-2 transition-all"
                    >
                      <i className="fa-solid fa-wand-magic-sparkles"></i> Create Custom Format
                    </button>
                  </div>
                </div>
                
                <div className="flex gap-2 mb-10 border-b border-slate-200 pb-4">
                  {['Grammar', 'Vocabulary', 'Reading', 'Mixed', 'Generals', 'Custom'].map(tab => (
                    <button
                      key={tab}
                      onClick={() => setArchitectTab(tab as any)}
                      className={`px-6 py-3 rounded-xl text-xs font-bold uppercase tracking-widest transition-all ${architectTab === tab ? 'bg-slate-800 text-white shadow-md' : 'bg-slate-100 text-slate-500 hover:bg-slate-200'}`}
                    >
                      {tab}
                    </button>
                  ))}
                </div>

                <div className="space-y-20">
                  {architectTab === 'Grammar' && (
                    <>
                  {/* Subjects Section */}
                  <div className="bg-white p-8 rounded-[40px] border border-slate-100 shadow-sm">
                    <CollapsibleSection
                      title="Localization Subjects"
                      subtitle="Select country for names and places"
                      icon="fa-earth-americas"
                      iconBg="bg-amber-100"
                      iconColor="text-amber-600"
                      isCollapsed={!!collapsedSections['localization_subjects']}
                      onToggle={() => setCollapsedSections(prev => ({ ...prev, localization_subjects: !prev.localization_subjects }))}
                    >
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
                        {SUBJECTS.map(subject => (
                          <button
                            key={subject.id}
                            onClick={() => {
                              setActiveSubject(subject.id);
                              setIsRandomSubject(false);
                            }}
                            className={`p-4 rounded-2xl border-2 transition-all flex flex-col gap-1 items-center text-center ${activeSubject === subject.id && !isRandomSubject ? 'border-amber-500 bg-amber-50 shadow-md' : 'border-slate-100 bg-slate-50 hover:border-amber-200'}`}
                          >
                            <span className="text-[10px] font-black uppercase text-slate-900">{subject.name}</span>
                            <span className="text-[8px] text-slate-400">{subject.names[0]}, {subject.names[1]}...</span>
                          </button>
                        ))}
                        <button
                          onClick={() => setIsRandomSubject(true)}
                          className={`p-4 rounded-2xl border-2 transition-all flex flex-col gap-1 items-center text-center col-span-2 md:col-span-1 ${isRandomSubject ? 'border-amber-500 bg-amber-50 shadow-md' : 'border-slate-100 bg-slate-50 hover:border-amber-200'}`}
                        >
                          <span className="text-[10px] font-black uppercase text-slate-900">🎲 Random</span>
                          <span className="text-[8px] text-slate-400">AI picks country</span>
                        </button>
                      </div>
                    </CollapsibleSection>
                  </div>

                  {/* MCQ Section */}
                  <div className="bg-white p-8 rounded-[40px] border border-slate-100 shadow-sm">
                    <CollapsibleSection
                      title="Multiple Choice (MCQ) Grid"
                      subtitle="Configure layout, spacing, and option styles"
                      icon="fa-list-check"
                      iconBg="bg-orange-100"
                      iconColor="text-orange-600"
                      isCollapsed={!!collapsedSections['mcq_grid']}
                      onToggle={() => setCollapsedSections(prev => ({ ...prev, mcq_grid: !prev.mcq_grid }))}
                      rightElement={
                        <button 
                          onClick={() => {
                            setDesignTargetTypeId('mcq');
                            setSettingsTab('FORMAT_DESIGN');
                            setShowSettings(true);
                          }}
                          className="px-4 py-2 bg-orange-600 text-white rounded-xl text-[10px] font-black uppercase tracking-widest hover:bg-orange-700 transition-all flex items-center gap-2"
                        >
                          <i className="fa-solid fa-plus"></i> Add NEW
                        </button>
                      }
                    >
                      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mt-8">
                      <div className="space-y-4">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Option Style</label>
                        <div className="grid grid-cols-3 gap-2 max-h-[250px] overflow-y-auto p-2 bg-slate-50 rounded-2xl no-scrollbar border border-slate-100">
                          {[
                            { id: 0, label: 'None', icon: 'fa-font' },
                            { id: 1, label: 'Round', icon: 'fa-circle-dot' },
                            { id: 2, label: 'Box', icon: 'fa-square' },
                            { id: 3, label: 'Paren', icon: 'fa-brackets-round' },
                            { id: 4, label: 'Under', icon: 'fa-underline' },
                            { id: 5, label: 'Bold', icon: 'fa-bold' },
                            { id: 6, label: 'Diamond', icon: 'fa-diamond' },
                            { id: 7, label: 'Bracket', icon: 'fa-brackets-square' },
                            { id: 8, label: 'Circle Fill', icon: 'fa-circle' },
                            { id: 9, label: 'Square Fill', icon: 'fa-square-full' },
                            { id: 10, label: 'Double Paren', icon: 'fa-angles-left' },
                            { id: 11, label: 'Roman', icon: 'fa-i' },
                            { id: 12, label: 'Lower', icon: 'fa-a' },
                            { id: 13, label: 'Dotted', icon: 'fa-ellipsis' },
                            { id: 14, label: 'Arrow', icon: 'fa-arrow-right' },
                            { id: 15, label: 'Check', icon: 'fa-check' },
                            { id: 16, label: 'Star', icon: 'fa-star' },
                            { id: 17, label: 'Hexagon', icon: 'fa-hexagon' },
                            { id: 18, label: 'Triangle', icon: 'fa-caret-up' },
                            { id: 19, label: 'Plus', icon: 'fa-plus' }
                          ].map((s) => (
                            <button
                              key={s.id}
                              onClick={() => setMcqStyle(s.id)}
                              className={`h-12 rounded-2xl border flex flex-col items-center justify-center gap-1 transition-all ${mcqStyle === s.id ? 'bg-blue-600 border-blue-600 text-white shadow-lg' : 'bg-white border-slate-100 text-slate-400 hover:border-blue-200'}`}
                            >
                              <i className={`fa-solid ${s.icon} text-[10px]`}></i>
                              <span className="text-[8px] font-bold uppercase">{s.label}</span>
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="space-y-4">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Grid Layout</label>
                        <div className="grid grid-cols-1 gap-2">
                          {[
                            { id: 'single', label: '1 Option Per Line', icon: 'fa-grip-lines' },
                            { id: 'double', label: '2 Options Per Line', icon: 'fa-grip-lines-vertical' },
                            { id: 'quad', label: '4 Options Per Line', icon: 'fa-grip' }
                          ].map((layout) => (
                            <button
                              key={layout.id}
                              onClick={() => setMcqLayout(layout.id as any)}
                              className={`h-12 px-6 rounded-2xl border flex items-center gap-4 transition-all ${mcqLayout === layout.id ? 'bg-blue-600 border-blue-600 text-white shadow-lg' : 'bg-slate-50 border-slate-100 text-slate-400 hover:border-blue-200'}`}
                            >
                              <i className={`fa-solid ${layout.icon} text-[12px]`}></i>
                              <span className="text-[10px] font-black uppercase tracking-wider">{layout.label}</span>
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="space-y-4">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Vertical Spacing</label>
                        <div className="flex bg-slate-100 p-1.5 rounded-2xl gap-1">
                          {[
                            { id: 'none', label: 'Compact' },
                            { id: 'one', label: 'Spacious' }
                          ].map((s) => (
                            <button 
                              key={s.id} 
                              onClick={() => setMcqSpacing(s.id as any)} 
                              className={`flex-1 py-3 rounded-xl text-[10px] font-black uppercase transition-all ${mcqSpacing === s.id ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-400'}`}
                            >
                              {s.label}
                            </button>
                          ))}
                        </div>
                        <div className="bg-slate-50 p-4 rounded-2xl border border-slate-100">
                          <p className="text-[10px] font-bold text-slate-900 mb-2">Preview:</p>
                          <div className="space-y-1">
                            <p className="text-[8px] font-bold text-slate-900">1. The chef ____ the soup.</p>
                            <div className={`grid ${mcqLayout === 'single' ? 'grid-cols-1' : mcqLayout === 'double' ? 'grid-cols-2' : 'grid-cols-4'} gap-1`}>
                              {['A', 'B', 'C', 'D'].map(opt => (
                                <div key={opt} className="text-[7px] text-slate-500 flex items-center gap-1">
                                  <span className={
                                    mcqStyle === 1 ? 'rounded-full border border-slate-300 px-1' : 
                                    mcqStyle === 2 ? 'border border-slate-300 px-1' : 
                                    mcqStyle === 3 ? '('+opt+')' : 
                                    mcqStyle === 4 ? 'underline' : 
                                    mcqStyle === 5 ? 'font-bold' : 
                                    mcqStyle === 6 ? 'rotate-45 border border-slate-300 px-1 inline-block' :
                                    mcqStyle === 7 ? '['+opt+']' :
                                    mcqStyle === 8 ? 'rounded-full bg-slate-900 text-white px-1' :
                                    mcqStyle === 9 ? 'bg-slate-900 text-white px-1' :
                                    mcqStyle === 10 ? '(('+opt+'))' :
                                    ''
                                  }>
                                    {mcqStyle === 3 ? `(${opt})` : 
                                     mcqStyle === 7 ? `[${opt}]` : 
                                     mcqStyle === 10 ? `((${opt}))` : 
                                     mcqStyle === 6 ? <span className="-rotate-45 inline-block">{opt}</span> :
                                     `${opt}.`}
                                  </span>
                                  <span>Option</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Custom MCQ Designs */}
                    {(customDesigns || []).filter(d => d.type === 'mcq').length > 0 && (
                      <div className="mt-8 pt-8 border-t border-slate-100">
                        <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-4">Custom MCQ Designs</h4>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                          {customDesigns.filter(d => d.type === 'mcq').map(design => (
                            <div 
                              key={design.id}
                              onClick={() => {
                                setPaperStyles(prev => ({ ...prev, mcq: design.id }));
                                alert(`Applied custom MCQ design: ${design.name}`);
                              }}
                              className={`bg-white p-6 rounded-[32px] border-2 transition-all cursor-pointer group ${paperStyles.mcq === design.id ? 'border-orange-500 shadow-md' : 'border-slate-200 hover:border-orange-300 shadow-sm'}`}
                            >
                              <div className="flex items-center gap-3 mb-4">
                                <div className={`h-8 w-8 rounded-xl flex items-center justify-center transition-all ${paperStyles.mcq === design.id ? 'bg-orange-600 text-white' : 'bg-slate-50 text-slate-400 group-hover:bg-orange-100 group-hover:text-orange-600'}`}>
                                  <i className="fa-solid fa-wand-magic-sparkles"></i>
                                </div>
                                <span className="text-[11px] font-black uppercase text-slate-900">{design.name}</span>
                              </div>
                              <p className="text-[9px] text-slate-400 font-medium mb-4">Custom formatting and AI instructions applied.</p>
                              <div className="h-24 bg-slate-50 rounded-2xl border border-dashed border-slate-200 flex items-center justify-center">
                                <div className="text-[10px] font-bold text-slate-300 uppercase">Custom Design</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </CollapsibleSection>
                </div>

                {/* Popular Paper Styles Section */}
                <div className="bg-white p-8 rounded-[40px] border border-slate-100 shadow-sm">
                  <CollapsibleSection
                    title="Popular Paper Styles"
                    subtitle="Choose a base layout for your test paper. Selecting a style sets it as your default."
                    icon="fa-file-lines"
                    iconBg="bg-emerald-100"
                    iconColor="text-emerald-600"
                    isCollapsed={!!collapsedSections['paper_styles']}
                    onToggle={() => setCollapsedSections(prev => ({ ...prev, paper_styles: !prev.paper_styles }))}
                  >
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
                      {[
                        { id: 0, name: 'Option 1: Clean White', desc: 'Pure white paper with subtle border.', icon: 'fa-file-lines' },
                        { id: 1, name: 'Option 2: Orange Mix', desc: 'White paper with orange accents.', icon: 'fa-palette' },
                        { id: 2, name: 'Option 3: Modern Emerald', desc: 'Professional green theme.', icon: 'fa-gem' },
                        { id: 3, name: 'Option 4: Soft Lavender', desc: 'Elegant purple theme.', icon: 'fa-feather' },
                        { id: 4, name: 'Option 5: Mint', desc: 'Very light green paper.', icon: 'fa-leaf' },
                        { id: 5, name: 'Option 6: Peach', desc: 'Very light orange paper.', icon: 'fa-sun' },
                        { id: 6, name: 'Option 7: Sky', desc: 'Very light blue paper.', icon: 'fa-cloud' },
                        { id: 7, name: 'Option 8: Lavender', desc: 'Very light purple paper.', icon: 'fa-moon' },
                        { id: 8, name: 'Option 9: Citrus', desc: 'Light Green & Orange mix.', icon: 'fa-lemon' },
                        { id: 9, name: 'Option 10: Rose', desc: 'Very light pink paper.', icon: 'fa-heart' },
                        { id: 10, name: 'Option 11: Stars', desc: 'Decorative stars background.', icon: 'fa-star' },
                        { id: 11, name: 'Option 12: Flowers', desc: 'Decorative flowers background.', icon: 'fa-flower' },
                        { id: 12, name: 'Option 13: Hearts', desc: 'Decorative hearts background.', icon: 'fa-heart' },
                        { id: 13, name: 'Option 14: Bubbles', desc: 'Decorative bubbles background.', icon: 'fa-soap' },
                        { id: 14, name: 'Option 15: Leaves', desc: 'Decorative leaves background.', icon: 'fa-leaf' },
                        { id: 15, name: 'Option 16: Rainbow', desc: 'Subtle rainbow gradient.', icon: 'fa-rainbow' },
                        { id: 16, name: 'Option 17: Galaxy', desc: 'Dark galaxy themed paper.', icon: 'fa-user-astronaut' },
                        { id: 17, name: 'Option 18: Notebook', desc: 'Classic spiral notebook style.', icon: 'fa-book-open' },
                        { id: 18, name: 'Option 19: Vintage', desc: 'Aged parchment style.', icon: 'fa-scroll' },
                        { id: 19, name: 'Option 20: Modern', desc: 'Geometric modern art style.', icon: 'fa-shapes' },
                      ].map((style) => (
                        <div 
                          key={style.id}
                          onClick={() => {
                            setGlobalLayout(style.id);
                            if (style.id === 3) {
                              setMcqLayout('quad');
                            }
                          }}
                          className={`p-8 rounded-[40px] border-2 cursor-pointer transition-all ${globalLayout === style.id ? 'border-emerald-500 bg-emerald-50/30 shadow-xl scale-[1.02]' : 'border-slate-100 bg-white hover:border-emerald-200 shadow-sm'}`}
                        >
                          <div className="flex justify-between items-start mb-6">
                            <div className={`h-14 w-14 rounded-2xl flex items-center justify-center text-2xl ${globalLayout === style.id ? 'bg-emerald-500 text-white' : 'bg-slate-100 text-slate-400'}`}>
                              <i className={`fa-solid ${style.icon}`}></i>
                            </div>
                            {globalLayout === style.id && <div className="h-8 w-8 bg-emerald-500 text-white rounded-full flex items-center justify-center shadow-lg animate-in zoom-in"><i className="fa-solid fa-check"></i></div>}
                          </div>
                          <h4 className="text-lg font-black text-slate-900 uppercase tracking-tight mb-2">{style.name}</h4>
                          <p className="text-xs font-medium text-slate-500 leading-relaxed">{style.desc}</p>
                        </div>
                      ))}
                    </div>
                  </CollapsibleSection>
                </div>

                {/* Table Styles Section */}
                <div className="bg-white p-8 rounded-[40px] border border-slate-100 shadow-sm">
                  <CollapsibleSection
                    title="Table Styles"
                    subtitle="Choose the visual style for tables and columns in your exercises."
                    icon="fa-table"
                    iconBg="bg-purple-100"
                    iconColor="text-purple-600"
                    isCollapsed={!!collapsedSections['table_styles']}
                    onToggle={() => setCollapsedSections(prev => ({ ...prev, table_styles: !prev.table_styles }))}
                    rightElement={
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          setDesignTargetTypeId('table_style');
                          setEditingCustomDesignId(null);
                          setSettingsTab('FORMAT_DESIGN');
                          setShowSettings(true);
                        }}
                        className="px-4 py-2 bg-purple-600 text-white rounded-xl text-[10px] font-black uppercase tracking-widest hover:bg-purple-700 transition-all flex items-center gap-2"
                      >
                        <i className="fa-solid fa-plus"></i> Add NEW
                      </button>
                    }
                  >
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-8">
                      {[
                        { id: 'plain', name: 'Plain Tables', desc: 'Clean layout with minimal borders.', icon: 'fa-table-cells-large' },
                        { id: 'grid', name: 'Grid Tables', desc: 'Standard grid with all borders visible.', icon: 'fa-table-cells' },
                        { id: 'list', name: 'List Tables', desc: 'Horizontal lines only, list style.', icon: 'fa-table-list' },
                      ].map((style) => (
                        <div 
                          key={style.id}
                          onClick={() => setTableStyle(style.id)}
                          className={`p-8 rounded-[40px] border-2 cursor-pointer transition-all ${tableStyle === style.id ? 'border-purple-500 bg-purple-50/30 shadow-xl scale-[1.02]' : 'border-slate-100 bg-white hover:border-purple-200 shadow-sm'}`}
                        >
                          <div className="flex justify-between items-start mb-6">
                            <div className={`h-14 w-14 rounded-2xl flex items-center justify-center text-2xl ${tableStyle === style.id ? 'bg-purple-500 text-white' : 'bg-slate-100 text-slate-400'}`}>
                              <i className={`fa-solid ${style.icon}`}></i>
                            </div>
                            {tableStyle === style.id && <div className="h-8 w-8 bg-purple-500 text-white rounded-full flex items-center justify-center shadow-lg animate-in zoom-in"><i className="fa-solid fa-check"></i></div>}
                          </div>
                          <h4 className="text-lg font-black text-slate-900 uppercase tracking-tight mb-2">{style.name}</h4>
                          <p className="text-xs font-medium text-slate-500 leading-relaxed">{style.desc}</p>
                        </div>
                      ))}
                      
                      {/* Custom Table Styles */}
                      {customDesigns.filter(d => d.type === 'table_style').map(design => (
                        <div 
                          key={design.id}
                          onClick={() => setTableStyle(design.id)}
                          className={`p-8 rounded-[40px] border-2 cursor-pointer transition-all relative group ${tableStyle === design.id ? 'border-purple-500 bg-purple-50/30 shadow-xl scale-[1.02]' : 'border-slate-100 bg-white hover:border-purple-200 shadow-sm'}`}
                        >
                          <div className="flex justify-between items-start mb-6">
                            <div className={`h-14 w-14 rounded-2xl flex items-center justify-center text-2xl ${tableStyle === design.id ? 'bg-purple-500 text-white' : 'bg-slate-100 text-slate-400'}`}>
                              <i className="fa-solid fa-table-columns"></i>
                            </div>
                            <div className="flex gap-2">
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setEditingCustomDesignId(design.id);
                                  setSettingsTab('FORMAT_DESIGN');
                                  setShowSettings(true);
                                }}
                                className="h-8 w-8 bg-white text-slate-400 rounded-full flex items-center justify-center shadow-sm hover:text-blue-600 hover:bg-blue-50 transition-all opacity-0 group-hover:opacity-100"
                              >
                                <i className="fa-solid fa-pen text-xs"></i>
                              </button>
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  if (confirm('Delete this custom table style?')) {
                                    const newDesigns = customDesigns.filter(d => d.id !== design.id);
                                    setCustomDesigns(newDesigns);
                                    localStorage.setItem('dp_custom_designs_v46', JSON.stringify(newDesigns));
                                    if (tableStyle === design.id) setTableStyle('plain');
                                  }
                                }}
                                className="h-8 w-8 bg-white text-slate-400 rounded-full flex items-center justify-center shadow-sm hover:text-red-600 hover:bg-red-50 transition-all opacity-0 group-hover:opacity-100"
                              >
                                <i className="fa-solid fa-trash text-xs"></i>
                              </button>
                              {tableStyle === design.id && <div className="h-8 w-8 bg-purple-500 text-white rounded-full flex items-center justify-center shadow-lg animate-in zoom-in"><i className="fa-solid fa-check"></i></div>}
                            </div>
                          </div>
                          <h4 className="text-lg font-black text-slate-900 uppercase tracking-tight mb-2">Custom: {design.name}</h4>
                          <p className="text-xs font-medium text-slate-500 leading-relaxed italic">Custom formatting applied.</p>
                        </div>
                      ))}
                    </div>
                  </CollapsibleSection>
                </div>

                {/* Header & Footer Styles Section */}
                  <div className="bg-white p-8 rounded-[40px] border border-slate-100 shadow-sm">
                    <CollapsibleSection
                      title="Header & Footer Styles"
                      subtitle="Choose from 20 professional header layouts with custom colors"
                      icon="fa-heading"
                      iconBg="bg-blue-100"
                      iconColor="text-blue-600"
                      isCollapsed={!!collapsedSections['header_footer_styles']}
                      onToggle={() => setCollapsedSections(prev => ({ ...prev, header_footer_styles: !prev.header_footer_styles }))}
                    >
                      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4 max-h-[500px] overflow-y-auto p-6 bg-slate-50 rounded-[40px] no-scrollbar border border-slate-100 mt-8">
                      {[
                        { color: 'bg-slate-900', accent: 'bg-slate-400' },
                        { color: 'bg-blue-600', accent: 'bg-blue-200' },
                        { color: 'bg-emerald-600', accent: 'bg-emerald-200' },
                        { color: 'bg-rose-600', accent: 'bg-rose-200' },
                        { color: 'bg-amber-600', accent: 'bg-amber-200' },
                        { color: 'bg-indigo-600', accent: 'bg-indigo-200' },
                        { color: 'bg-purple-600', accent: 'bg-purple-200' },
                        { color: 'bg-teal-600', accent: 'bg-teal-200' },
                        { color: 'bg-orange-600', accent: 'bg-orange-200' },
                        { color: 'bg-cyan-600', accent: 'bg-cyan-200' },
                        { color: 'bg-fuchsia-600', accent: 'bg-fuchsia-200' },
                        { color: 'bg-pink-600', accent: 'bg-pink-200' },
                        { color: 'bg-lime-600', accent: 'bg-lime-200' },
                        { color: 'bg-violet-600', accent: 'bg-violet-200' },
                        { color: 'bg-sky-600', accent: 'bg-sky-200' },
                        { color: 'bg-red-600', accent: 'bg-red-200' },
                        { color: 'bg-zinc-600', accent: 'bg-zinc-200' },
                        { color: 'bg-neutral-900', accent: 'bg-neutral-400' },
                        { color: 'bg-stone-600', accent: 'bg-stone-200' },
                        { color: 'bg-slate-700', accent: 'bg-slate-300' }
                      ].map((style, i) => (
                        <button
                          key={i}
                          onClick={() => setBrandSettings({ ...brandSettings, headerStyle: i })}
                          className={`aspect-video rounded-2xl border-2 transition-all flex flex-col items-center justify-center gap-2 group ${brandSettings.headerStyle === i ? 'border-blue-600 bg-white shadow-lg' : 'border-slate-100 bg-white hover:border-blue-200'}`}
                        >
                          <div className={`w-12 h-1.5 ${style.color} rounded-full group-hover:scale-110 transition-transform`}></div>
                          <div className="flex gap-1">
                            <div className={`w-4 h-1 ${style.accent} rounded-full`}></div>
                            <div className={`w-4 h-1 ${style.accent} rounded-full`}></div>
                          </div>
                          <span className="text-[8px] font-black text-slate-400 uppercase">Style {i + 1}</span>
                        </button>
                      ))}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 pt-4">
                      <div className="space-y-4">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Main Title (School Name)</label>
                        <input value={brandSettings.schoolName} onChange={e => setBrandSettings({ ...brandSettings, schoolName: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-blue-500 font-bold text-slate-700" placeholder="e.g. HARVARD ACADEMY" />
                      </div>
                      <div className="space-y-4">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Footer Text</label>
                        <input value={brandSettings.footerText} onChange={e => setBrandSettings({ ...brandSettings, footerText: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-blue-500 font-bold text-slate-700" placeholder="e.g. Confidential - Academic Use Only" />
                      </div>
                    </div>
                  </CollapsibleSection>
                </div>

                {/* Matching Section */}
                <div className="bg-white p-8 rounded-[40px] border border-slate-100 shadow-sm">
                  <CollapsibleSection
                    title="Matching Exercise Styles"
                    subtitle="Choose your preferred matching layout"
                    icon="fa-arrows-left-right"
                    iconBg="bg-indigo-100"
                    iconColor="text-indigo-600"
                    isCollapsed={!!collapsedSections['matching_styles']}
                    onToggle={() => setCollapsedSections(prev => ({ ...prev, matching_styles: !prev.matching_styles }))}
                    rightElement={
                      <button 
                        onClick={() => {
                          setDesignTargetTypeId('matching');
                          setSettingsTab('FORMAT_DESIGN');
                          setShowSettings(true);
                        }}
                        className="px-4 py-2 bg-indigo-600 text-white rounded-xl text-[10px] font-black uppercase tracking-widest hover:bg-indigo-700 transition-all flex items-center gap-2"
                      >
                        <i className="fa-solid fa-plus"></i> Add NEW
                      </button>
                    }
                  >
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
                      {[
                        { id: 'classic', label: 'Classic Matching', icon: 'fa-list-ol', desc: 'Numbers on left, letters on right with blanks.' },
                        { id: 'boxed', label: 'Boxed Answers', icon: 'fa-square', desc: 'Square boxes for letters, word bank at bottom.' },
                        { id: 'column', label: 'Column A/B', icon: 'fa-columns', desc: 'Formal A/B comparison with vertical divider.' }
                      ].map(style => (
                        <div 
                          key={style.id} 
                          onClick={() => setPaperStyles(prev => ({ ...prev, matching: style.id }))}
                          className={`bg-white p-6 rounded-[32px] border-2 transition-all cursor-pointer group ${paperStyles.matching === style.id ? 'border-indigo-500 shadow-md' : 'border-slate-200 hover:border-indigo-300 shadow-sm'}`}
                        >
                          <div className="flex items-center gap-3 mb-4">
                            <div className={`h-8 w-8 rounded-xl flex items-center justify-center transition-all ${paperStyles.matching === style.id ? 'bg-indigo-600 text-white' : 'bg-slate-50 text-slate-400 group-hover:bg-indigo-100 group-hover:text-indigo-600'}`}>
                              <i className={`fa-solid ${style.icon}`}></i>
                            </div>
                            <span className="text-[11px] font-black uppercase text-slate-900">{style.label}</span>
                          </div>
                          <p className="text-[9px] text-slate-400 font-medium mb-4">{style.desc}</p>
                          <div className="h-24 bg-slate-50 rounded-2xl border border-dashed border-slate-200 flex items-center justify-center">
                            <div className="text-[10px] font-bold text-slate-300 uppercase">Preview Area</div>
                          </div>
                        </div>
                      ))}
                      {/* Custom Matching Designs */}
                      {customDesigns.filter(d => d.type === 'matching').map(design => (
                        <div 
                          key={design.id}
                          onClick={() => {
                            setPaperStyles(prev => ({ ...prev, matching: design.id }));
                            alert(`Applied custom matching design: ${design.name}`);
                          }}
                          className={`bg-white p-6 rounded-[32px] border-2 transition-all cursor-pointer group ${paperStyles.matching === design.id ? 'border-indigo-500 shadow-md' : 'border-slate-200 hover:border-indigo-300 shadow-sm'}`}
                        >
                          <div className="flex items-center gap-3 mb-4">
                            <div className={`h-8 w-8 rounded-xl flex items-center justify-center transition-all ${paperStyles.matching === design.id ? 'bg-indigo-600 text-white' : 'bg-slate-50 text-slate-400 group-hover:bg-indigo-100 group-hover:text-indigo-600'}`}>
                              <i className="fa-solid fa-wand-magic-sparkles"></i>
                            </div>
                            <span className="text-[11px] font-black uppercase text-slate-900">{design.name}</span>
                          </div>
                          <p className="text-[9px] text-slate-400 font-medium mb-4">Custom formatting and AI instructions applied.</p>
                          <div className="h-24 bg-slate-50 rounded-2xl border border-dashed border-slate-200 flex items-center justify-center">
                            <div className="text-[10px] font-bold text-slate-300 uppercase">Custom Design</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CollapsibleSection>
                </div>

                {/* Other Exercise Types */}
                <div className="bg-white p-8 rounded-[40px] border border-slate-100 shadow-sm">
                  <CollapsibleSection
                    title="Other Exercise Types"
                    subtitle="Configure Correct/Incorrect and True/False layouts"
                    icon="fa-circle-check"
                    iconBg="bg-emerald-100"
                    iconColor="text-emerald-600"
                    isCollapsed={!!collapsedSections['other_exercise_types']}
                    onToggle={() => setCollapsedSections(prev => ({ ...prev, other_exercise_types: !prev.other_exercise_types }))}
                  >
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
                      {/* Correct/Incorrect */}
                      <div className="bg-slate-50 p-8 rounded-[40px] border border-slate-100 shadow-sm space-y-6">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4">
                            <div className="h-10 w-10 bg-emerald-100 rounded-2xl flex items-center justify-center text-emerald-600">
                              <i className="fa-solid fa-circle-check"></i>
                            </div>
                            <h4 className="text-[12px] font-black text-slate-900 uppercase">Correct / Incorrect</h4>
                          </div>
                          <button 
                            onClick={() => {
                              setDesignTargetTypeId('correct_incorrect');
                              setSettingsTab('FORMAT_DESIGN');
                              setShowSettings(true);
                            }}
                            className="h-8 w-8 bg-emerald-600 text-white rounded-lg flex items-center justify-center hover:bg-emerald-700 transition-all"
                            title="Add New Style"
                          >
                            <i className="fa-solid fa-plus text-[10px]"></i>
                          </button>
                        </div>
                        <div className="space-y-4">
                          <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Table Columns</label>
                          <div className="flex bg-slate-100 p-1.5 rounded-2xl gap-1">
                            {[1, 2, 3].map(cols => (
                              <button key={cols} onClick={() => setPaperStyles(prev => ({ ...prev, correctIncorrect: cols }))} className={`flex-1 py-3 rounded-xl text-[10px] font-black transition-all ${paperStyles.correctIncorrect === cols ? 'bg-white text-emerald-600 shadow-sm' : 'text-slate-400'}`}>{cols} Column{cols > 1 ? 's' : ''}</button>
                            ))}
                          </div>
                        </div>
                      </div>

                      {/* True / False */}
                      <div className="bg-slate-50 p-8 rounded-[40px] border border-slate-100 shadow-sm space-y-6">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4">
                            <div className="h-10 w-10 bg-blue-100 rounded-2xl flex items-center justify-center text-blue-600">
                              <i className="fa-solid fa-square-check"></i>
                            </div>
                            <h4 className="text-[12px] font-black text-slate-900 uppercase">True / False</h4>
                          </div>
                          <button 
                            onClick={() => {
                              setDesignTargetTypeId('true_false');
                              setSettingsTab('FORMAT_DESIGN');
                              setShowSettings(true);
                            }}
                            className="h-8 w-8 bg-blue-600 text-white rounded-lg flex items-center justify-center hover:bg-blue-700 transition-all"
                            title="Add New Style"
                          >
                            <i className="fa-solid fa-plus text-[10px]"></i>
                          </button>
                        </div>
                        <div className="space-y-4">
                          <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Layout Format</label>
                          <div className="flex bg-slate-100 p-1.5 rounded-2xl gap-1">
                            {[0, 1, 2].map(style => (
                              <button key={style} onClick={() => setPaperStyles(prev => ({ ...prev, tf: style }))} className={`flex-1 py-3 rounded-xl text-[10px] font-black transition-all ${paperStyles.tf === style ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-400'}`}>
                                {style === 0 ? 'Standard' : style === 1 ? 'Boxed' : 'Underlined'}
                              </button>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </CollapsibleSection>
                </div>

                  {/* Custom Exercise Types Library removed as per user request */}


                  {/* True or False Design Samples */}
                  <div className="space-y-6 border-t border-slate-100 pt-8">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="h-10 w-10 bg-blue-100 text-blue-600 rounded-xl flex items-center justify-center">
                        <i className="fa-solid fa-toggle-on"></i>
                      </div>
                      <h4 className="text-lg font-bold text-slate-800 uppercase tracking-widest">True or False Designs</h4>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* T/F Design 1 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, tf: 0 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.tf === 0 ? 'border-blue-500 bg-blue-50/30 shadow-md' : 'border-slate-200 hover:border-blue-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 1: ( T / F ) at End</h5>
                          {paperStyles.tf === 0 && <div className="h-6 w-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-3">
                          <div className="flex justify-between"><span>1. The earth is flat.</span><span className="font-bold text-blue-600">( T / F )</span></div>
                          <div className="flex justify-between"><span>2. Water boils at 100 degrees Celsius.</span><span className="font-bold text-blue-600">( T / F )</span></div>
                          <div className="flex justify-between"><span>3. The sun rises in the west.</span><span className="font-bold text-blue-600">( T / F )</span></div>
                        </div>
                      </div>

                      {/* T/F Design 2 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, tf: 1 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.tf === 1 ? 'border-blue-500 bg-blue-50/30 shadow-md' : 'border-slate-200 hover:border-blue-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 2: Underscore Prefix</h5>
                          {paperStyles.tf === 1 && <div className="h-6 w-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-3">
                          <div><span className="font-bold text-blue-600">____</span> 1. The earth is flat.</div>
                          <div><span className="font-bold text-blue-600">____</span> 2. Water boils at 100 degrees Celsius.</div>
                          <div><span className="font-bold text-blue-600">____</span> 3. The sun rises in the west.</div>
                        </div>
                      </div>

                      {/* T/F Design 3 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, tf: 2 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.tf === 2 ? 'border-blue-500 bg-blue-50/30 shadow-md' : 'border-slate-200 hover:border-blue-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 3: Checkbox Prefix</h5>
                          {paperStyles.tf === 2 && <div className="h-6 w-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-3">
                          <div><span className="font-bold text-blue-600">[ ]</span> 1. The earth is flat.</div>
                          <div><span className="font-bold text-blue-600">[ ]</span> 2. Water boils at 100 degrees Celsius.</div>
                          <div><span className="font-bold text-blue-600">[ ]</span> 3. The sun rises in the west.</div>
                        </div>
                      </div>

                      {/* T/F Design 4 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, tf: 3 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.tf === 3 ? 'border-blue-500 bg-blue-50/30 shadow-md' : 'border-slate-200 hover:border-blue-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 4: True / False Labels</h5>
                          {paperStyles.tf === 3 && <div className="h-6 w-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-4">
                          <div>
                            <div>1. The earth is flat.</div>
                            <div className="ml-4 mt-1 font-bold text-blue-600 text-xs">True / False</div>
                          </div>
                          <div>
                            <div>2. Water boils at 100 degrees Celsius.</div>
                            <div className="ml-4 mt-1 font-bold text-blue-600 text-xs">True / False</div>
                          </div>
                          <div>
                            <div>3. The sun rises in the west.</div>
                            <div className="ml-4 mt-1 font-bold text-blue-600 text-xs">True / False</div>
                          </div>
                        </div>
                      </div>

                      {/* Custom T/F Designs */}
                      {customDesigns.filter(d => d.type === 'true_false' || d.type === 'tf').map(design => (
                        <div 
                          key={design.id}
                          onClick={() => {
                            setPaperStyles(prev => ({ ...prev, tf: design.id }));
                          }}
                          className={`p-6 rounded-2xl border-2 cursor-pointer transition-all group relative ${paperStyles.tf === design.id ? 'border-blue-500 bg-blue-50/30 shadow-md' : 'border-slate-200 hover:border-blue-300 bg-white'}`}
                        >
                          <div className="flex justify-between items-center mb-4">
                            <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Custom: {design.name}</h5>
                            <div className="flex gap-1">
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const newName = prompt("Enter new name for this style:", design.name);
                                  if (newName) {
                                    setCustomDesigns(prev => prev.map(d => d.id === design.id ? { ...d, name: newName } : d));
                                  }
                                }}
                                className="h-6 w-6 text-slate-300 hover:text-blue-500 transition-colors"
                              >
                                <i className="fa-solid fa-pen text-[10px]"></i>
                              </button>
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteCustomDesign(design.id);
                                }}
                                className="h-6 w-6 text-slate-300 hover:text-rose-500 transition-colors"
                              >
                                <i className="fa-solid fa-trash text-[10px]"></i>
                              </button>
                            </div>
                          </div>
                          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-[10px] text-slate-400 italic">
                            Custom formatting and AI instructions applied.
                          </div>
                          {paperStyles.tf === design.id && <div className="absolute top-2 right-2 h-6 w-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs shadow-sm"><i className="fa-solid fa-check"></i></div>}
                        </div>
                      ))}

                      {/* Add New T/F Style Card */}
                      <div 
                        onClick={() => {
                          setDesignTargetTypeId('true_false');
                          setSettingsTab('FORMAT_DESIGN');
                          setShowSettings(true);
                        }}
                        className="p-6 rounded-2xl border-2 border-dashed border-slate-200 hover:border-blue-400 hover:bg-blue-50/30 cursor-pointer transition-all flex flex-col items-center justify-center gap-3 group"
                      >
                        <div className="h-12 w-12 bg-slate-100 text-slate-400 rounded-full flex items-center justify-center group-hover:bg-blue-100 group-hover:text-blue-600 transition-all">
                          <i className="fa-solid fa-plus text-xl"></i>
                        </div>
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest group-hover:text-blue-600">Add New Style</span>
                      </div>
                    </div>
                  </div>

                  {/* Correct or Incorrect Design Samples */}
                  <div className="space-y-6 border-t border-slate-100 pt-8">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="h-10 w-10 bg-emerald-100 text-emerald-600 rounded-xl flex items-center justify-center">
                        <i className="fa-solid fa-circle-check"></i>
                      </div>
                      <h4 className="text-lg font-bold text-slate-800 uppercase tracking-widest">Correct or Incorrect Designs</h4>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* C/I Design 1 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, correctIncorrect: 0 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.correctIncorrect === 0 ? 'border-emerald-500 bg-emerald-50/30 shadow-md' : 'border-slate-200 hover:border-emerald-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 1: Checkbox Prefix</h5>
                          {paperStyles.correctIncorrect === 0 && <div className="h-6 w-6 bg-emerald-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-3">
                          <div><span className="font-bold text-emerald-600">[ ]</span> 1. She don't like apples.</div>
                          <div><span className="font-bold text-emerald-600">[ ]</span> 2. He goes to school every day.</div>
                          <div><span className="font-bold text-emerald-600">[ ]</span> 3. They is playing football.</div>
                        </div>
                      </div>

                      {/* C/I Design 2 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, correctIncorrect: 1 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.correctIncorrect === 1 ? 'border-emerald-500 bg-emerald-50/30 shadow-md' : 'border-slate-200 hover:border-emerald-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 2: ( C / I ) at End</h5>
                          {paperStyles.correctIncorrect === 1 && <div className="h-6 w-6 bg-emerald-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-3">
                          <div className="flex justify-between"><span>1. She don't like apples.</span><span className="font-bold text-emerald-600">( C / I )</span></div>
                          <div className="flex justify-between"><span>2. He goes to school every day.</span><span className="font-bold text-emerald-600">( C / I )</span></div>
                          <div className="flex justify-between"><span>3. They is playing football.</span><span className="font-bold text-emerald-600">( C / I )</span></div>
                        </div>
                      </div>

                      {/* C/I Design 3 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, correctIncorrect: 2 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.correctIncorrect === 2 ? 'border-emerald-500 bg-emerald-50/30 shadow-md' : 'border-slate-200 hover:border-emerald-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 3: Correct / Incorrect Labels</h5>
                          {paperStyles.correctIncorrect === 2 && <div className="h-6 w-6 bg-emerald-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-4">
                          <div>
                            <div>1. She don't like apples.</div>
                            <div className="ml-4 mt-1 font-bold text-emerald-600 text-xs">Correct / Incorrect</div>
                          </div>
                          <div>
                            <div>2. He goes to school every day.</div>
                            <div className="ml-4 mt-1 font-bold text-emerald-600 text-xs">Correct / Incorrect</div>
                          </div>
                          <div>
                            <div>3. They is playing football.</div>
                            <div className="ml-4 mt-1 font-bold text-emerald-600 text-xs">Correct / Incorrect</div>
                          </div>
                        </div>
                      </div>

                      {/* C/I Design 4 - (C/I) at Beginning (Default) */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, correctIncorrect: 3 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.correctIncorrect === 3 ? 'border-emerald-500 bg-emerald-50/30 shadow-md' : 'border-slate-200 hover:border-emerald-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 4: ( C / I ) at Beginning</h5>
                          {paperStyles.correctIncorrect === 3 && <div className="h-6 w-6 bg-emerald-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-3">
                          <div className="flex gap-4"><span className="font-bold text-emerald-600 w-12">( C / I )</span><span>1. She don't like apples.</span></div>
                          <div className="flex gap-4"><span className="font-bold text-emerald-600 w-12">( C / I )</span><span>2. He goes to school every day.</span></div>
                          <div className="flex gap-4"><span className="font-bold text-emerald-600 w-12">( C / I )</span><span>3. They is playing football.</span></div>
                        </div>
                      </div>

                      {/* Custom C/I Designs */}
                      {customDesigns.filter(d => d.type === 'correct_incorrect' || d.type === 'correctIncorrect').map(design => (
                        <div 
                          key={design.id}
                          onClick={() => {
                            setPaperStyles(prev => ({ ...prev, correctIncorrect: design.id }));
                          }}
                          className={`p-6 rounded-2xl border-2 transition-all cursor-pointer group relative ${paperStyles.correctIncorrect === design.id ? 'border-emerald-500 bg-emerald-50/30 shadow-md' : 'border-slate-200 hover:border-emerald-300 bg-white'}`}
                        >
                          <div className="flex justify-between items-center mb-4">
                            <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Custom: {design.name}</h5>
                            <div className="flex gap-1">
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const newName = prompt("Enter new name for this style:", design.name);
                                  if (newName) {
                                    setCustomDesigns(prev => prev.map(d => d.id === design.id ? { ...d, name: newName } : d));
                                  }
                                }}
                                className="h-6 w-6 text-slate-300 hover:text-blue-500 transition-colors"
                              >
                                <i className="fa-solid fa-pen text-[10px]"></i>
                              </button>
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteCustomDesign(design.id);
                                }}
                                className="h-6 w-6 text-slate-300 hover:text-rose-500 transition-colors"
                              >
                                <i className="fa-solid fa-trash text-[10px]"></i>
                              </button>
                            </div>
                          </div>
                          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-[10px] text-slate-400 italic">
                            Custom formatting and AI instructions applied.
                          </div>
                          {paperStyles.correctIncorrect === design.id && <div className="absolute top-2 right-2 h-6 w-6 bg-emerald-500 text-white rounded-full flex items-center justify-center text-xs shadow-sm"><i className="fa-solid fa-check"></i></div>}
                        </div>
                      ))}

                      {/* Add New C/I Style Card */}
                      <div 
                        onClick={() => {
                          setDesignTargetTypeId('correct_incorrect');
                          setSettingsTab('FORMAT_DESIGN');
                          setShowSettings(true);
                        }}
                        className="p-6 rounded-2xl border-2 border-dashed border-slate-200 hover:border-emerald-400 hover:bg-emerald-50/30 cursor-pointer transition-all flex flex-col items-center justify-center gap-3 group"
                      >
                        <div className="h-12 w-12 bg-slate-100 text-slate-400 rounded-full flex items-center justify-center group-hover:bg-emerald-100 group-hover:text-emerald-600 transition-all">
                          <i className="fa-solid fa-plus text-xl"></i>
                        </div>
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest group-hover:text-emerald-600">Add New Style</span>
                      </div>
                    </div>
                  </div>

                  {/* Circle Design Samples */}
                  <div className="space-y-6 border-t border-slate-100 pt-8">
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 bg-purple-100 text-purple-600 rounded-xl flex items-center justify-center">
                          <i className="fa-solid fa-circle-dot"></i>
                        </div>
                        <h4 className="text-lg font-bold text-slate-800 uppercase tracking-widest">Circle Designs</h4>
                      </div>
                      <button 
                        onClick={() => {
                          setDesignTargetTypeId('circle');
                          setSettingsTab('FORMAT_DESIGN');
                          setShowSettings(true);
                        }}
                        className="px-4 py-2 bg-purple-600 text-white rounded-xl text-[10px] font-black uppercase tracking-widest hover:bg-purple-700 transition-all flex items-center gap-2"
                      >
                        <i className="fa-solid fa-plus"></i> Add NEW
                      </button>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Circle Design 1 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, circle: 0 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.circle === 0 ? 'border-purple-500 bg-purple-50/30 shadow-md' : 'border-slate-200 hover:border-purple-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 1: Standard Bold</h5>
                          {paperStyles.circle === 0 && <div className="h-6 w-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-3">
                          <div>1. I <span className="font-bold border border-slate-800 rounded-full px-2 py-0.5">am</span> / is a student.</div>
                          <div>2. She <span className="font-bold border border-slate-800 rounded-full px-2 py-0.5">likes</span> / like apples.</div>
                          <div>3. They are <span className="font-bold border border-slate-800 rounded-full px-2 py-0.5">playing</span> / play football.</div>
                        </div>
                      </div>

                      {/* Circle Design 2 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, circle: 1 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.circle === 1 ? 'border-purple-500 bg-purple-50/30 shadow-md' : 'border-slate-200 hover:border-purple-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 2: Underlined</h5>
                          {paperStyles.circle === 1 && <div className="h-6 w-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-3">
                          <div>1. I <span className="underline border border-slate-800 rounded-full px-2 py-0.5">am</span> / is a student.</div>
                          <div>2. She <span className="underline border border-slate-800 rounded-full px-2 py-0.5">likes</span> / like apples.</div>
                          <div>3. They are <span className="underline border border-slate-800 rounded-full px-2 py-0.5">playing</span> / play football.</div>
                        </div>
                      </div>

                      {/* Circle Design 3 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, circle: 2 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.circle === 2 ? 'border-purple-500 bg-purple-50/30 shadow-md' : 'border-slate-200 hover:border-purple-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 3: Italicized</h5>
                          {paperStyles.circle === 2 && <div className="h-6 w-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-3">
                          <div>1. I <span className="italic border border-slate-800 rounded-full px-2 py-0.5">am</span> / is a student.</div>
                          <div>2. She <span className="italic border border-slate-800 rounded-full px-2 py-0.5">likes</span> / like apples.</div>
                          <div>3. They are <span className="italic border border-slate-800 rounded-full px-2 py-0.5">playing</span> / play football.</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Sentence Completion Design Samples */}
                  <div className="space-y-6 border-t border-slate-100 pt-8">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="h-10 w-10 bg-indigo-100 text-indigo-600 rounded-xl flex items-center justify-center">
                        <i className="fa-solid fa-pen-to-square"></i>
                      </div>
                      <h4 className="text-lg font-bold text-slate-800 uppercase tracking-widest">Sentence Completion Designs</h4>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Sentence Completion Design 1 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, sentenceCompletion: 0 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.sentenceCompletion === 0 ? 'border-indigo-500 bg-indigo-50/30 shadow-md' : 'border-slate-200 hover:border-indigo-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 1: Standard Blank</h5>
                          {paperStyles.sentenceCompletion === 0 && <div className="h-6 w-6 bg-indigo-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-3">
                          <div>1. The cat is sleeping on the _______________.</div>
                          <div>2. I need to buy some _______________ from the store.</div>
                        </div>
                      </div>

                      {/* Sentence Completion Design 2 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, sentenceCompletion: 1 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.sentenceCompletion === 1 ? 'border-indigo-500 bg-indigo-50/30 shadow-md' : 'border-slate-200 hover:border-indigo-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 2: Base Word in Parentheses</h5>
                          {paperStyles.sentenceCompletion === 1 && <div className="h-6 w-6 bg-indigo-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-3">
                          <div>1. The cat is _______________ (sleep) on the mat.</div>
                          <div>2. She _______________ (go) to the store yesterday.</div>
                        </div>
                      </div>

                      {/* Custom Sentence Completion Designs */}
                      {customDesigns.filter(d => d.type === 'sentence_completion' || d.type === 'sentenceCompletion').map(design => (
                        <div 
                          key={design.id}
                          onClick={() => {
                            setPaperStyles(prev => ({ ...prev, sentenceCompletion: design.id }));
                          }}
                          className={`p-6 rounded-2xl border-2 transition-all cursor-pointer group relative ${paperStyles.sentenceCompletion === design.id ? 'border-indigo-500 bg-indigo-50/30 shadow-md' : 'border-slate-200 hover:border-indigo-300 bg-white'}`}
                        >
                          <div className="flex justify-between items-center mb-4">
                            <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Custom: {design.name}</h5>
                            <div className="flex gap-1">
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const newName = prompt("Enter new name for this style:", design.name);
                                  if (newName) {
                                    setCustomDesigns(prev => prev.map(d => d.id === design.id ? { ...d, name: newName } : d));
                                  }
                                }}
                                className="h-6 w-6 text-slate-300 hover:text-blue-500 transition-colors"
                              >
                                <i className="fa-solid fa-pen text-[10px]"></i>
                              </button>
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteCustomDesign(design.id);
                                }}
                                className="h-6 w-6 text-slate-300 hover:text-rose-500 transition-colors"
                              >
                                <i className="fa-solid fa-trash text-[10px]"></i>
                              </button>
                            </div>
                          </div>
                          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-[10px] text-slate-400 italic">
                            Custom formatting and AI instructions applied.
                          </div>
                          {paperStyles.sentenceCompletion === design.id && <div className="absolute top-2 right-2 h-6 w-6 bg-indigo-500 text-white rounded-full flex items-center justify-center text-xs shadow-sm"><i className="fa-solid fa-check"></i></div>}
                        </div>
                      ))}

                      {/* Add New Sentence Completion Style Card */}
                      <div 
                        onClick={() => {
                          setDesignTargetTypeId('sentence_completion');
                          setSettingsTab('FORMAT_DESIGN');
                          setShowSettings(true);
                        }}
                        className="p-6 rounded-2xl border-2 border-dashed border-slate-200 hover:border-indigo-400 hover:bg-indigo-50/30 cursor-pointer transition-all flex flex-col items-center justify-center gap-3 group"
                      >
                        <div className="h-12 w-12 bg-slate-100 text-slate-400 rounded-full flex items-center justify-center group-hover:bg-indigo-100 group-hover:text-indigo-600 transition-all">
                          <i className="fa-solid fa-plus text-xl"></i>
                        </div>
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest group-hover:text-indigo-600">Add New Style</span>
                      </div>
                    </div>
                  </div>

                  {/* Word Box Design Samples */}
                  <div className="space-y-6 border-t border-slate-100 pt-8">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="h-10 w-10 bg-teal-100 text-teal-600 rounded-xl flex items-center justify-center">
                        <i className="fa-solid fa-box-open"></i>
                      </div>
                      <h4 className="text-lg font-bold text-slate-800 uppercase tracking-widest">Word Box Designs</h4>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Word Box Design 1 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, wordBox: 0 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.wordBox === 0 ? 'border-teal-500 bg-teal-50/30 shadow-md' : 'border-slate-200 hover:border-teal-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 1: Comma-separated</h5>
                          {paperStyles.wordBox === 0 && <div className="h-6 w-6 bg-teal-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-3">
                          <div className="border-2 border-slate-800 p-3 text-center font-bold">
                            apple, banana, cherry, date, elderberry
                          </div>
                          <div>1. My favorite fruit is the _______________.</div>
                        </div>
                      </div>

                      {/* Word Box Design 2 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, wordBox: 1 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.wordBox === 1 ? 'border-teal-500 bg-teal-50/30 shadow-md' : 'border-slate-200 hover:border-teal-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 2: Bulleted List</h5>
                          {paperStyles.wordBox === 1 && <div className="h-6 w-6 bg-teal-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-3">
                          <div className="border-2 border-slate-800 p-3 flex justify-center gap-6 font-bold">
                            <span>• apple</span>
                            <span>• banana</span>
                            <span>• cherry</span>
                          </div>
                          <div>1. My favorite fruit is the _______________.</div>
                        </div>
                      </div>

                      {/* Custom Word Box Designs */}
                      {customDesigns.filter(d => d.type === 'word_box' || d.type === 'wordBox').map(design => (
                        <div 
                          key={design.id}
                          onClick={() => {
                            setPaperStyles(prev => ({ ...prev, wordBox: design.id }));
                          }}
                          className={`p-6 rounded-2xl border-2 transition-all cursor-pointer group relative ${paperStyles.wordBox === design.id ? 'border-teal-500 bg-teal-50/30 shadow-md' : 'border-slate-200 hover:border-teal-300 bg-white'}`}
                        >
                          <div className="flex justify-between items-center mb-4">
                            <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Custom: {design.name}</h5>
                            <div className="flex gap-1">
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const newName = prompt("Enter new name for this style:", design.name);
                                  if (newName) {
                                    setCustomDesigns(prev => prev.map(d => d.id === design.id ? { ...d, name: newName } : d));
                                  }
                                }}
                                className="h-6 w-6 text-slate-300 hover:text-blue-500 transition-colors"
                              >
                                <i className="fa-solid fa-pen text-[10px]"></i>
                              </button>
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteCustomDesign(design.id);
                                }}
                                className="h-6 w-6 text-slate-300 hover:text-rose-500 transition-colors"
                              >
                                <i className="fa-solid fa-trash text-[10px]"></i>
                              </button>
                            </div>
                          </div>
                          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-[10px] text-slate-400 italic">
                            Custom formatting and AI instructions applied.
                          </div>
                          {paperStyles.wordBox === design.id && <div className="absolute top-2 right-2 h-6 w-6 bg-teal-500 text-white rounded-full flex items-center justify-center text-xs shadow-sm"><i className="fa-solid fa-check"></i></div>}
                        </div>
                      ))}

                      {/* Add New Word Box Style Card */}
                      <div 
                        onClick={() => {
                          setDesignTargetTypeId('word_box');
                          setSettingsTab('FORMAT_DESIGN');
                          setShowSettings(true);
                        }}
                        className="p-6 rounded-2xl border-2 border-dashed border-slate-200 hover:border-teal-400 hover:bg-teal-50/30 cursor-pointer transition-all flex flex-col items-center justify-center gap-3 group"
                      >
                        <div className="h-12 w-12 bg-slate-100 text-slate-400 rounded-full flex items-center justify-center group-hover:bg-teal-100 group-hover:text-teal-600 transition-all">
                          <i className="fa-solid fa-plus text-xl"></i>
                        </div>
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest group-hover:text-teal-600">Add New Style</span>
                      </div>
                    </div>
                  </div>

                  {/* Cloze Design Samples */}
                  <div className="space-y-6 border-t border-slate-100 pt-8">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="h-10 w-10 bg-cyan-100 text-cyan-600 rounded-xl flex items-center justify-center">
                        <i className="fa-solid fa-align-left"></i>
                      </div>
                      <h4 className="text-lg font-bold text-slate-800 uppercase tracking-widest">Cloze Designs</h4>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Cloze Design 1 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, cloze: 0 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.cloze === 0 ? 'border-cyan-500 bg-cyan-50/30 shadow-md' : 'border-slate-200 hover:border-cyan-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 1: Paragraph Style</h5>
                          {paperStyles.cloze === 0 && <div className="h-6 w-6 bg-cyan-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm leading-relaxed">
                          Yesterday, I (1) ____________ to the park. The weather (2) ____________ beautiful, and many children (3) ____________ playing games.
                        </div>
                      </div>

                      {/* Custom Cloze Designs */}
                      {customDesigns.filter(d => d.type === 'cloze' || d.type === 'cloze_paragraph').map(design => (
                        <div 
                          key={design.id}
                          onClick={() => {
                            setPaperStyles(prev => ({ ...prev, cloze: design.id }));
                          }}
                          className={`p-6 rounded-2xl border-2 transition-all cursor-pointer group relative ${paperStyles.cloze === design.id ? 'border-cyan-500 bg-cyan-50/30 shadow-md' : 'border-slate-200 hover:border-cyan-300 bg-white'}`}
                        >
                          <div className="flex justify-between items-center mb-4">
                            <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Custom: {design.name}</h5>
                            <div className="flex gap-1">
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const newName = prompt("Enter new name for this style:", design.name);
                                  if (newName) {
                                    setCustomDesigns(prev => prev.map(d => d.id === design.id ? { ...d, name: newName } : d));
                                  }
                                }}
                                className="h-6 w-6 text-slate-300 hover:text-blue-500 transition-colors"
                              >
                                <i className="fa-solid fa-pen text-[10px]"></i>
                              </button>
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteCustomDesign(design.id);
                                }}
                                className="h-6 w-6 text-slate-300 hover:text-rose-500 transition-colors"
                              >
                                <i className="fa-solid fa-trash text-[10px]"></i>
                              </button>
                            </div>
                          </div>
                          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-[10px] text-slate-400 italic">
                            Custom formatting and AI instructions applied.
                          </div>
                          {paperStyles.cloze === design.id && <div className="absolute top-2 right-2 h-6 w-6 bg-cyan-500 text-white rounded-full flex items-center justify-center text-xs shadow-sm"><i className="fa-solid fa-check"></i></div>}
                        </div>
                      ))}

                      {/* Add New Cloze Style Card */}
                      <div 
                        onClick={() => {
                          setDesignTargetTypeId('cloze');
                          setSettingsTab('FORMAT_DESIGN');
                          setShowSettings(true);
                        }}
                        className="p-6 rounded-2xl border-2 border-dashed border-slate-200 hover:border-cyan-400 hover:bg-cyan-50/30 cursor-pointer transition-all flex flex-col items-center justify-center gap-3 group"
                      >
                        <div className="h-12 w-12 bg-slate-100 text-slate-400 rounded-full flex items-center justify-center group-hover:bg-cyan-100 group-hover:text-cyan-600 transition-all">
                          <i className="fa-solid fa-plus text-xl"></i>
                        </div>
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest group-hover:text-cyan-600">Add New Style</span>
                      </div>
                    </div>
                  </div>

                  {/* Double MCQ Design Samples */}
                  <div className="space-y-6 border-t border-slate-100 pt-8">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="h-10 w-10 bg-rose-100 text-rose-600 rounded-xl flex items-center justify-center">
                        <i className="fa-solid fa-layer-group"></i>
                      </div>
                      <h4 className="text-lg font-bold text-slate-800 uppercase tracking-widest">Double MCQ Designs</h4>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Double MCQ Design 1 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, doubleMcq: 0 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.doubleMcq === 0 ? 'border-rose-500 bg-rose-50/30 shadow-md' : 'border-slate-200 hover:border-rose-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 1: Inline Pairs</h5>
                          {paperStyles.doubleMcq === 0 && <div className="h-6 w-6 bg-rose-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-4 border border-slate-200 rounded-xl font-serif text-sm space-y-3">
                          <div>1. If I (1) ________ more time, I (2) ________ to the party.</div>
                          <div className="grid grid-cols-2 gap-2 text-[10px] font-bold text-rose-600">
                            <span>A. have / will go</span>
                            <span>B. had / would go</span>
                            <span>C. have / would go</span>
                            <span>D. had / will go</span>
                          </div>
                        </div>
                      </div>

                      {/* Custom Double MCQ Designs */}
                      {customDesigns.filter(d => d.type === 'double_mcq' || d.id === 'g_pair').map(design => (
                        <div 
                          key={design.id}
                          onClick={() => {
                            setPaperStyles(prev => ({ ...prev, doubleMcq: design.id }));
                          }}
                          className={`p-6 rounded-2xl border-2 transition-all cursor-pointer group relative ${paperStyles.doubleMcq === design.id ? 'border-rose-500 bg-rose-50/30 shadow-md' : 'border-slate-200 hover:border-rose-300 bg-white'}`}
                        >
                          <div className="flex justify-between items-center mb-4">
                            <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Custom: {design.name}</h5>
                            <div className="flex gap-1">
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const newName = prompt("Enter new name for this style:", design.name);
                                  if (newName) {
                                    setCustomDesigns(prev => prev.map(d => d.id === design.id ? { ...d, name: newName } : d));
                                  }
                                }}
                                className="h-6 w-6 text-slate-300 hover:text-blue-500 transition-colors"
                              >
                                <i className="fa-solid fa-pen text-[10px]"></i>
                              </button>
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteCustomDesign(design.id);
                                }}
                                className="h-6 w-6 text-slate-300 hover:text-rose-500 transition-colors"
                              >
                                <i className="fa-solid fa-trash text-[10px]"></i>
                              </button>
                            </div>
                          </div>
                          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-[10px] text-slate-400 italic">
                            Custom formatting and AI instructions applied.
                          </div>
                          {paperStyles.doubleMcq === design.id && <div className="absolute top-2 right-2 h-6 w-6 bg-rose-500 text-white rounded-full flex items-center justify-center text-xs shadow-sm"><i className="fa-solid fa-check"></i></div>}
                        </div>
                      ))}

                      {/* Add New Double MCQ Style Card */}
                      <div 
                        onClick={() => {
                          setDesignTargetTypeId('double_mcq');
                          setSettingsTab('FORMAT_DESIGN');
                          setShowSettings(true);
                        }}
                        className="p-6 rounded-2xl border-2 border-dashed border-slate-200 hover:border-rose-400 hover:bg-rose-50/30 cursor-pointer transition-all flex flex-col items-center justify-center gap-3 group"
                      >
                        <div className="h-12 w-12 bg-slate-100 text-slate-400 rounded-full flex items-center justify-center group-hover:bg-rose-100 group-hover:text-rose-600 transition-all">
                          <i className="fa-solid fa-plus text-xl"></i>
                        </div>
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest group-hover:text-rose-600">Add New Style</span>
                      </div>
                    </div>
                  </div>

                  {/* Custom Grammar Designs */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
                    {customDesigns.filter(d => d.category === 'Grammar' && d.type !== 'correct_incorrect' && d.type !== 'true_false' && d.type !== 'tf').map(design => (
                      <div 
                        key={design.id}
                        onClick={() => {
                          setPaperStyles(prev => ({ ...prev, mcq: design.id }));
                          alert(`Applied custom grammar design: ${design.name}`);
                        }}
                        className="p-8 rounded-[40px] border-2 border-amber-100 bg-white hover:border-amber-300 cursor-pointer transition-all shadow-sm group"
                      >
                        <div className="flex justify-between items-start mb-6">
                          <div className="h-14 w-14 rounded-2xl bg-amber-100 text-amber-600 flex items-center justify-center text-2xl group-hover:bg-amber-600 group-hover:text-white transition-all">
                            <i className="fa-solid fa-wand-magic-sparkles"></i>
                          </div>
                          <span className="px-3 py-1 bg-slate-100 text-slate-500 rounded-full text-[8px] font-black uppercase tracking-widest">Grammar Custom</span>
                        </div>
                        <h4 className="text-lg font-black text-slate-900 uppercase tracking-tight mb-2">{design.name}</h4>
                        <p className="text-xs font-medium text-slate-500 leading-relaxed">Custom formatting and AI instructions applied.</p>
                      </div>
                    ))}
                  </div>
                    </>
                  )}

                  {architectTab === 'Mixed' && (
                    <div className="space-y-12">
                      <div className="bg-white p-8 rounded-[40px] border border-slate-100 shadow-sm">
                        <CollapsibleSection
                          title="Mixed Subject Test Configuration"
                          subtitle="Combine Grammar, Reading, and Vocabulary"
                          icon="fa-layer-group"
                          iconBg="bg-indigo-100"
                          iconColor="text-indigo-600"
                          isCollapsed={!!collapsedSections['mixed_config']}
                          onToggle={() => setCollapsedSections(prev => ({ ...prev, mixed_config: !prev.mixed_config }))}
                        >
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
                            <div className="p-6 bg-slate-50 rounded-3xl border border-slate-100">
                              <h5 className="text-[11px] font-black text-slate-900 uppercase mb-4 flex items-center gap-2">
                                <i className="fa-solid fa-book text-orange-500"></i> Grammar
                              </h5>
                              <ul className="space-y-2">
                                <li className="text-[10px] font-bold text-slate-600 uppercase flex items-center gap-2"><i className="fa-solid fa-check text-emerald-500"></i> Multiple Choice (MCQ)</li>
                                <li className="text-[10px] font-bold text-slate-600 uppercase flex items-center gap-2"><i className="fa-solid fa-check text-emerald-500"></i> Correct / Incorrect</li>
                                <li className="text-[10px] font-bold text-slate-600 uppercase flex items-center gap-2"><i className="fa-solid fa-check text-emerald-500"></i> Cloze Passage</li>
                                <li className="text-[10px] font-bold text-slate-600 uppercase flex items-center gap-2"><i className="fa-solid fa-check text-emerald-500"></i> Double MCQ</li>
                              </ul>
                            </div>
                            <div className="p-6 bg-slate-50 rounded-3xl border border-slate-100">
                              <h5 className="text-[11px] font-black text-slate-900 uppercase mb-4 flex items-center gap-2">
                                <i className="fa-solid fa-file-lines text-blue-500"></i> Reading
                              </h5>
                              <ul className="space-y-2">
                                <li className="text-[10px] font-bold text-slate-600 uppercase flex items-center gap-2"><i className="fa-solid fa-check text-emerald-500"></i> True / False</li>
                                <li className="text-[10px] font-bold text-slate-600 uppercase flex items-center gap-2"><i className="fa-solid fa-check text-emerald-500"></i> Multiple Choice (MCQ)</li>
                                <li className="text-[10px] font-bold text-slate-600 uppercase flex items-center gap-2"><i className="fa-solid fa-check text-emerald-500"></i> Inferential Reading</li>
                              </ul>
                            </div>
                            <div className="p-6 bg-slate-50 rounded-3xl border border-slate-100">
                              <h5 className="text-[11px] font-black text-slate-900 uppercase mb-4 flex items-center gap-2">
                                <i className="fa-solid fa-language text-purple-500"></i> Vocabulary
                              </h5>
                              <ul className="space-y-2">
                                <li className="text-[10px] font-bold text-slate-600 uppercase flex items-center gap-2"><i className="fa-solid fa-check text-emerald-500"></i> Matching Vocabulary</li>
                                <li className="text-[10px] font-bold text-slate-600 uppercase flex items-center gap-2"><i className="fa-solid fa-check text-emerald-500"></i> Circle</li>
                              </ul>
                            </div>
                          </div>
                        </CollapsibleSection>
                      </div>

                      {/* Custom Mixed Designs */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {customDesigns.filter(d => d.category === 'Mixed').map(design => (
                          <div 
                            key={design.id}
                            onClick={() => {
                              setPaperStyles(prev => ({ ...prev, mcq: design.id, tf: design.id, correctIncorrect: design.id, vocabulary: design.id }));
                              alert(`Applied custom mixed design: ${design.name}`);
                            }}
                            className="p-8 rounded-[40px] border-2 border-indigo-100 bg-white hover:border-indigo-300 cursor-pointer transition-all shadow-sm group"
                          >
                            <div className="flex justify-between items-start mb-6">
                              <div className="h-14 w-14 rounded-2xl bg-indigo-100 text-indigo-600 flex items-center justify-center text-2xl group-hover:bg-indigo-600 group-hover:text-white transition-all">
                                <i className="fa-solid fa-wand-magic-sparkles"></i>
                              </div>
                              <span className="px-3 py-1 bg-slate-100 text-slate-500 rounded-full text-[8px] font-black uppercase tracking-widest">Mixed Custom</span>
                            </div>
                            <h4 className="text-lg font-black text-slate-900 uppercase tracking-tight mb-2">{design.name}</h4>
                            <p className="text-xs font-medium text-slate-500 leading-relaxed">Custom formatting and AI instructions applied.</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {architectTab === 'Vocabulary' && (
                    <>
                      {/* Vocabulary Design Samples */}
                  <div className="space-y-6 md:col-span-2 mt-8 border-t border-slate-100 pt-8">
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 bg-rose-100 text-rose-600 rounded-xl flex items-center justify-center">
                          <i className="fa-solid fa-spell-check"></i>
                        </div>
                        <h4 className="text-lg font-bold text-slate-800 uppercase tracking-widest">Vocabulary Matching Designs</h4>
                      </div>
                      <button 
                        onClick={() => {
                          setDesignTargetTypeId('vocabulary');
                          setSettingsTab('FORMAT_DESIGN');
                          setShowSettings(true);
                        }}
                        className="px-4 py-2 bg-rose-600 text-white rounded-xl text-[10px] font-black uppercase tracking-widest hover:bg-rose-700 transition-all flex items-center gap-2"
                      >
                        <i className="fa-solid fa-plus"></i> Add NEW
                      </button>
                    </div>
                    
                    <div className="grid grid-cols-1 gap-8">
                      {/* Design 1 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, vocabulary: 0 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.vocabulary === 0 ? 'border-rose-500 bg-rose-50/30 shadow-md' : 'border-slate-200 hover:border-rose-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 1: Classic Fill-in-the-blank</h5>
                          {paperStyles.vocabulary === 0 && <div className="h-6 w-6 bg-rose-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-6 border border-slate-200 rounded-xl font-serif text-sm">
                          <div className="font-bold text-blue-800 mb-4 bg-blue-50 p-2">PART A: Study the following vocabulary words and their corresponding definitions.</div>
                          <div className="grid grid-cols-[150px_1fr] gap-2">
                            <div>1. Abundant</div><div><span className="text-blue-500">__</span>: Existing in large quantities.</div>
                            <div>2. Benevolent</div><div><span className="text-blue-500">__</span>: Well-meaning and kindly.</div>
                            <div>3. Candid</div><div><span className="text-blue-500">__</span>: Truthful and straightforward.</div>
                          </div>
                        </div>
                      </div>

                      {/* Design 2 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, vocabulary: 1 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.vocabulary === 1 ? 'border-rose-500 bg-rose-50/30 shadow-md' : 'border-slate-200 hover:border-rose-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 2: Alternating Rows (Italic)</h5>
                          {paperStyles.vocabulary === 1 && <div className="h-6 w-6 bg-rose-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-6 border border-slate-200 rounded-xl font-serif text-sm">
                          <div className="font-bold text-blue-800 mb-4 bg-blue-50 p-2">PART A: Study the following vocabulary words and their corresponding definitions.</div>
                          <div className="grid grid-cols-[150px_1fr]">
                            <div className="py-1 italic">1. Abundant</div><div className="py-1">Existing in large quantities.</div>
                            <div className="py-1 italic bg-gray-100">2. Benevolent</div><div className="py-1 bg-gray-100">Well-meaning and kindly.</div>
                            <div className="py-1 italic">3. Candid</div><div className="py-1">Truthful and straightforward.</div>
                          </div>
                        </div>
                      </div>

                      {/* Design 3 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, vocabulary: 2 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.vocabulary === 2 ? 'border-rose-500 bg-rose-50/30 shadow-md' : 'border-slate-200 hover:border-rose-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 3: Standard Alternating Rows</h5>
                          {paperStyles.vocabulary === 2 && <div className="h-6 w-6 bg-rose-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-6 border border-slate-200 rounded-xl font-serif text-sm">
                          <div className="font-bold text-blue-800 mb-4 bg-blue-50 p-2">PART A: Study the following vocabulary words and their corresponding definitions.</div>
                          <div className="grid grid-cols-[150px_1fr]">
                            <div className="py-1">1. Abundant</div><div className="py-1">Existing in large quantities.</div>
                            <div className="py-1 bg-gray-100">2. Benevolent</div><div className="py-1 bg-gray-100">Well-meaning and kindly.</div>
                            <div className="py-1">3. Candid</div><div className="py-1">Truthful and straightforward.</div>
                          </div>
                        </div>
                      </div>

                      {/* Design 4 */}
                      <div 
                        onClick={() => setPaperStyles(prev => ({ ...prev, vocabulary: 3 }))}
                        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.vocabulary === 3 ? 'border-rose-500 bg-rose-50/30 shadow-md' : 'border-slate-200 hover:border-rose-300 bg-white'}`}
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h5 className="font-bold text-slate-700">Design 4: Bordered Table (Reversed)</h5>
                          {paperStyles.vocabulary === 3 && <div className="h-6 w-6 bg-rose-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                        </div>
                        <div className="bg-white p-6 border border-slate-200 rounded-xl font-serif text-sm">
                          <div className="font-bold text-blue-800 mb-4 bg-blue-50 p-2">PART A: Study the following vocabulary words and their corresponding definitions.</div>
                          <table className="w-full border-collapse border border-slate-400">
                            <tbody>
                              <tr>
                                <td className="border border-slate-400 p-2">Existing in large quantities</td>
                                <td className="border border-slate-400 p-2">1. Abundant</td>
                              </tr>
                              <tr>
                                <td className="border border-slate-400 p-2">Well-meaning and kindly</td>
                                <td className="border border-slate-400 p-2">2. Benevolent</td>
                              </tr>
                              <tr>
                                <td className="border border-slate-400 p-2">Truthful and straightforward</td>
                                <td className="border border-slate-400 p-2">3. Candid</td>
                              </tr>
                            </tbody>
                          </table>
                        </div>
                      </div>

                      {/* Custom Vocabulary Designs */}
                      {customDesigns.filter(d => d.category === 'Vocabulary').map(design => (
                        <div 
                          key={design.id}
                          onClick={() => {
                            setPaperStyles(prev => ({ ...prev, vocabulary: design.id }));
                            alert(`Applied custom vocabulary design: ${design.name}`);
                          }}
                          className="p-6 rounded-2xl border-2 border-rose-100 bg-white hover:border-rose-300 cursor-pointer transition-all shadow-sm group"
                        >
                          <div className="flex justify-between items-center mb-4">
                            <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Custom: {design.name}</h5>
                            <div className="h-6 w-6 bg-rose-100 text-rose-600 rounded-full flex items-center justify-center text-[10px]"><i className="fa-solid fa-wand-magic-sparkles"></i></div>
                          </div>
                          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-[10px] text-slate-400 italic">
                            Custom formatting and AI instructions applied.
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                    </>
                  )}

                  {architectTab === 'Reading' && (
                    <div className="space-y-6">
                      <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center gap-3">
                          <div className="h-10 w-10 bg-emerald-100 text-emerald-600 rounded-xl flex items-center justify-center">
                            <i className="fa-solid fa-book-open"></i>
                          </div>
                          <h4 className="text-lg font-bold text-slate-800 uppercase tracking-widest">Reading Passage Designs</h4>
                        </div>
                        <button 
                          onClick={() => {
                            setDesignTargetTypeId('reading');
                            setSettingsTab('FORMAT_DESIGN');
                            setShowSettings(true);
                          }}
                          className="px-4 py-2 bg-emerald-600 text-white rounded-xl text-[10px] font-black uppercase tracking-widest hover:bg-emerald-700 transition-all flex items-center gap-2"
                        >
                          <i className="fa-solid fa-plus"></i> Add NEW
                        </button>
                      </div>
                      
                      <div className="grid grid-cols-1 gap-6">
                        {/* Reading Design 1 */}
                        <div 
                          onClick={() => setPaperStyles(prev => ({ ...prev, readingPassage: 0 }))}
                          className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.readingPassage === 0 ? 'border-emerald-500 bg-emerald-50/30 shadow-md' : 'border-slate-200 hover:border-emerald-300 bg-white'}`}
                        >
                          <div className="flex justify-between items-center mb-4">
                            <h5 className="font-bold text-slate-700">Design 1: Standard Single Column</h5>
                            {paperStyles.readingPassage === 0 && <div className="h-6 w-6 bg-emerald-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                          </div>
                          <div className="bg-white p-6 border border-slate-200 rounded-xl font-serif text-sm">
                            <h6 className="font-bold text-center mb-4">The Great Barrier Reef</h6>
                            <p className="text-justify leading-relaxed">The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometres over an area of approximately 344,400 square kilometres. The reef is located in the Coral Sea, off the coast of Queensland, Australia.</p>
                          </div>
                        </div>

                        {/* Reading Design 2 */}
                        <div 
                          onClick={() => setPaperStyles(prev => ({ ...prev, readingPassage: 1 }))}
                          className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.readingPassage === 1 ? 'border-emerald-500 bg-emerald-50/30 shadow-md' : 'border-slate-200 hover:border-emerald-300 bg-white'}`}
                        >
                          <div className="flex justify-between items-center mb-4">
                            <h5 className="font-bold text-slate-700">Design 2: Two-Column Layout</h5>
                            {paperStyles.readingPassage === 1 && <div className="h-6 w-6 bg-emerald-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                          </div>
                          <div className="bg-white p-6 border border-slate-200 rounded-xl font-serif text-sm">
                            <h6 className="font-bold text-center mb-4">The Great Barrier Reef</h6>
                            <div className="columns-2 gap-6 text-justify leading-relaxed">
                              <p>The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometres over an area of approximately 344,400 square kilometres.</p>
                              <p>The reef is located in the Coral Sea, off the coast of Queensland, Australia. It can be seen from outer space and is the world's biggest single structure made by living organisms.</p>
                            </div>
                          </div>
                        </div>

                        {/* Reading Design 3 */}
                        <div 
                          onClick={() => setPaperStyles(prev => ({ ...prev, readingPassage: 2 }))}
                          className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${paperStyles.readingPassage === 2 ? 'border-emerald-500 bg-emerald-50/30 shadow-md' : 'border-slate-200 hover:border-emerald-300 bg-white'}`}
                        >
                          <div className="flex justify-between items-center mb-4">
                            <h5 className="font-bold text-slate-700">Design 3: Bordered Box</h5>
                            {paperStyles.readingPassage === 2 && <div className="h-6 w-6 bg-emerald-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                          </div>
                          <div className="bg-white p-6 border-2 border-slate-800 rounded-xl font-serif text-sm">
                            <h6 className="font-bold text-center mb-4">The Great Barrier Reef</h6>
                            <p className="text-justify leading-relaxed">The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometres over an area of approximately 344,400 square kilometres. The reef is located in the Coral Sea, off the coast of Queensland, Australia.</p>
                          </div>
                        </div>

                        {/* Custom Reading Designs */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                          {customDesigns.filter(d => d.category === 'Reading' && d.type !== 'true_false' && d.type !== 'tf').map(design => (
                            <div 
                              key={design.id}
                              onClick={() => {
                                setPaperStyles(prev => ({ ...prev, readingPassage: design.id }));
                                alert(`Applied custom reading design: ${design.name}`);
                              }}
                              className="p-6 rounded-2xl border-2 border-emerald-100 bg-white hover:border-emerald-300 cursor-pointer transition-all shadow-sm group"
                            >
                              <div className="flex justify-between items-center mb-4">
                                <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Custom: {design.name}</h5>
                                <div className="h-6 w-6 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center text-[10px]"><i className="fa-solid fa-wand-magic-sparkles"></i></div>
                              </div>
                              <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-[10px] text-slate-400 italic">
                                Custom formatting and AI instructions applied.
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {architectTab === 'Generals' && (
                    <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
                      <div className="bg-indigo-50/50 p-10 rounded-[40px] border border-indigo-100 space-y-8">
                        <div className="flex items-center gap-4">
                          <div className="h-12 w-12 bg-indigo-600 text-white rounded-2xl flex items-center justify-center shadow-lg shadow-indigo-200">
                            <i className="fa-solid fa-layer-group"></i>
                          </div>
                          <div>
                            <h4 className="text-xl font-black text-slate-900 uppercase tracking-tight">Generals Domain</h4>
                            <p className="text-xs text-slate-500 font-medium">Mix Grammar, Reading, and Vocabulary into a unified assessment.</p>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                          <div className="bg-white p-8 rounded-3xl border border-slate-100 shadow-sm space-y-6">
                            <div className="flex items-center justify-between">
                              <h5 className="text-[11px] font-black text-slate-900 uppercase tracking-widest">True / False (Universal)</h5>
                              <div className="h-6 w-6 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center text-[10px]"><i className="fa-solid fa-check"></i></div>
                            </div>
                            <div className="space-y-3">
                              <div className="flex items-center gap-3">
                                <span className="text-[10px] font-black border-2 border-slate-900 px-2 py-0.5">______</span>
                                <div className="h-2 w-full bg-slate-100 rounded-full"></div>
                              </div>
                              <p className="text-[9px] text-slate-400 font-medium italic leading-relaxed">Neural Engine will generate T/F statements covering grammar rules, vocabulary definitions, and reading comprehension facts simultaneously.</p>
                            </div>
                          </div>

                          <div className="bg-white p-8 rounded-3xl border border-slate-100 shadow-sm space-y-6">
                            <div className="flex items-center justify-between">
                              <h5 className="text-[11px] font-black text-slate-900 uppercase tracking-widest">Correct / Incorrect (Universal)</h5>
                              <div className="h-6 w-6 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center text-[10px]"><i className="fa-solid fa-check"></i></div>
                            </div>
                            <div className="space-y-3">
                              <div className="flex items-center gap-3">
                                <span className="text-[10px] font-black text-master-green uppercase">C / I</span>
                                <div className="h-2 w-full bg-slate-100 rounded-full"></div>
                              </div>
                              <p className="text-[9px] text-slate-400 font-medium italic leading-relaxed">Neural Engine will test situational accuracy across all domains. E.g., "I must to go" (Grammar) or "He is a bustling person" (Vocab) or "The text says he died" (Reading).</p>
                            </div>
                          </div>
                        </div>

                        <div className="p-6 bg-white/60 rounded-3xl border border-white flex items-center gap-4">
                          <i className="fa-solid fa-circle-info text-indigo-500"></i>
                          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wide">Generals mode is active when you select "FULL GENERAL TEST" in the Command tab.</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {architectTab === 'Custom' && (
                    <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
                      <div className="bg-white p-10 rounded-[40px] border border-slate-100 shadow-sm space-y-10">
                        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
                          <div className="flex items-center gap-4">
                            <div className="h-12 w-12 bg-orange-600 text-white rounded-2xl flex items-center justify-center shadow-lg shadow-orange-200">
                              <i className="fa-solid fa-puzzle-piece"></i>
                            </div>
                            <div>
                              <h4 className="text-xl font-black text-slate-900 uppercase tracking-tight">Custom Subject Architect</h4>
                              <p className="text-xs text-slate-500 font-medium">Filter and manage your specialized test formats.</p>
                            </div>
                          </div>
                          
                          <div className="flex bg-slate-100 p-1 rounded-2xl border border-slate-200 overflow-x-auto no-scrollbar max-w-full">
                            {['All', 'MCQ', 'Matching', 'T/F', 'C/I', 'Box Style', 'Other'].map(sub => (
                              <button
                                key={sub}
                                onClick={() => setCustomArchitectSubTab(sub)}
                                className={`px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all whitespace-nowrap ${customArchitectSubTab === sub ? 'bg-white text-orange-600 shadow-sm' : 'text-slate-400 hover:text-slate-600'}`}
                              >
                                {sub}
                              </button>
                            ))}
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                          {/* Filtered Custom Designs */}
                          {customDesigns
                            .filter(d => customArchitectSubTab === 'All' || d.type === customArchitectSubTab || (customArchitectSubTab === 'Other' && !['MCQ', 'Matching', 'T/F', 'C/I', 'Box Style'].includes(d.type)))
                            .map(design => (
                              <div 
                                key={design.id}
                                onClick={() => {
                                  const designType = design.type || 'mcq';
                                  setPaperStyles(prev => ({ ...prev, [designType]: design.id }));
                                  if (design.style.mcqStyle !== undefined) setMcqStyle(design.style.mcqStyle);
                                  alert(`Applied custom design: ${design.name}`);
                                }}
                                className="p-8 rounded-[40px] border-2 border-orange-100 bg-white hover:border-orange-300 cursor-pointer transition-all shadow-sm hover:shadow-md group"
                              >
                                <div className="flex justify-between items-start mb-6">
                                  <div className="h-14 w-14 rounded-2xl bg-orange-100 text-orange-600 flex items-center justify-center text-2xl group-hover:bg-orange-600 group-hover:text-white transition-all">
                                    <i className="fa-solid fa-wand-magic-sparkles"></i>
                                  </div>
                                  <span className="px-3 py-1 bg-slate-100 text-slate-500 rounded-full text-[8px] font-black uppercase tracking-widest">{design.type || 'Custom'}</span>
                                </div>
                                <h4 className="text-lg font-black text-slate-900 uppercase tracking-tight mb-2">{design.name}</h4>
                                <p className="text-xs font-medium text-slate-500 leading-relaxed">Custom formatting and AI instructions applied.</p>
                              </div>
                            ))}
                          
                          {/* Empty State for Filter */}
                          {customDesigns.filter(d => customArchitectSubTab === 'All' || d.type === customArchitectSubTab).length === 0 && (
                            <div className="md:col-span-2 py-20 text-center border-2 border-dashed border-slate-100 rounded-[40px]">
                              <div className="h-20 w-20 bg-slate-50 text-slate-300 rounded-full flex items-center justify-center mx-auto mb-6 text-3xl">
                                <i className="fa-solid fa-magnifying-glass"></i>
                              </div>
                              <h5 className="text-slate-400 font-black uppercase tracking-widest text-sm">No designs found for "{customArchitectSubTab}"</h5>
                              <p className="text-slate-300 text-xs mt-2">Create a new format and save it to this category.</p>
                            </div>
                          )}
                        </div>

                        {/* Big Custom Section */}
                        <div className="pt-10 border-t border-slate-100">
                          <div className="flex items-center gap-4 mb-8">
                            <div className="h-10 w-10 bg-purple-600 text-white rounded-xl flex items-center justify-center shadow-lg shadow-purple-200">
                              <i className="fa-solid fa-star"></i>
                            </div>
                            <div>
                              <h4 className="text-lg font-black text-slate-900 uppercase tracking-tight">Big Custom Templates</h4>
                              <p className="text-xs text-slate-500 font-medium">Advanced multi-part templates for complex assessments.</p>
                            </div>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            {[
                              { id: 'bc1', name: 'Words in a Box Style', type: 'Box Style', icon: 'fa-box-archive' },
                              { id: 'bc2', name: 'Scientific Method MCQ', type: 'MCQ', icon: 'fa-flask' },
                              { id: 'bc3', name: 'Historical Timeline Matching', type: 'Matching', icon: 'fa-timeline' }
                            ].map(temp => (
                              <div key={temp.id} className="p-6 bg-slate-50 rounded-3xl border border-slate-100 hover:border-purple-300 cursor-pointer transition-all group">
                                <div className="h-12 w-12 bg-white rounded-2xl flex items-center justify-center text-purple-600 mb-4 shadow-sm group-hover:bg-purple-600 group-hover:text-white transition-all">
                                  <i className={`fa-solid ${temp.icon}`}></i>
                                </div>
                                <h5 className="text-[11px] font-black text-slate-900 uppercase mb-1">{temp.name}</h5>
                                <span className="text-[8px] font-bold text-slate-400 uppercase tracking-widest">{temp.type}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="bg-orange-50 border border-orange-100 rounded-3xl p-8 flex items-start gap-6">
                <div className="h-12 w-12 bg-white rounded-2xl flex items-center justify-center text-orange-600 shadow-sm flex-shrink-0">
                  <i className="fa-solid fa-lightbulb text-xl"></i>
                </div>
                <div>
                  <h4 className="text-sm font-bold text-orange-900 uppercase tracking-widest mb-2">Pro Tip</h4>
                  <p className="text-xs text-orange-700 leading-relaxed">
                    Setting a style as default will ensure the AI Neural Engine always prioritizes that specific formatting when generating new test items. You can change these defaults at any time to experiment with different textbook-quality layouts.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {viewMode === 'header_footer_design' && (
        <section 
          style={{ 
            marginLeft: isSidebarOpen && sidebarSide === 'left' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px',
            marginRight: isSidebarOpen && sidebarSide === 'right' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px'
          }}
          className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-500 bg-slate-50 transition-all duration-300"
        >
          <div className="p-4 lg:p-6 bg-white border-b border-slate-200 flex flex-wrap gap-4 justify-between items-center z-10 no-print shadow-sm">
            <button onClick={() => setViewMode('generator')} className="border border-slate-200 text-slate-600 px-6 lg:px-8 py-3 rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-slate-50 flex items-center gap-4 group transition-all">
              <i className="fa-solid fa-arrow-left group-hover:-translate-x-1 transition-transform"></i> WORKSPACE
            </button>
            <div className="flex-1 text-center">
              <h2 className="text-slate-800 font-bold uppercase tracking-widest text-[12px]">Header & Footer Styles Workspace</h2>
            </div>
            <div className="flex gap-2">
              <button 
                onClick={() => {
                  setBrandSettings(prev => ({ ...prev, headerStyle: paperDesign }));
                  setViewMode('generator');
                }}
                className="px-6 py-3 bg-blue-600 text-white rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-blue-700 shadow-sm flex items-center gap-2 transition-all"
              >
                <i className="fa-solid fa-check"></i> Save & Set Default
              </button>
            </div>
          </div>
          <div className="flex-1 bg-slate-50 overflow-y-auto p-8 no-scrollbar">
            <div className="max-w-4xl mx-auto space-y-10">
              <div className="bg-white rounded-[32px] p-10 border border-slate-100 shadow-sm">
                <h3 className="text-xl font-black text-slate-900 mb-2 uppercase tracking-tight">Header & Footer Architect</h3>
                <p className="text-sm text-slate-500 mb-8">Select from professional header and footer designs for your paper test.</p>
                
                <div className="bg-slate-50 p-8 rounded-3xl border border-slate-100 mb-10 space-y-8">
                  <h4 className="text-[11px] font-black text-slate-900 uppercase tracking-widest flex items-center gap-2">
                    <i className="fa-solid fa-sliders text-orange-500"></i> Advanced Header Customization
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div className="space-y-4">
                      <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Custom Header Text (Overrides Default)</label>
                      <input 
                        value={brandSettings.customHeaderText || ''} 
                        onChange={e => setBrandSettings({ ...brandSettings, customHeaderText: e.target.value })} 
                        className="w-full bg-white border border-slate-200 rounded-2xl px-6 py-4 outline-none focus:border-orange-500 font-bold text-slate-700 shadow-sm" 
                        placeholder="e.g. FINAL TERM EXAMINATION - SEMESTER 1" 
                      />
                    </div>
                    <div className="space-y-4">
                      <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Header Ruler Style</label>
                      <div className="flex bg-white p-1.5 rounded-2xl gap-1 overflow-x-auto no-scrollbar border border-slate-200 shadow-sm">
                        {[0, 1, 2, 3, 4, 5, 6].map(style => (
                          <button 
                            key={style} 
                            onClick={() => setBrandSettings({ ...brandSettings, headerRulerStyle: style })} 
                            className={`px-4 py-2 rounded-xl text-[10px] font-black transition-all shrink-0 ${brandSettings.headerRulerStyle === style ? 'bg-orange-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-600'}`}
                          >
                            {style === 0 ? 'None' : `Ruler ${style}`}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  {/* Header Design 1 */}
                  <div 
                    onClick={() => setPaperDesign(0)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 0 ? 'border-blue-500 bg-blue-50/30 shadow-md' : 'border-slate-200 hover:border-blue-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 1: Classic Professional</h5>
                      {paperDesign === 0 && <div className="h-6 w-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="border-b-2 border-black pb-4 mb-6">
                        <div className="flex justify-between font-bold text-[10px] mb-2">
                          <span>{brandSettings.studentLabel}: _________________</span>
                          <span>{brandSettings.dateLabel}: ____/____/____</span>
                        </div>
                        <div className="flex justify-between font-bold text-[10px]">
                          <span>{brandSettings.classLabel}: _________________</span>
                          <span>{brandSettings.teacherLabel}: _________________</span>
                        </div>
                        <h1 className="text-center mt-6 text-xl font-black uppercase tracking-tighter">{brandSettings.schoolName}</h1>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 2 */}
                  <div 
                    onClick={() => setPaperDesign(1)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 1 ? 'border-blue-500 bg-blue-50/30 shadow-md' : 'border-slate-200 hover:border-blue-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 2: Boxed Header</h5>
                      {paperDesign === 1 && <div className="h-6 w-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="border-2 border-black p-6 text-center">
                        <h1 className="text-lg font-black mb-4 uppercase tracking-widest">{brandSettings.schoolName}</h1>
                        <div className="border-t border-black pt-4 grid grid-cols-2 gap-4 text-left text-[9px] font-bold">
                          <div>{brandSettings.studentLabel}: ____________</div>
                          <div>{brandSettings.dateLabel}: ____________</div>
                          <div>{brandSettings.classLabel}: ____________</div>
                          <div>{brandSettings.scoreLabel}: ____ / ____</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 3 */}
                  <div 
                    onClick={() => setPaperDesign(2)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 2 ? 'border-blue-500 bg-blue-50/30 shadow-md' : 'border-slate-200 hover:border-blue-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 3: Modern Minimal</h5>
                      {paperDesign === 2 && <div className="h-6 w-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="text-[8px] border-bottom border-slate-300 pb-2 mb-4 flex justify-between uppercase font-bold text-slate-400 tracking-widest">
                        <span>{brandSettings.schoolName}</span>
                        <span>Academic Year: 2025-2026</span>
                      </div>
                      <h1 className="text-2xl font-black tracking-tighter mb-4 uppercase">WORKSHEET TITLE</h1>
                      <div className="bg-slate-100 p-4 rounded-lg flex gap-6 text-[9px] font-bold text-slate-600">
                        <span>{brandSettings.studentLabel}: _________</span>
                        <span>{brandSettings.idLabel}: _____</span>
                        <span>{brandSettings.dateLabel}: _____</span>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 4 */}
                  <div 
                    onClick={() => setPaperDesign(3)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 3 ? 'border-blue-500 bg-blue-50/30 shadow-md' : 'border-slate-200 hover:border-blue-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 4: Accent Sidebar</h5>
                      {paperDesign === 3 && <div className="h-6 w-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="border-l-[6px] border-blue-600 pl-6">
                        <h1 className="text-xl font-black text-blue-900 uppercase tracking-tight mb-1">{brandSettings.schoolName}</h1>
                        <div className="text-[10px] text-slate-400 font-bold mb-4 uppercase tracking-widest">Topic: General Assessment</div>
                        <div className="flex gap-6 text-[9px] font-bold text-slate-500 border-t border-dashed border-slate-200 pt-4">
                          <span>{brandSettings.studentLabel}: _________</span>
                          <span>{brandSettings.classLabel}: _____</span>
                          <span>{brandSettings.dateLabel}: _____</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 5 */}
                  <div 
                    onClick={() => setPaperDesign(4)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 4 ? 'border-blue-500 bg-blue-50/30 shadow-md' : 'border-slate-200 hover:border-blue-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 5: Dark Header</h5>
                      {paperDesign === 4 && <div className="h-6 w-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="bg-slate-900 text-white p-8 rounded-xl relative overflow-hidden">
                        <div className="absolute right-[-20px] top-[-20px] w-24 h-24 bg-white/5 rounded-full"></div>
                        <h1 className="text-xl font-black uppercase tracking-widest mb-4 relative z-10">{brandSettings.schoolName}</h1>
                        <div className="grid grid-cols-3 gap-4 text-[9px] font-bold opacity-80 relative z-10">
                          <div className="border-b border-white/30 pb-1">{brandSettings.studentLabel}: _______</div>
                          <div className="border-b border-white/30 pb-1">{brandSettings.idLabel}: _______</div>
                          <div className="border-b border-white/30 pb-1">{brandSettings.scoreLabel}: ____</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 6 - Green Nature */}
                  <div 
                    onClick={() => setPaperDesign(5)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 5 ? 'border-emerald-500 bg-emerald-50/30 shadow-md' : 'border-slate-200 hover:border-emerald-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 6: Green Nature</h5>
                      {paperDesign === 5 && <div className="h-6 w-6 bg-emerald-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="border: 4pt solid #16a34a; padding: 10pt; border-radius: 8pt; background: #f0fdf4;">
                        <div className="flex justify-between items-center border-bottom: 2pt solid #16a34a; padding-bottom: 5pt; margin-bottom: 5pt;">
                          <h1 className="text-lg font-black text-emerald-800 uppercase">{brandSettings.schoolName}</h1>
                          <div className="text-[8px] text-emerald-600 font-bold">
                            <div>{brandSettings.dateLabel}: ____/____/____</div>
                            <div>{brandSettings.classLabel}: ___________</div>
                          </div>
                        </div>
                        <div className="text-[9px] text-emerald-900 font-bold">{brandSettings.studentLabel}: ________________________________</div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 7 - Emerald Professional */}
                  <div 
                    onClick={() => setPaperDesign(6)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 6 ? 'border-emerald-500 bg-emerald-50/30 shadow-md' : 'border-slate-200 hover:border-emerald-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 7: Emerald Professional</h5>
                      {paperDesign === 6 && <div className="h-6 w-6 bg-emerald-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="bg-emerald-900 text-white p-6 rounded-lg text-center">
                        <h1 className="text-xl font-light tracking-[4px] uppercase mb-4">{brandSettings.schoolName}</h1>
                        <div className="flex justify-around text-[8px] font-mono opacity-80">
                          <span>[ {brandSettings.studentLabel}: ________ ]</span>
                          <span>[ {brandSettings.dateLabel}: __/__/__ ]</span>
                          <span>[ {brandSettings.scoreLabel}: ____ ]</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 8 - Yellow/Gold */}
                  <div 
                    onClick={() => setPaperDesign(7)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 7 ? 'border-yellow-500 bg-yellow-50/30 shadow-md' : 'border-slate-200 hover:border-yellow-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 8: Boxed Gold</h5>
                      {paperDesign === 7 && <div className="h-6 w-6 bg-yellow-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="border border-slate-200 rounded-lg overflow-hidden">
                        <div className="bg-yellow-400 p-3 flex justify-between items-center">
                          <h1 className="text-sm font-black text-yellow-900 uppercase">{brandSettings.schoolName}</h1>
                          <div className="bg-white px-3 py-1 rounded-full text-[8px] font-bold text-yellow-800">{brandSettings.scoreLabel}: ____</div>
                        </div>
                        <div className="p-3 grid grid-cols-2 gap-4 text-[9px] font-bold">
                          <div className="border-b border-slate-100">{brandSettings.studentLabel}: ________</div>
                          <div className="border-b border-slate-100">{brandSettings.dateLabel}: ________</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 9 - Modern Red */}
                  <div 
                    onClick={() => setPaperDesign(8)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 8 ? 'border-rose-500 bg-rose-50/30 shadow-md' : 'border-slate-200 hover:border-rose-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 9: Modern Red</h5>
                      {paperDesign === 8 && <div className="h-6 w-6 bg-rose-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="border-t-4 border-rose-600 pt-4 flex justify-between items-start">
                        <div>
                          <h1 className="text-2xl font-black text-rose-900 leading-none uppercase">{brandSettings.schoolName}</h1>
                          <div className="text-[9px] text-rose-500 font-bold mt-2 uppercase tracking-widest">Academic Evaluation</div>
                        </div>
                        <div className="text-right text-[9px] font-serif italic text-slate-500">
                          <div>{brandSettings.studentLabel}: _________</div>
                          <div>{brandSettings.classLabel}: _________</div>
                          <div>{brandSettings.dateLabel}: _________</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 9b - Modern Green (Default) */}
                  <div 
                    onClick={() => setPaperDesign(18)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 18 ? 'border-emerald-500 bg-emerald-50/30 shadow-md' : 'border-slate-200 hover:border-emerald-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 9b: Modern Green</h5>
                      {paperDesign === 18 && <div className="h-6 w-6 bg-emerald-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="border-t-4 border-emerald-600 pt-4 flex justify-between items-start">
                        <div>
                          <h1 className="text-2xl font-black text-emerald-900 leading-none uppercase">{brandSettings.schoolName}</h1>
                          <div className="text-[9px] text-emerald-500 font-bold mt-2 uppercase tracking-widest">Academic Evaluation</div>
                        </div>
                        <div className="text-right text-[9px] font-serif italic text-slate-500">
                          <div>{brandSettings.studentLabel}: _________</div>
                          <div>{brandSettings.classLabel}: _________</div>
                          <div>{brandSettings.dateLabel}: _________</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 9c - Modern Blue */}
                  <div 
                    onClick={() => setPaperDesign(19)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 19 ? 'border-blue-500 bg-blue-50/30 shadow-md' : 'border-slate-200 hover:border-blue-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 9c: Modern Blue</h5>
                      {paperDesign === 19 && <div className="h-6 w-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="border-t-4 border-blue-600 pt-4 flex justify-between items-start">
                        <div>
                          <h1 className="text-2xl font-black text-blue-900 leading-none uppercase">{brandSettings.schoolName}</h1>
                          <div className="text-[9px] text-blue-500 font-bold mt-2 uppercase tracking-widest">Academic Evaluation</div>
                        </div>
                        <div className="text-right text-[9px] font-serif italic text-slate-500">
                          <div>{brandSettings.studentLabel}: _________</div>
                          <div>{brandSettings.classLabel}: _________</div>
                          <div>{brandSettings.dateLabel}: _________</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 9d - Modern Purple */}
                  <div 
                    onClick={() => setPaperDesign(20)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 20 ? 'border-purple-500 bg-purple-50/30 shadow-md' : 'border-slate-200 hover:border-purple-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 9d: Modern Purple</h5>
                      {paperDesign === 20 && <div className="h-6 w-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="border-t-4 border-purple-600 pt-4 flex justify-between items-start">
                        <div>
                          <h1 className="text-2xl font-black text-purple-900 leading-none uppercase">{brandSettings.schoolName}</h1>
                          <div className="text-[9px] text-purple-500 font-bold mt-2 uppercase tracking-widest">Academic Evaluation</div>
                        </div>
                        <div className="text-right text-[9px] font-serif italic text-slate-500">
                          <div>{brandSettings.studentLabel}: _________</div>
                          <div>{brandSettings.classLabel}: _________</div>
                          <div>{brandSettings.dateLabel}: _________</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 9e - Modern Orange */}
                  <div 
                    onClick={() => setPaperDesign(21)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 21 ? 'border-orange-500 bg-orange-50/30 shadow-md' : 'border-slate-200 hover:border-orange-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 9e: Modern Orange</h5>
                      {paperDesign === 21 && <div className="h-6 w-6 bg-orange-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="border-t-4 border-orange-600 pt-4 flex justify-between items-start">
                        <div>
                          <h1 className="text-2xl font-black text-orange-900 leading-none uppercase">{brandSettings.schoolName}</h1>
                          <div className="text-[9px] text-orange-500 font-bold mt-2 uppercase tracking-widest">Academic Evaluation</div>
                        </div>
                        <div className="text-right text-[9px] font-serif italic text-slate-500">
                          <div>{brandSettings.studentLabel}: _________</div>
                          <div>{brandSettings.classLabel}: _________</div>
                          <div>{brandSettings.dateLabel}: _________</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 10 - Clean Minimal */}
                  <div 
                    onClick={() => setPaperDesign(9)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 9 ? 'border-slate-800 bg-slate-50 shadow-md' : 'border-slate-200 hover:border-slate-400 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 10: Clean Minimal</h5>
                      {paperDesign === 9 && <div className="h-6 w-6 bg-slate-800 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="flex flex-col gap-3">
                        <h1 className="text-sm font-medium text-slate-400 border-l-2 border-slate-200 pl-4 uppercase">{brandSettings.schoolName}</h1>
                        <h1 className="text-2xl font-black text-slate-900 leading-none uppercase tracking-tighter">WORKSHEET TITLE</h1>
                        <div className="h-px bg-slate-100 w-full"></div>
                        <div className="flex gap-10 text-[9px] font-bold text-slate-400">
                          <span>{brandSettings.studentLabel}: _________</span>
                          <span>{brandSettings.dateLabel}: _________</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 11 - School Logo Header */}
                  <div 
                    onClick={() => setPaperDesign(10)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 10 ? 'border-orange-500 bg-orange-50/30 shadow-md' : 'border-slate-200 hover:border-orange-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 11: School Logo Header</h5>
                      {paperDesign === 10 && <div className="h-6 w-6 bg-orange-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="flex items-center gap-4 border-b-2 border-black pb-4">
                        {brandSettings.logoData ? (
                          <img src={brandSettings.logoData} className="h-12 w-12 object-contain" />
                        ) : (
                          <div className="h-12 w-12 bg-slate-100 rounded-lg flex items-center justify-center text-[8px] text-slate-400 font-bold text-center uppercase">Logo</div>
                        )}
                        <div className="flex-1">
                          <h1 className="text-sm font-black uppercase tracking-tight">{brandSettings.schoolName}</h1>
                          <div className="text-[8px] text-slate-400 font-bold uppercase tracking-widest">{brandSettings.schoolAddress}</div>
                        </div>
                        <div className="flex flex-col gap-1">
                          <input 
                            type="file" 
                            id="logo-upload-h11" 
                            className="hidden" 
                            accept="image/*"
                            onChange={(e) => {
                              const file = e.target.files?.[0];
                              if (file) {
                                const reader = new FileReader();
                                reader.onload = (ev) => {
                                  setBrandSettings(prev => ({ ...prev, logoData: ev.target?.result as string }));
                                };
                                reader.readAsDataURL(file);
                              }
                            }}
                          />
                          <button 
                            onClick={(e) => { e.stopPropagation(); document.getElementById('logo-upload-h11')?.click(); }}
                            className="h-8 w-8 bg-slate-50 text-slate-400 rounded-lg flex items-center justify-center hover:bg-orange-100 hover:text-orange-600 transition-all border border-slate-100"
                            title="Upload Logo"
                          >
                            <i className="fa-solid fa-upload text-[10px]"></i>
                          </button>
                          {brandSettings.logoData && (
                            <button 
                              onClick={(e) => { e.stopPropagation(); setBrandSettings(prev => ({ ...prev, logoData: undefined })); }}
                              className="h-8 w-8 bg-slate-50 text-rose-400 rounded-lg flex items-center justify-center hover:bg-rose-100 hover:text-rose-600 transition-all border border-slate-100"
                              title="Remove Logo"
                            >
                              <i className="fa-solid fa-trash text-[10px]"></i>
                            </button>
                          )}
                        </div>
                      </div>
                      <div className="mt-4 flex gap-4 text-[8px] font-bold text-slate-400">
                        <span>{brandSettings.studentLabel}: _________</span>
                        <span>{brandSettings.dateLabel}: _________</span>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 12 - Royal Gold */}
                  <div 
                    onClick={() => setPaperDesign(11)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 11 ? 'border-yellow-600 bg-yellow-50/30 shadow-md' : 'border-slate-200 hover:border-yellow-400 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 12: Royal Gold</h5>
                      {paperDesign === 11 && <div className="h-6 w-6 bg-yellow-600 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="border-4 border-yellow-600 p-4 bg-yellow-50">
                        <h1 className="text-center text-lg font-black text-yellow-900 uppercase">{brandSettings.schoolName}</h1>
                        <div className="mt-4 flex justify-between text-[8px] font-bold text-yellow-700">
                          <span>{brandSettings.studentLabel}: _________</span>
                          <span>{brandSettings.scoreLabel}: ____</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 13 - Deep Ocean */}
                  <div 
                    onClick={() => setPaperDesign(12)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 12 ? 'border-blue-800 bg-blue-50/30 shadow-md' : 'border-slate-200 hover:border-blue-400 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 13: Deep Ocean</h5>
                      {paperDesign === 12 && <div className="h-6 w-6 bg-blue-800 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="bg-blue-900 text-white p-6 rounded-none">
                        <h1 className="text-xl font-black uppercase tracking-widest mb-2">{brandSettings.schoolName}</h1>
                        <div className="h-1 bg-blue-400 w-1/4 mb-4"></div>
                        <div className="flex gap-4 text-[8px] font-bold opacity-70">
                          <span>{brandSettings.studentLabel}: _________</span>
                          <span>{brandSettings.dateLabel}: _________</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 14 - Sunset Vibrant */}
                  <div 
                    onClick={() => setPaperDesign(13)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 13 ? 'border-rose-600 bg-rose-50/30 shadow-md' : 'border-slate-200 hover:border-rose-400 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 14: Sunset Vibrant</h5>
                      {paperDesign === 13 && <div className="h-6 w-6 bg-rose-600 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="bg-gradient-to-r from-rose-600 to-orange-500 text-white p-6 rounded-2xl">
                        <h1 className="text-lg font-black uppercase">{brandSettings.schoolName}</h1>
                        <div className="mt-4 grid grid-cols-2 gap-2 text-[8px] font-bold">
                          <div className="bg-white/20 p-1 rounded">{brandSettings.studentLabel}: _______</div>
                          <div className="bg-white/20 p-1 rounded">{brandSettings.classLabel}: _______</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 15 - Cyberpunk */}
                  <div 
                    onClick={() => setPaperDesign(14)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 14 ? 'border-black bg-slate-100 shadow-md' : 'border-slate-200 hover:border-slate-400 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 15: Cyberpunk</h5>
                      {paperDesign === 14 && <div className="h-6 w-6 bg-black text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="border-2 border-black p-4 relative">
                        <div className="absolute -top-3 -left-3 bg-black text-white px-2 py-1 text-[8px] font-black uppercase">TOP SECRET</div>
                        <h1 className="text-xl font-black uppercase tracking-tighter mt-2">{brandSettings.schoolName}</h1>
                        <div className="mt-4 flex gap-4 text-[8px] font-mono">
                          <span>&gt; {brandSettings.studentLabel}: _________</span>
                          <span>&gt; {brandSettings.dateLabel}: _________</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 16 - Academic Heavy */}
                  <div 
                    onClick={() => setPaperDesign(15)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 15 ? 'border-slate-900 bg-slate-50 shadow-md' : 'border-slate-200 hover:border-slate-400 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 16: Academic Heavy</h5>
                      {paperDesign === 15 && <div className="h-6 w-6 bg-slate-900 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="border-b-4 border-double border-black pb-4 text-center">
                        <h1 className="text-2xl font-serif font-black uppercase">{brandSettings.schoolName}</h1>
                        <p className="text-[8px] font-serif italic mt-1">Established 1995 • Academic Excellence</p>
                        <div className="mt-4 flex justify-center gap-8 text-[9px] font-bold">
                          <span>{brandSettings.studentLabel}: _________________</span>
                          <span>{brandSettings.dateLabel}: ____/____/____</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 17 - Art Deco */}
                  <div 
                    onClick={() => setPaperDesign(16)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 16 ? 'border-indigo-600 bg-indigo-50/30 shadow-md' : 'border-slate-200 hover:border-indigo-400 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 17: Art Deco</h5>
                      {paperDesign === 16 && <div className="h-6 w-6 bg-indigo-600 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="border-8 border-double border-indigo-600 p-4 text-center">
                        <h1 className="text-lg font-black text-indigo-900 uppercase tracking-[0.2em]">{brandSettings.schoolName}</h1>
                        <div className="h-px bg-indigo-200 w-full my-4"></div>
                        <div className="flex justify-between text-[8px] font-bold text-indigo-600 uppercase">
                          <span>{brandSettings.studentLabel}: _______</span>
                          <span>{brandSettings.scoreLabel}: ____</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 18 - Futuristic */}
                  <div 
                    onClick={() => setPaperDesign(17)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 17 ? 'border-sky-500 bg-sky-50/30 shadow-md' : 'border-slate-200 hover:border-sky-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 18: Futuristic</h5>
                      {paperDesign === 17 && <div className="h-6 w-6 bg-sky-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="bg-sky-50 border border-sky-200 p-6 rounded-none relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-16 h-16 bg-sky-500/10 rotate-45 translate-x-8 -translate-y-8"></div>
                        <h1 className="text-xl font-black text-sky-900 uppercase tracking-tighter mb-4">{brandSettings.schoolName}</h1>
                        <div className="flex gap-4 text-[8px] font-bold text-sky-600">
                          <span className="bg-white px-2 py-1 border border-sky-100">{brandSettings.studentLabel}: _________</span>
                          <span className="bg-white px-2 py-1 border border-sky-100">{brandSettings.dateLabel}: _________</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 19 - Midnight Architect */}
                  <div 
                    onClick={() => setPaperDesign(18)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 18 ? 'border-slate-800 bg-slate-50 shadow-md' : 'border-slate-200 hover:border-slate-400 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 19: Midnight Architect</h5>
                      {paperDesign === 18 && <div className="h-6 w-6 bg-slate-800 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="bg-slate-800 text-white p-6 flex justify-between items-center">
                        <div>
                          <h1 className="text-lg font-black uppercase tracking-widest">{brandSettings.schoolName}</h1>
                          <p className="text-[8px] opacity-60">Examination Department</p>
                        </div>
                        <div className="text-right text-[8px] font-bold">
                          <div>{brandSettings.studentLabel}: _______</div>
                          <div>{brandSettings.scoreLabel}: ____</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Header Design 20 - Rainbow Gradient */}
                  <div 
                    onClick={() => setPaperDesign(19)}
                    className={`p-8 rounded-2xl border-2 cursor-pointer transition-all ${paperDesign === 19 ? 'border-purple-500 bg-purple-50/30 shadow-md' : 'border-slate-200 hover:border-purple-300 bg-white'}`}
                  >
                    <div className="flex justify-between items-center mb-6">
                      <h5 className="font-bold text-slate-700 uppercase tracking-widest text-xs">Style 20: Rainbow Gradient</h5>
                      {paperDesign === 19 && <div className="h-6 w-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs"><i className="fa-solid fa-check"></i></div>}
                    </div>
                    <div className="bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                      <div className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 text-white p-6 rounded-xl">
                        <h1 className="text-center text-lg font-black uppercase tracking-widest">{brandSettings.schoolName}</h1>
                        <div className="mt-4 flex justify-around text-[8px] font-bold">
                          <span>{brandSettings.studentLabel}: _______</span>
                          <span>{brandSettings.dateLabel}: _______</span>
                          <span>{brandSettings.scoreLabel}: ____</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {viewMode === 'ielts_master' && (
        <section 
          style={{ 
            marginLeft: isSidebarOpen && sidebarSide === 'left' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px',
            marginRight: isSidebarOpen && sidebarSide === 'right' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px'
          }}
          className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-500 bg-slate-50 transition-all duration-300"
        >
          <div className="p-4 lg:p-6 bg-white border-b border-slate-200 flex flex-wrap gap-4 justify-between items-center z-10 no-print shadow-sm">
            <button onClick={() => setViewMode('generator')} className="border border-slate-200 text-slate-600 px-6 lg:px-8 py-3 rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-slate-50 flex items-center gap-4 group transition-all">
              <i className="fa-solid fa-arrow-left group-hover:-translate-x-1 transition-transform"></i> WORKSPACE
            </button>
            <div className="flex-1 text-center">
              <h2 className="text-slate-800 font-bold uppercase tracking-widest text-[12px]">IELTS Mastermind</h2>
            </div>
            <div className="flex gap-2">
              <a 
                href="https://chanthy-master-engine-gbcdawq79gtmzdw7cqfh7f.streamlit.app/?tool=ielts_master&embed=true" 
                target="_blank" 
                rel="noopener noreferrer"
                className="px-6 py-3 bg-orange-600 text-white rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-orange-700 shadow-sm flex items-center gap-2 transition-all"
              >
                <i className="fa-solid fa-arrow-up-right-from-square"></i> Launch Tool
              </a>
            </div>
          </div>
          <div className="flex-1 bg-white overflow-hidden relative">
            <div className="absolute inset-0 flex flex-col items-center justify-center p-10 text-center bg-slate-50 -z-10">
              <i className="fa-solid fa-circle-exclamation text-4xl text-slate-300 mb-4"></i>
              <p className="text-slate-500 font-bold text-sm">If the tool refuses to connect, please use the "Launch Tool" button above.</p>
            </div>
            <iframe 
              src="https://chanthy-master-engine-gbcdawq79gtmzdw7cqfh7f.streamlit.app/?tool=ielts_master&embed=true"
              className="w-full h-full min-h-[800px] border-none"
              title="IELTS Master Tool"
              sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads allow-modals allow-top-navigation-by-user-activation"
            />
          </div>
        </section>
      )}

      {viewMode === 'dpss_studio' && (
        <section 
          style={{ 
            marginLeft: isSidebarOpen && sidebarSide === 'left' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px',
            marginRight: isSidebarOpen && sidebarSide === 'right' ? (windowWidth >= 1024 ? `${sidebarWidth}px` : '0px') : '0px'
          }}
          className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-500 bg-slate-50 transition-all duration-300"
        >
          <div className="p-4 lg:p-6 bg-white border-b border-slate-200 flex flex-wrap gap-4 justify-between items-center z-10 no-print shadow-sm">
            <button onClick={() => setViewMode('generator')} className="border border-slate-200 text-slate-600 px-6 lg:px-8 py-3 rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-slate-50 flex items-center gap-4 group transition-all">
              <i className="fa-solid fa-arrow-left group-hover:-translate-x-1 transition-transform"></i> WORKSPACE
            </button>
            <div className="flex-1 text-center">
              <h2 className="text-slate-800 font-bold uppercase tracking-widest text-[12px]">DPSS Studio</h2>
            </div>
            <div className="flex gap-2">
              <a 
                href="https://chanthy-master-engine-gbcdawq79gtmzdw7cqfh7f.streamlit.app/?tool=dpss_studio&embed=true" 
                target="_blank" 
                rel="noopener noreferrer"
                className="px-6 py-3 bg-orange-600 text-white rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-orange-700 shadow-sm flex items-center gap-2 transition-all"
              >
                <i className="fa-solid fa-arrow-up-right-from-square"></i> Launch Tool
              </a>
            </div>
          </div>
          <div className="flex-1 bg-white overflow-hidden relative">
            <div className="absolute inset-0 flex flex-col items-center justify-center p-10 text-center bg-slate-50 -z-10">
              <i className="fa-solid fa-circle-exclamation text-4xl text-slate-300 mb-4"></i>
              <p className="text-slate-500 font-bold text-sm">If the tool refuses to connect, please use the "Launch Tool" button above.</p>
            </div>
            <iframe 
              src="https://chanthy-master-engine-gbcdawq79gtmzdw7cqfh7f.streamlit.app/?tool=dpss_studio&embed=true"
              className="w-full h-full min-h-[800px] border-none"
              title="DPSS Studio Tool"
              sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads allow-modals allow-top-navigation-by-user-activation"
            />
          </div>
        </section>
      )}
      {!showSettings && isAssistantVisible && (
        <div className="fixed bottom-24 right-6 w-[340px] max-w-[90vw] h-[400px] bg-white/95 backdrop-blur-xl rounded-[40px] border border-slate-100 shadow-2xl overflow-hidden flex flex-col animate-in slide-in-from-bottom-10 z-[200]">
            <div className="p-4 border-b border-slate-50 flex items-center justify-between bg-slate-50/50">
              <div className="flex items-center gap-3">
                <div className="h-2 w-2 bg-orange-500 rounded-full animate-pulse"></div>
                <span className="text-[10px] font-bold uppercase tracking-widest text-slate-800">Live Assistant</span>
              </div>
              <button onClick={() => setIsAssistantVisible(false)} className="h-8 w-8 rounded-full bg-white flex items-center justify-center text-slate-400 hover:text-slate-600 shadow-sm transition-all">
                <i className="fa-solid fa-minus"></i>
              </button>
            </div>
            <div className="flex-1 overflow-hidden p-4">
              <NeuralChatAssistant 
                messages={chatMessages} 
                input={chatInput} 
                onInputChange={setChatInput} 
                onSendMessage={handleAssistantMessage} 
                isGenerating={isGenerating} 
                quickSource={sourceMaterial} 
                inline={true} 
              />
            </div>
        </div>
      )}
      {!showSettings && (
        <button 
          onClick={() => setIsAssistantVisible(!isAssistantVisible)} 
          className={`fixed bottom-6 right-6 h-16 w-16 rounded-full flex items-center justify-center text-white shadow-2xl transition-all z-[200] ${isAssistantVisible ? 'bg-orange-600 rotate-90' : 'bg-slate-800 hover:bg-slate-900'}`}
        >
          <i className={`fa-solid ${isAssistantVisible ? 'fa-xmark' : 'fa-wand-magic-sparkles text-xl'}`}></i>
        </button>
      )}

      {showSettings && (
        <div className={`fixed inset-0 z-[250] bg-slate-950/80 backdrop-blur-2xl flex items-center justify-center ${isSettingsFullScreen ? 'p-0' : 'p-4'}`}>
          <div className={`bg-[#f8fafc] bg-[radial-gradient(circle_at_top_right,rgba(234,88,12,0.03),transparent_40%),radial-gradient(circle_at_bottom_left,rgba(37,99,235,0.03),transparent_40%)] overflow-hidden shadow-2xl flex flex-col border border-white/50 transition-all duration-500 ${isSettingsFullScreen ? 'w-full h-full rounded-none' : 'rounded-[48px] lg:rounded-[64px] w-full max-w-4xl h-full max-h-[75vh]'}`}>
             <div className={`${isSettingsFullScreen && settingsTab === 'FORMAT_DESIGN' ? 'hidden' : 'p-8 lg:p-12 pb-4'} flex justify-between items-center`}>
               <div className="flex items-center gap-4">
                 <div className="h-4 w-4 bg-orange-600 rounded-full animate-pulse"></div>
                 <h2 className="text-[12px] font-black uppercase text-slate-900 tracking-widest">Workspace Control Node</h2>
               </div>
               <div className="flex items-center gap-3">
                 <button 
                   onClick={() => setIsSettingsFullScreen(!isSettingsFullScreen)} 
                   className="h-10 w-10 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-900 transition-all"
                   title={isSettingsFullScreen ? "Exit Full Screen" : "Full Screen"}
                 >
                   <i className={`fa-solid ${isSettingsFullScreen ? 'fa-compress' : 'fa-expand'} text-sm`}></i>
                 </button>
                 <button onClick={() => setShowSettings(false)} className="h-10 w-10 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-900"><i className="fa-solid fa-xmark text-xl"></i></button>
               </div>
             </div>
             <div className={`${isSettingsFullScreen && settingsTab === 'FORMAT_DESIGN' ? 'fixed top-2 left-4 z-[300]' : 'px-6 lg:px-12 mb-8'}`}>
               <div className={`flex bg-slate-100/70 p-1 rounded-[32px] gap-1 overflow-x-auto no-scrollbar shadow-inner ${isSettingsFullScreen && settingsTab === 'FORMAT_DESIGN' ? 'scale-75 origin-left opacity-50 hover:opacity-100 transition-opacity' : ''}`}>
                 {['COMMAND', 'ACCOUNT', 'ENGINE', 'BACKBONE LOGIC', 'DESIGN', 'FORMAT_DESIGN', 'LOGO'].map(tab => (
                   <button 
                     key={tab} 
                     onClick={() => setSettingsTab(tab as SettingsTab)} 
                     className={`px-4 lg:px-6 py-2 rounded-[28px] text-[9px] font-black uppercase tracking-widest transition-all ${settingsTab === tab ? 'bg-orange-600 text-white shadow-xl' : 'text-slate-400 hover:text-slate-600'}`}
                   >
                     {tab === 'FORMAT_DESIGN' ? 'DESIGN TEST FORMAT' : tab.replace('_', ' ')}
                   </button>
                 ))}
               </div>
             </div>
             {isSettingsFullScreen && settingsTab === 'FORMAT_DESIGN' && (
               <button 
                 onClick={() => setShowSettings(false)} 
                 className="fixed top-4 right-4 z-[300] h-10 w-10 bg-white/80 backdrop-blur-md rounded-full flex items-center justify-center text-slate-400 hover:text-slate-900 shadow-lg"
               >
                 <i className="fa-solid fa-xmark text-xl"></i>
               </button>
             )}
             <div className={`flex-1 overflow-y-auto ${isSettingsFullScreen && settingsTab === 'FORMAT_DESIGN' ? 'px-2 pb-2' : 'px-6 lg:px-12 pb-12'} space-y-12 no-scrollbar`}>
                {settingsTab === 'FORMAT_DESIGN' && (
                  <div className="h-full min-h-[600px] animate-in fade-in slide-in-from-bottom-6">
                    <FormatDesignEditor 
                      currentStyle={editingCustomDesignId ? customDesigns.find(d => d.id === editingCustomDesignId)?.style : { mcqStyle, mcqLayout, mcqSpacing, paperStyles }}
                      initialName={editingCustomDesignId ? customDesigns.find(d => d.id === editingCustomDesignId)?.name : undefined}
                      designTargetTypeId={editingCustomDesignId ? customDesigns.find(d => d.id === editingCustomDesignId)?.type : designTargetTypeId}
                      initialCategory={(() => {
                        const targetType = editingCustomDesignId ? customDesigns.find(d => d.id === editingCustomDesignId)?.type : designTargetTypeId;
                        if (targetType === 'matching') return 'Vocabulary';
                        if (targetType === 'correct_incorrect') return 'Grammar';
                        if (targetType === 'true_false') return 'Reading';
                        if (targetType === 'mcq') return 'Grammar';
                        if (targetType === 'vocabulary') return 'Vocabulary';
                        if (targetType === 'reading') return 'Reading';
                        return 'General';
                      })()}
                      onSave={async (design) => {
                        const targetType = editingCustomDesignId ? customDesigns.find(d => d.id === editingCustomDesignId)?.type : designTargetTypeId;
                        console.log('onSave called with targetType:', targetType);
                        
                        // Normalize type for paperStyles mapping
                        let normalizedType = targetType || design.category.toLowerCase();
                        if (normalizedType === 'correct_incorrect') normalizedType = 'correctIncorrect';
                        if (normalizedType === 'true_false') normalizedType = 'tf';
                        if (normalizedType === 'reading') normalizedType = 'readingPassage';
                        if (normalizedType === 'sentence_completion') normalizedType = 'sentenceCompletion';
                        if (normalizedType === 'word_box') normalizedType = 'wordBox';
                        if (normalizedType === 'reading_passage') normalizedType = 'readingPassage';
                        if (normalizedType === 'matching') normalizedType = 'matching';
                        if (normalizedType === 'vocabulary') normalizedType = 'vocabulary';
                        if (normalizedType === 'circle') normalizedType = 'circle';
                        if (normalizedType === 'cloze') normalizedType = 'cloze';
                        if (normalizedType === 'double_mcq') normalizedType = 'doubleMcq';

                        let generatedPrompt = `Apply custom design format for ${design.category}`;
                        if (normalizedType === 'table_style') {
                          const s = design.style;
                          generatedPrompt = `Use a real HTML <table> with ${s.columns} columns. The table MUST have background-color: ${s.backgroundColor}; border: ${s.tableBorderWidth}px ${s.tableBorderStyle} ${s.tableBorderColor}; border-radius: ${s.containerBorderRadius}px; padding: ${s.containerPadding}px; margin: ${s.containerMargin}px. Every <td> MUST have padding: ${s.tablePadding}px; border: ${s.tableBorderWidth}px ${s.tableBorderStyle} ${s.tableBorderColor}; text-align: left. The text MUST have font-family: ${s.fontFamily}; font-size: ${s.fontSize}px; color: ${s.textColor}; font-weight: ${s.fontWeight}; text-decoration: ${s.textDecoration}.`;
                        }

                        if (editingCustomDesignId) {
                          const updatedDesign = {
                            ...customDesigns.find(d => d.id === editingCustomDesignId)!,
                            name: design.name,
                            category: design.category,
                            style: design.style,
                            prompt: generatedPrompt
                          };
                          setCustomDesigns(prev => prev.map(d => d.id === editingCustomDesignId ? updatedDesign : d));
                          
                          if (auth.currentUser) {
                            try {
                              const designRef = doc(db, 'customDesigns', editingCustomDesignId);
                              await setDoc(designRef, updatedDesign, { merge: true });
                            } catch (error) {
                              handleFirestoreError(error, OperationType.WRITE, `customDesigns/${editingCustomDesignId}`);
                            }
                          }
                          setEditingCustomDesignId(null);
                          alert(`Design "${design.name}" updated!`);
                        } else {
                          const newDesign = {
                            id: 'custom_' + Date.now(),
                            name: design.name,
                            type: normalizedType,
                            category: design.category,
                            style: design.style,
                            prompt: generatedPrompt,
                            uid: auth.currentUser?.uid || 'anonymous'
                          };
                          
                          console.log('Saving new design:', newDesign);
                          setCustomDesigns(prev => [...prev, newDesign]);
      
                          // Save to Firestore if logged in
                          if (auth.currentUser) {
                            try {
                              const designRef = doc(db, 'customDesigns', newDesign.id);
                              await setDoc(designRef, {
                                ...newDesign,
                                createdAt: Timestamp.now()
                              });
                            } catch (error) {
                              handleFirestoreError(error, OperationType.WRITE, `customDesigns/${newDesign.id}`);
                            }
                          }
                          
                          // Automatically select the new design
                          const validPaperStyleKeys = ['mcq', 'matching', 'tf', 'correctIncorrect', 'vocabulary', 'readingPassage', 'circle', 'sentenceCompletion', 'wordBox', 'cloze', 'doubleMcq'];
                          if (validPaperStyleKeys.includes(newDesign.type)) {
                            setPaperStyles(prev => ({ ...prev, [newDesign.type]: newDesign.id }));
                          }
                          
                          // If we were designing for a specific type, link it
                          if (designTargetTypeId) {
                            setCustomExerciseTypes(prev => prev.map(t => 
                              t.id === designTargetTypeId ? { ...t, styleId: newDesign.id } : t
                            ));
                            setDesignTargetTypeId(null);
                          }
                          
                          alert(`Design "${design.name}" saved to ${design.category} library!`);
                        }
                        setShowSettings(false);
                      }}
                    />
                  </div>
                )}
                {settingsTab === 'LOGO' && (
                  <div className="space-y-12 animate-in fade-in slide-in-from-bottom-6">
                    <div className="space-y-8">
                      <h3 className="text-[13px] font-black text-slate-900 uppercase tracking-widest">Branding & Logo Registry</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                        <div className="space-y-6">
                          <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">School Identity</label>
                          <input value={brandSettings.schoolName} onChange={e => setBrandSettings({ ...brandSettings, schoolName: e.target.value })} className="w-full bg-slate-100 border border-slate-200 rounded-3xl px-8 py-5 text-[14px] font-black text-slate-900 uppercase focus:border-orange-500 outline-none" placeholder="School Name" />
                          <input value={brandSettings.schoolAddress} onChange={e => setBrandSettings({ ...brandSettings, schoolAddress: e.target.value })} className="w-full bg-slate-100 border border-slate-200 rounded-3xl px-8 py-5 text-[14px] font-black text-slate-900 uppercase focus:border-orange-500 outline-none" placeholder="Address / Motto" />
                        </div>
                        <div className="space-y-6">
                          <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Header Logo (A4 Precision)</label>
                          <div className="border-4 border-dashed border-slate-200 rounded-[48px] p-10 flex flex-col items-center justify-center gap-6 hover:border-orange-500 transition-all cursor-pointer relative" onClick={() => logoUploadRef.current?.click()}>
                            {brandSettings.logoData ? <img src={brandSettings.logoData} className="max-h-24 w-auto rounded-xl" /> : <i className="fa-solid fa-cloud-arrow-up text-4xl text-slate-300"></i>}
                            <span className="text-[10px] font-black text-slate-400 uppercase">Upload Header Graphic</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-8">
                      <div className="flex justify-between items-center px-2">
                        <h3 className="text-[13px] font-black text-slate-900 uppercase tracking-widest">Neural Logo Registry ({brandSettings.logos.filter(l => !!l).length} / {brandSettings.logos.length})</h3>
                        <div className="flex gap-4">
                          <button onClick={() => { if(window.confirm("Clear all logos to free up space?")) setBrandSettings(prev => ({ ...prev, logos: Array(30).fill(undefined) })); }} className="text-[11px] font-black text-rose-500 uppercase border-b-2 border-rose-500">Clear All</button>
                          <button onClick={() => logoUploadRef.current?.click()} className="text-[11px] font-black text-orange-600 uppercase border-b-2 border-orange-600">+ Add Logo</button>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6">
                        {brandSettings.logos.map((logo, idx) => (
                          <div key={idx} className={`aspect-video rounded-3xl border-2 flex items-center justify-center relative group overflow-hidden transition-all ${logo ? 'border-slate-200 bg-white' : 'border-dashed border-slate-100 bg-slate-50/50'}`}>
                            {logo ? (
                              <>
                                <img src={logo} className="max-h-full max-w-full p-4 object-contain" />
                                <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-3">
                                  <button onClick={() => setBrandSettings(prev => ({ ...prev, logoData: logo }))} className="h-10 w-10 bg-white rounded-full flex items-center justify-center text-slate-900 hover:bg-orange-500 hover:text-white transition-all shadow-lg"><i className="fa-solid fa-eye"></i></button>
                                  <button onClick={() => removeLogo(idx)} className="h-10 w-10 bg-rose-500 rounded-full flex items-center justify-center text-white hover:bg-rose-600 transition-all shadow-lg"><i className="fa-solid fa-trash-can"></i></button>
                                </div>
                              </>
                            ) : (
                              <div 
                                onClick={() => logoUploadRef.current?.click()} 
                                className="w-full h-full flex items-center justify-center cursor-pointer group/slot"
                              >
                                <i className="fa-solid fa-plus text-2xl text-slate-200 group-hover/slot:text-orange-500 transition-colors"></i>
                              </div>
                            )}
                            <div className="absolute bottom-3 left-4 text-[8px] font-black text-slate-300 uppercase tracking-widest">Slot {idx + 1}</div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-8 pt-10 border-t border-slate-100">
                      <div className="flex justify-between items-center px-2">
                        <h3 className="text-[13px] font-black text-slate-900 uppercase tracking-widest">Custom Header & Footer Styles</h3>
                        <button 
                          onClick={() => {
                            const styleName = prompt("Enter template name for this header/footer design:");
                            if (styleName) {
                              alert(`Header/Footer style "${styleName}" saved to templates.`);
                            }
                          }}
                          className="text-[11px] font-black text-emerald-600 uppercase border-b-2 border-emerald-600"
                        >
                          Save as Template
                        </button>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="p-6 bg-white border-2 border-slate-200 rounded-3xl hover:border-orange-500 transition-all cursor-pointer group">
                          <div className="flex items-center gap-3 mb-4">
                            <div className="h-8 w-8 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 group-hover:bg-orange-100 group-hover:text-orange-600 transition-all">
                              <i className="fa-solid fa-heading"></i>
                            </div>
                            <span className="text-[10px] font-black uppercase text-slate-900">Classic Centered</span>
                          </div>
                          <div className="h-20 bg-slate-50 rounded-2xl border border-dashed border-slate-200 flex flex-col items-center justify-center gap-1 p-4">
                            <div className="h-2 w-24 bg-slate-300 rounded-full"></div>
                            <div className="h-1.5 w-32 bg-slate-200 rounded-full"></div>
                          </div>
                        </div>
                        <div className="p-6 bg-white border-2 border-slate-200 rounded-3xl hover:border-orange-500 transition-all cursor-pointer group">
                          <div className="flex items-center gap-3 mb-4">
                            <div className="h-8 w-8 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 group-hover:bg-orange-100 group-hover:text-orange-600 transition-all">
                              <i className="fa-solid fa-align-left"></i>
                            </div>
                            <span className="text-[10px] font-black uppercase text-slate-900">Modern Left-Aligned</span>
                          </div>
                          <div className="h-20 bg-slate-50 rounded-2xl border border-dashed border-slate-200 flex flex-col items-start justify-center gap-1 p-4">
                            <div className="h-2 w-24 bg-slate-300 rounded-full"></div>
                            <div className="h-1.5 w-32 bg-slate-200 rounded-full"></div>
                          </div>
                        </div>
                        <div className="p-6 bg-white border-2 border-slate-200 rounded-3xl hover:border-orange-500 transition-all cursor-pointer group">
                          <div className="flex items-center gap-3 mb-4">
                            <div className="h-8 w-8 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 group-hover:bg-orange-100 group-hover:text-orange-600 transition-all">
                              <i className="fa-solid fa-columns"></i>
                            </div>
                            <span className="text-[10px] font-black uppercase text-slate-900">Split Header</span>
                          </div>
                          <div className="h-20 bg-slate-50 rounded-2xl border border-dashed border-slate-200 flex items-center justify-between p-4">
                            <div className="h-8 w-8 bg-slate-300 rounded-lg"></div>
                            <div className="flex flex-col items-end gap-1">
                              <div className="h-2 w-16 bg-slate-300 rounded-full"></div>
                              <div className="h-1.5 w-20 bg-slate-200 rounded-full"></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                {settingsTab === 'COMMAND' && (
                   <div className="space-y-10 animate-in fade-in slide-in-from-bottom-6">
                     <div className="flex justify-between items-center px-2"><h3 className="text-[13px] font-black text-slate-900 uppercase tracking-widest">Instruction Templates</h3><button onClick={addTemplate} className="text-[11px] font-black text-orange-600 uppercase border-b-2 border-orange-600">+ New Part</button></div>
                     <div className="flex bg-slate-100/50 p-1.5 rounded-[24px] gap-1 overflow-x-auto no-scrollbar shadow-sm border border-slate-100 self-start">{['GRAMMAR', 'VOCABULARY', 'READING', 'GENERALS', 'TABLES', 'KIDS'].map(cat => (<button key={cat} onClick={() => setActiveTemplateCategory(cat)} className={`px-6 py-2.5 rounded-[18px] text-[10px] font-black uppercase tracking-widest transition-all ${activeTemplateCategory === cat ? 'bg-orange-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-600'}`}>{cat}</button>))}</div>
                     <div className="space-y-3">
                        {instructionTemplates.filter(t => t.category === activeTemplateCategory).map(t => {
                            const isExpanded = expandedTemplateId === t.id;
                            return (
                              <div key={t.id} className={`bg-white border rounded-[32px] overflow-hidden transition-all duration-300 ${isExpanded ? 'border-orange-200 shadow-xl' : 'border-slate-100 shadow-sm'}`}>
                                 <div className="p-6 lg:p-8 cursor-pointer flex items-center justify-between" onClick={() => setExpandedTemplateId(isExpanded ? null : t.id)}><div className="flex items-center gap-4 flex-1"><div className={`h-8 w-8 rounded-full flex items-center justify-center transition-transform ${isExpanded ? 'rotate-90 bg-orange-600 text-white' : 'bg-slate-50 text-slate-400'}`}><i className="fa-solid fa-chevron-right text-[10px]"></i></div><div className="flex flex-col gap-0.5"><div className={`text-[13px] font-black uppercase tracking-wide transition-colors ${isExpanded ? 'text-orange-600' : 'text-slate-900'}`}>{t.label}</div>{!isExpanded && <div className="text-[9px] font-black text-slate-300 uppercase line-clamp-1">{t.prompt.slice(0, 100)}...</div>}</div></div><div className="flex items-center gap-3"><div className="px-4 py-1.5 rounded-full bg-slate-100 text-slate-400 text-[8px] font-black uppercase">{t.category}</div>{isExpanded && <button onClick={() => deleteTemplate(t.id)} className="h-8 w-8 bg-rose-50 text-rose-500 rounded-full flex items-center justify-center hover:bg-rose-500 hover:text-white transition-all"><i className="fa-solid fa-trash-can text-[10px]"></i></button>}</div></div>
                                 {isExpanded && (<div className="px-8 pb-8 space-y-6 animate-in fade-in slide-in-from-top-4"><div className="h-px bg-slate-100 w-full mb-6"></div><div className="space-y-4"><label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Display Name</label><input value={t.label} onChange={e => updateTemplate(t.id, { label: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-orange-500 font-bold text-slate-700" /></div><div className="space-y-4"><label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Neural Prompt Logic</label><textarea value={t.prompt} onChange={e => updateTemplate(t.id, { prompt: e.target.value })} className="w-full h-32 bg-slate-50 border border-slate-100 rounded-3xl p-6 text-[11px] text-slate-600 font-medium italic outline-none resize-none focus:bg-white transition-all" /></div></div>)}
                              </div>
                            );
                        })}
                     </div>
                   </div>
                )}
                {settingsTab === 'ENGINE' && (
                  <div className="space-y-10 animate-in fade-in slide-in-from-bottom-6">
                    <h3 className="text-[13px] font-black text-slate-900 uppercase tracking-widest">Neural Core Configuration</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {[
                        { id: NeuralEngine.GEMINI_3_FLASH_LITE, name: 'Gemini 3.1 Flash Lite', desc: 'Ultra-fast, low-latency generation.' },
                        { id: NeuralEngine.GEMINI_3_FLASH, name: 'Gemini 3 Flash', desc: 'High-speed, balanced reasoning.' },
                        { id: NeuralEngine.GEMINI_3_PRO, name: 'Gemini 3 Pro', desc: 'Maximum intelligence for complex tests.' },
                        { id: NeuralEngine.GPT_4O, name: 'GPT-4o', desc: 'Advanced multimodal capabilities.' },
                        { id: NeuralEngine.GROK_3, name: 'Grok 3', desc: 'Real-time knowledge and reasoning.' },
                        { id: NeuralEngine.DEEPSEEK_V3, name: 'DeepSeek V3', desc: 'Efficient large-scale processing.' }
                      ].map(engine => (
                        <div key={engine.id} className={`p-8 rounded-[40px] border-2 transition-all ${activeEngine === engine.id ? 'bg-white border-orange-600 shadow-xl' : 'bg-slate-50 border-slate-100'}`}>
                          <div className="flex justify-between items-start mb-6">
                            <div className="space-y-1">
                              <div className="text-[14px] font-black text-slate-900 uppercase">{engine.name}</div>
                              <div className="text-[10px] font-medium text-slate-400">{engine.desc}</div>
                            </div>
                            {activeEngine === engine.id && <div className="h-6 w-6 bg-orange-600 rounded-full flex items-center justify-center text-white text-[10px]"><i className="fa-solid fa-check"></i></div>}
                          </div>
                          <div className="space-y-4">
                            {(engine.id === NeuralEngine.GEMINI_3_FLASH_LITE || engine.id === NeuralEngine.GEMINI_3_FLASH || engine.id === NeuralEngine.GEMINI_3_PRO) && (
                              <button 
                                onClick={async () => {
                                  if ((window as any).aistudio?.openSelectKey) {
                                    await (window as any).aistudio.openSelectKey();
                                  } else {
                                    alert("The 'Select AI Studio Key' feature only works when you are using the app inside the AI Studio preview pane. \n\nIf you are viewing the app at the Shared URL directly, you must set a 'GEMINI_API_KEY' in your environment variables for it to work standalone.");
                                  }
                                }}
                                className="w-full bg-slate-100 border border-slate-200 text-slate-600 py-3 rounded-2xl text-[10px] font-black uppercase tracking-widest hover:bg-slate-200 transition-all flex items-center justify-center gap-2"
                              >
                                <i className="fa-solid fa-key"></i>
                                Select AI Studio Key
                              </button>
                            )}
                            <input 
                              type="password"
                              value={externalKeys[engine.id as keyof ExternalKeys] || ''} 
                              onChange={e => setExternalKeys({ ...externalKeys, [engine.id]: e.target.value })}
                              placeholder={ (engine.id === NeuralEngine.GEMINI_3_FLASH_LITE || engine.id === NeuralEngine.GEMINI_3_FLASH || engine.id === NeuralEngine.GEMINI_3_PRO) ? "Or Paste Custom Gemini Key" : "Custom API Key (Optional)" }
                              className="w-full bg-white border border-slate-200 rounded-2xl px-6 py-3 text-[11px] outline-none focus:border-orange-500"
                            />
                            <div className="flex gap-2">
                              <button 
                                onClick={() => setActiveEngine(engine.id as NeuralEngine)}
                                className={`flex-1 py-3 rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all ${activeEngine === engine.id ? 'bg-orange-600 text-white shadow-lg' : 'bg-white border border-slate-200 text-slate-400 hover:text-slate-900'}`}
                              >
                                {activeEngine === engine.id ? 'Currently Active' : 'Switch Engine'}
                              </button>
                              {externalKeys[engine.id as keyof ExternalKeys] && (
                                <button 
                                  onClick={() => {
                                    const newKeys = { ...externalKeys };
                                    delete newKeys[engine.id as keyof ExternalKeys];
                                    setExternalKeys(newKeys);
                                  }}
                                  className="px-4 py-3 bg-red-50 text-red-600 rounded-2xl text-[10px] font-black uppercase hover:bg-red-100 transition-all"
                                  title="Clear Custom Key"
                                >
                                  <i className="fa-solid fa-trash-can"></i>
                                </button>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {settingsTab === 'DESIGN' && (
                  <div className="space-y-10 animate-in fade-in slide-in-from-bottom-6">
                    <h3 className="text-[13px] font-black text-slate-900 uppercase tracking-widest">Typography & Layout</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                      <div className="space-y-6">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Primary Font</label>
                        <div className="grid grid-cols-2 gap-2 max-h-[200px] overflow-y-auto p-2 bg-slate-100 rounded-2xl no-scrollbar">
                          {FONTS.map(font => (
                            <button 
                              key={font.name} 
                              onClick={() => setBrandSettings({ ...brandSettings, activeFont: font.name })} 
                              className={`py-2 px-3 rounded-xl text-[10px] font-black transition-all text-left truncate ${brandSettings.activeFont === font.name ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-400 hover:text-slate-600'}`}
                              style={{ fontFamily: font.family }}
                            >
                              {font.name}
                            </button>
                          ))}
                        </div>
                        <div className="flex items-center justify-between px-2">
                          <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Randomize on Generate</span>
                          <button 
                            onClick={() => setBrandSettings({ ...brandSettings, randomizeFont: !brandSettings.randomizeFont })}
                            className={`w-12 h-6 rounded-full transition-all relative ${brandSettings.randomizeFont ? 'bg-orange-600' : 'bg-slate-200'}`}
                          >
                            <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${brandSettings.randomizeFont ? 'left-7' : 'left-1'}`}></div>
                          </button>
                        </div>
                      </div>
                      <div className="space-y-6">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Font Size (px)</label>
                        <div className="flex items-center gap-4">
                          <input type="range" min="8" max="24" value={brandSettings.fontSize} onChange={e => setBrandSettings({ ...brandSettings, fontSize: parseInt(e.target.value) })} className="flex-1 accent-orange-600" />
                          <span className="text-xl font-black text-slate-900 w-12">{brandSettings.fontSize}</span>
                        </div>
                      </div>
                      <div className="space-y-6">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Font Weight</label>
                        <div className="flex bg-slate-100 p-1.5 rounded-2xl gap-1">
                          {['400', '500', '600', '700', '800', '900'].map(weight => (
                            <button key={weight} onClick={() => setBrandSettings({ ...brandSettings, fontWeight: weight })} className={`flex-1 py-2 rounded-xl text-[10px] font-black transition-all ${brandSettings.fontWeight === weight ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-400'}`}>{weight}</button>
                          ))}
                        </div>
                      </div>
                      <div className="space-y-6">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Letter Spacing</label>
                        <div className="flex items-center gap-4">
                          <input type="range" min="-2" max="10" step="0.5" value={brandSettings.letterSpacing} onChange={e => setBrandSettings({ ...brandSettings, letterSpacing: parseFloat(e.target.value) })} className="flex-1 accent-orange-600" />
                          <span className="text-xl font-black text-slate-900 w-12">{brandSettings.letterSpacing}</span>
                        </div>
                      </div>
                      <div className="space-y-6">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Text Transform</label>
                        <div className="flex bg-slate-100 p-1.5 rounded-2xl gap-1">
                          {['none', 'uppercase', 'capitalize'].map(transform => (
                            <button key={transform} onClick={() => setBrandSettings({ ...brandSettings, textTransform: transform as any })} className={`flex-1 py-2 rounded-xl text-[10px] font-black uppercase transition-all ${brandSettings.textTransform === transform ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-400'}`}>{transform}</button>
                          ))}
                        </div>
                      </div>
                    </div>
                    <div className="space-y-6">
                      <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Worksheet Theme</label>
                      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                        {THEMES.map(theme => (
                          <button key={theme.id} onClick={() => setActiveThemeId(theme.id)} className={`p-4 rounded-2xl border-2 transition-all text-left space-y-2 ${activeThemeId === theme.id ? 'border-orange-600 bg-white shadow-lg' : 'border-slate-100 bg-slate-50 hover:border-slate-200'}`}>
                            <div className="w-full h-2 rounded-full" style={{ backgroundColor: theme.color }}></div>
                            <div className="text-[10px] font-black uppercase text-slate-900 truncate">{theme.name}</div>
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-8 pt-6 border-t border-slate-100">
                      <h3 className="text-[13px] font-black text-slate-900 uppercase tracking-widest">Header & Footer Customization</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="space-y-4">
                          <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Custom Header Text (Overrides Default)</label>
                          <input value={brandSettings.customHeaderText || ''} onChange={e => setBrandSettings({ ...brandSettings, customHeaderText: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-orange-500 font-bold text-slate-700" placeholder="e.g. FINAL TERM EXAMINATION - SEMESTER 1" />
                        </div>
                        <div className="space-y-4">
                          <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Header Ruler Style</label>
                          <div className="flex bg-slate-100 p-1.5 rounded-2xl gap-1 overflow-x-auto no-scrollbar">
                            {[0, 1, 2, 3, 4, 5, 6].map(style => (
                              <button key={style} onClick={() => setBrandSettings({ ...brandSettings, headerRulerStyle: style })} className={`px-4 py-2 rounded-xl text-[10px] font-black transition-all shrink-0 ${brandSettings.headerRulerStyle === style ? 'bg-white text-orange-600 shadow-sm' : 'text-slate-400'}`}>
                                {style === 0 ? 'None' : `Ruler ${style}`}
                              </button>
                            ))}
                          </div>
                        </div>
                        <div className="space-y-4">
                          <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Main Title (School Name)</label>
                          <input value={brandSettings.schoolName} onChange={e => setBrandSettings({ ...brandSettings, schoolName: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-orange-500 font-bold text-slate-700" placeholder="e.g. HARVARD ACADEMY" />
                        </div>
                        <div className="space-y-4">
                          <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Footer Text</label>
                          <input value={brandSettings.footerText} onChange={e => setBrandSettings({ ...brandSettings, footerText: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-orange-500 font-bold text-slate-700" placeholder="e.g. Confidential - Academic Use Only" />
                        </div>
                        <div className="space-y-4">
                          <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Student Label</label>
                          <input value={brandSettings.studentLabel} onChange={e => setBrandSettings({ ...brandSettings, studentLabel: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-orange-500 font-bold text-slate-700" placeholder="e.g. NAME" />
                        </div>
                        <div className="space-y-4">
                          <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">ID Label</label>
                          <input value={brandSettings.idLabel} onChange={e => setBrandSettings({ ...brandSettings, idLabel: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-orange-500 font-bold text-slate-700" placeholder="e.g. STUDENT ID" />
                        </div>
                        <div className="space-y-4">
                          <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Score Label</label>
                          <input value={brandSettings.scoreLabel} onChange={e => setBrandSettings({ ...brandSettings, scoreLabel: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-orange-500 font-bold text-slate-700" placeholder="e.g. TOTAL SCORE" />
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-4">
                            <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Date Label</label>
                            <input value={brandSettings.dateLabel} onChange={e => setBrandSettings({ ...brandSettings, dateLabel: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-orange-500 font-bold text-slate-700" placeholder="DATE" />
                          </div>
                          <div className="space-y-4">
                            <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Class Label</label>
                            <input value={brandSettings.classLabel} onChange={e => setBrandSettings({ ...brandSettings, classLabel: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-orange-500 font-bold text-slate-700" placeholder="CLASS" />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                {settingsTab === 'BACKBONE LOGIC' && (
                  <div className="space-y-12 animate-in fade-in slide-in-from-bottom-6 duration-700">
                    <div className="space-y-8">
                       <div className="flex justify-between items-center px-2">
                         <h3 className="text-[13px] font-black text-master-green uppercase tracking-widest">Master Protocols</h3>
                         {!(session?.code === 'dpss' || session?.code === 'gratitude' || session?.code === 'virtues') && (
                           <div className="flex items-center gap-2 text-rose-500 animate-pulse">
                             <i className="fa-solid fa-lock text-[10px]"></i>
                             <span className="text-[10px] font-black uppercase tracking-widest">Restricted Access</span>
                           </div>
                         )}
                         {(session?.code === 'dpss' || session?.code === 'gratitude' || session?.code === 'virtues') && (
                           <button onClick={addProtocol} className="text-[11px] font-black text-master-green uppercase border-b-2 border-master-green">+ New Protocol</button>
                         )}
                       </div>
                       {(session?.code === 'dpss' || session?.code === 'gratitude' || session?.code === 'virtues') ? (
                         <>
                           <div className="flex bg-slate-100/50 p-1.5 rounded-[24px] gap-1 overflow-x-auto no-scrollbar shadow-sm border border-slate-100 self-start">
                             {['General', 'Grammar', 'Vocabulary', 'Reading', 'Generals'].map(cat => (
                               <button key={cat} onClick={() => setActiveProtocolCategory(cat as RuleCategory)} className={`px-6 py-2.5 rounded-[18px] text-[10px] font-black uppercase tracking-widest transition-all ${activeProtocolCategory === cat ? 'bg-master-green text-white shadow-md' : 'text-slate-400 hover:text-slate-600'}`}>{cat}</button>
                             ))}
                           </div>
                           <div className="space-y-3">
                             {masterProtocols.filter(p => p.category === activeProtocolCategory).map(p => {
                               const isExpanded = expandedProtocolId === p.id;
                               return (
                                 <div key={p.id} className={`bg-white border rounded-[32px] overflow-hidden transition-all duration-300 ${isExpanded ? 'border-master-green/30 shadow-xl' : 'border-slate-100 shadow-sm'}`}>
                                   <div className="p-6 lg:p-8 cursor-pointer flex items-center justify-between" onClick={() => setExpandedProtocolId(isExpanded ? null : p.id)}>
                                     <div className="flex items-center gap-4 flex-1">
                                       <div className={`h-8 w-8 rounded-full flex items-center justify-center transition-transform ${isExpanded ? 'rotate-90 bg-master-green text-white' : 'bg-slate-50 text-slate-400'}`}>
                                         <i className="fa-solid fa-chevron-right text-[10px]"></i>
                                       </div>
                                       <div className="flex flex-col gap-0.5">
                                         <div className={`text-[13px] font-black uppercase tracking-wide transition-colors ${isExpanded ? 'text-master-green' : 'text-slate-900'}`}>{p.label}</div>
                                         {!isExpanded && <div className="text-[9px] font-black text-slate-300 uppercase line-clamp-1">{p.promptInjection.slice(0, 100)}...</div>}
                                       </div>
                                     </div>
                                     <div className="flex items-center gap-3">
                                       <button 
                                         onClick={(e) => { e.stopPropagation(); updateProtocol(p.id, { priority: cyclePriority(p.priority) }); }}
                                         className={`px-3 py-1 rounded-full text-[8px] font-black uppercase transition-all hover:scale-105 ${p.priority === 'High' ? 'bg-rose-100 text-rose-600' : p.priority === 'Medium' ? 'bg-orange-100 text-orange-600' : p.priority === 'Average' ? 'bg-blue-100 text-blue-600' : 'bg-slate-100 text-slate-400'}`}
                                       >
                                         {p.priority}
                                       </button>
                                       <button 
                                         onClick={(e) => { e.stopPropagation(); updateProtocol(p.id, { active: !p.active }); }} 
                                         className={`px-4 py-1.5 rounded-full text-[8px] font-black uppercase transition-all ${p.active ? 'bg-master-green/10 text-master-green' : 'bg-slate-100 text-slate-400'}`}
                                       >
                                         {p.active ? 'Active' : 'Disabled'}
                                       </button>
                                       {isExpanded && <button onClick={(e) => { e.stopPropagation(); deleteProtocol(p.id); }} className="h-8 w-8 bg-rose-50 text-rose-500 rounded-full flex items-center justify-center hover:bg-rose-500 hover:text-white transition-all"><i className="fa-solid fa-trash-can text-[10px]"></i></button>}
                                     </div>
                                   </div>
                                   {isExpanded && (
                                     <div className="px-8 pb-8 space-y-6 animate-in fade-in slide-in-from-top-4">
                                       <div className="h-px bg-slate-100 w-full mb-6"></div>
                                       <div className="grid grid-cols-2 gap-4">
                                          <div className="space-y-4">
                                            <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Protocol Name</label>
                                            <input value={p.label} onChange={e => updateProtocol(p.id, { label: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-master-green font-bold text-slate-700" />
                                          </div>
                                          <div className="space-y-4">
                                            <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Priority Level</label>
                                            <button 
                                              onClick={() => updateProtocol(p.id, { priority: cyclePriority(p.priority) })}
                                              className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none hover:border-master-green font-bold text-slate-700 uppercase text-left flex justify-between items-center"
                                            >
                                              <span>{p.priority}</span>
                                              <i className="fa-solid fa-rotate text-[10px] text-slate-300"></i>
                                            </button>
                                          </div>
                                       </div>
                                       <div className="space-y-4">
                                         <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Protocol Logic</label>
                                         <textarea value={p.promptInjection} onChange={e => updateProtocol(p.id, { promptInjection: e.target.value })} className="w-full h-32 bg-slate-50 border border-slate-100 rounded-3xl p-6 text-[11px] text-slate-600 font-medium italic outline-none resize-none focus:bg-white transition-all" />
                                       </div>
                                     </div>
                                   )}
                                 </div>
                               );
                             })}
                           </div>
                         </>
                       ) : (
                         <div className="p-12 border-2 border-dashed border-slate-100 rounded-[40px] flex flex-col items-center justify-center gap-4 bg-slate-50/50">
                           <i className="fa-solid fa-shield-halved text-slate-200 text-4xl"></i>
                           <div className="text-[11px] font-black text-slate-400 uppercase tracking-widest">Neural Protocols Encrypted</div>
                           <div className="text-[9px] font-medium text-slate-400 text-center max-w-[200px]">Please authenticate with a Master code to modify core protocols.</div>
                         </div>
                       )}
                    </div>
                     <div className="space-y-8">
                        <div className="flex justify-between items-center px-2">
                          <h3 className="text-[13px] font-black text-strict-purple uppercase tracking-widest">Logic Node Registry</h3>
                          <button onClick={addRule} className="text-[11px] font-black text-strict-purple uppercase border-b-2 border-strict-purple">+ New Logic Node</button>
                        </div>
                        <div className="flex bg-slate-100/50 p-1.5 rounded-[24px] gap-1 overflow-x-auto no-scrollbar shadow-sm border border-slate-100 self-start">
                          {['General', 'Grammar', 'Vocabulary', 'Reading', 'Generals'].map(cat => (
                            <button key={cat} onClick={() => setActiveLogicCategory(cat as RuleCategory)} className={`px-6 py-2.5 rounded-[18px] text-[10px] font-black uppercase tracking-widest transition-all ${activeLogicCategory === cat ? 'bg-strict-purple text-white shadow-md' : 'text-slate-400 hover:text-slate-600'}`}>{cat}</button>
                          ))}
                        </div>
                        <div className="space-y-3">
                          {strictRules.filter(rule => rule.category === activeLogicCategory).map(rule => {
                            const isExpanded = expandedRuleId === rule.id;
                            return (
                              <div key={rule.id} className={`bg-white border rounded-[32px] overflow-hidden transition-all duration-300 ${isExpanded ? 'border-strict-purple/30 shadow-xl' : 'border-slate-100 shadow-sm'}`}>
                                <div className="p-6 lg:p-8 cursor-pointer flex items-center justify-between" onClick={() => setExpandedRuleId(isExpanded ? null : rule.id)}>
                                  <div className="flex items-center gap-4 flex-1">
                                    <div className={`h-8 w-8 rounded-full flex items-center justify-center transition-transform ${isExpanded ? 'rotate-90 bg-strict-purple text-white' : 'bg-slate-50 text-slate-400'}`}>
                                      <i className="fa-solid fa-chevron-right text-[10px]"></i>
                                    </div>
                                    <div className="flex flex-col gap-0.5">
                                      <div className={`text-[13px] font-black uppercase tracking-wide transition-colors ${isExpanded ? 'text-strict-purple' : 'text-slate-900'}`}>{rule.label}</div>
                                      {!isExpanded && <div className="text-[9px] font-black text-slate-300 uppercase line-clamp-1">{rule.promptInjection.slice(0, 100)}...</div>}
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-3">
                                    <button 
                                      onClick={(e) => { e.stopPropagation(); updateRule(rule.id, { priority: cyclePriority(rule.priority) }); }}
                                      className={`px-3 py-1 rounded-full text-[8px] font-black uppercase transition-all hover:scale-105 ${rule.priority === 'High' ? 'bg-rose-100 text-rose-600' : rule.priority === 'Medium' ? 'bg-orange-100 text-orange-600' : rule.priority === 'Average' ? 'bg-blue-100 text-blue-600' : 'bg-slate-100 text-slate-400'}`}
                                    >
                                      {rule.priority}
                                    </button>
                                    <button 
                                      onClick={(e) => { e.stopPropagation(); updateRule(rule.id, { active: !rule.active }); }} 
                                      className={`px-4 py-1.5 rounded-full text-[8px] font-black uppercase transition-all ${rule.active ? 'bg-strict-purple/10 text-strict-purple' : 'bg-slate-100 text-slate-400'}`}
                                    >
                                      {rule.active ? 'Active' : 'Disabled'}
                                    </button>
                                    {isExpanded && <button onClick={(e) => { e.stopPropagation(); deleteRule(rule.id); }} className="h-8 w-8 bg-rose-50 text-rose-500 rounded-full flex items-center justify-center hover:bg-rose-500 hover:text-white transition-all"><i className="fa-solid fa-trash-can text-[10px]"></i></button>}
                                  </div>
                                </div>
                                {isExpanded && (
                                  <div className="px-8 pb-8 space-y-6 animate-in fade-in slide-in-from-top-4">
                                    <div className="h-px bg-slate-100 w-full mb-6"></div>
                                    <div className="grid grid-cols-2 gap-4">
                                       <div className="space-y-4">
                                         <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Logic Name</label>
                                         <input value={rule.label} onChange={e => updateRule(rule.id, { label: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-strict-purple font-bold text-slate-700" />
                                       </div>
                                       <div className="space-y-4">
                                         <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Priority Level</label>
                                         <button 
                                           onClick={() => updateRule(rule.id, { priority: cyclePriority(rule.priority) })}
                                           className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none hover:border-strict-purple font-bold text-slate-700 uppercase text-left flex justify-between items-center"
                                         >
                                           <span>{rule.priority}</span>
                                           <i className="fa-solid fa-rotate text-[10px] text-slate-300"></i>
                                         </button>
                                       </div>
                                    </div>
                                    <div className="space-y-4">
                                      <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Prompt Injection</label>
                                      <textarea value={rule.promptInjection} onChange={e => updateRule(rule.id, { promptInjection: e.target.value })} className="w-full h-32 bg-slate-50 border border-slate-100 rounded-3xl p-6 text-[11px] text-slate-600 font-medium italic outline-none resize-none focus:bg-white transition-all" />
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                     </div>
                  </div>
                )}
                {settingsTab === 'ACCOUNT' && (
                  <div className="space-y-12 animate-in fade-in slide-in-from-bottom-6 duration-700">
                    {!isFirebaseConnected && (
                      <div className="bg-rose-50 border border-rose-100 p-8 rounded-[32px] space-y-4">
                        <div className="flex items-center gap-4 text-rose-600">
                          <i className="fa-solid fa-triangle-exclamation text-2xl"></i>
                          <h4 className="font-black uppercase tracking-widest text-sm">Cloud Connection Error</h4>
                        </div>
                        <p className="text-rose-500 text-[11px] font-bold leading-relaxed">
                          Your application is unable to connect to the Firebase cloud. This usually happens if you haven't set up your environment variables (like GEMINI_API_KEY or Firebase config) on your hosting provider (e.g., Vercel).
                          <br/><br/>
                          If you are seeing this on a published site, please ensure you have copied the <code className="bg-rose-100 px-2 py-0.5 rounded">firebase-applet-config.json</code> values to your environment.
                        </p>
                      </div>
                    )}
                    <div className="bg-white p-10 rounded-[48px] border border-slate-100 shadow-sm space-y-8">
                      <div className="flex items-center gap-6">
                        <div className="h-16 w-16 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 text-2xl">
                          <i className="fa-solid fa-cloud"></i>
                        </div>
                        <div>
                          <h3 className="text-[16px] font-black text-slate-900 uppercase tracking-widest">Cloud Sync Status</h3>
                          <p className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">Sync your branding and history across devices</p>
                        </div>
                      </div>

                      <div className="h-px bg-slate-100 w-full"></div>

                      {session?.email && session.email !== 'public@dpss.edu' ? (
                        <div className="space-y-6">
                          <div className="flex items-center justify-between p-6 bg-slate-50 rounded-3xl border border-slate-100">
                            <div className="flex items-center gap-4">
                              <div className="h-12 w-12 bg-orange-100 rounded-full flex items-center justify-center text-orange-600 font-black">
                                {session.name.charAt(0)}
                              </div>
                              <div>
                                <div className="text-[13px] font-black text-slate-900 uppercase">{session.name}</div>
                                <div className="text-[10px] font-medium text-slate-400">{session.email}</div>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="h-2 w-2 bg-green-500 rounded-full"></div>
                              <span className="text-[10px] font-black text-green-600 uppercase tracking-widest">Connected</span>
                            </div>
                          </div>
                          <button 
                            onClick={handleLogout}
                            className="w-full py-5 rounded-3xl bg-rose-50 text-rose-600 text-[11px] font-black uppercase tracking-widest hover:bg-rose-100 transition-all flex items-center justify-center gap-3"
                          >
                            <i className="fa-solid fa-right-from-bracket"></i>
                            Disconnect Cloud Account
                          </button>
                        </div>
                      ) : (
                        <div className="space-y-8">
                          <div className="p-8 bg-orange-50 rounded-[32px] border border-orange-100 text-center space-y-4">
                            <i className="fa-solid fa-shield-halved text-3xl text-orange-500"></i>
                            <div className="text-[13px] font-black text-slate-900 uppercase tracking-wide">Cloud Storage Disabled</div>
                            <p className="text-[11px] font-medium text-slate-500 max-w-md mx-auto">Sign in with your DPSS account to automatically save your brand settings, logos, and worksheet history to the cloud.</p>
                          </div>
                          <button 
                            onClick={handleGoogleLogin}
                            disabled={authLoading}
                            className="w-full py-6 rounded-[32px] bg-slate-900 text-white text-[12px] font-black uppercase tracking-widest hover:bg-black transition-all flex items-center justify-center gap-4 shadow-xl disabled:opacity-50"
                          >
                            {authLoading ? (
                              <i className="fa-solid fa-circle-notch fa-spin"></i>
                            ) : (
                              <i className="fa-brands fa-google"></i>
                            )}
                            Connect with Google Cloud
                          </button>
                        </div>
                      )}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      <div className="bg-slate-50 p-8 rounded-[40px] border border-slate-100 space-y-4">
                        <div className="h-10 w-10 bg-white rounded-2xl flex items-center justify-center text-orange-600 shadow-sm">
                          <i className="fa-solid fa-palette"></i>
                        </div>
                        <div className="text-[12px] font-black text-slate-900 uppercase tracking-widest">Brand Persistence</div>
                        <p className="text-[10px] font-medium text-slate-400 leading-relaxed">Your school name, address, and logo collection are automatically synced. No more re-uploading logos on different computers.</p>
                      </div>
                      <div className="bg-slate-50 p-8 rounded-[40px] border border-slate-100 space-y-4">
                        <div className="h-10 w-10 bg-white rounded-2xl flex items-center justify-center text-blue-600 shadow-sm">
                          <i className="fa-solid fa-clock-rotate-left"></i>
                        </div>
                        <div className="text-[12px] font-black text-slate-900 uppercase tracking-widest">Infinite History</div>
                        <p className="text-[10px] font-medium text-slate-400 leading-relaxed">Access your generated tests from anywhere. Your history is stored securely in your private cloud partition.</p>
                      </div>
                    </div>
                  </div>
                )}
             </div>
              <div className="p-12 bg-slate-50 border-t border-slate-100 flex flex-col items-center gap-8">
                <div className="flex items-center gap-4">
                  <button 
                    onClick={() => setIsBottomPanelHidden(!isBottomPanelHidden)}
                    className="px-3 py-1 bg-slate-100 text-slate-500 rounded-full text-[8px] font-black uppercase hover:bg-slate-200 transition-all"
                  >
                    {isBottomPanelHidden ? 'Show Panel' : 'Hide Panel'}
                  </button>
                </div>

                {!isBottomPanelHidden && (
                  <div className="flex justify-end gap-4 w-full">
                    <button onClick={hardReset} className="px-16 py-6 bg-rose-600 text-white rounded-full text-[12px] font-black uppercase shadow-xl hover:bg-rose-700 transition-all">Hard Reset</button>
                    <button onClick={syncWithDefaults} className="px-16 py-6 bg-slate-900 text-white rounded-full text-[12px] font-black uppercase shadow-xl hover:bg-black transition-all">Sync Settings</button>
                    <button onClick={() => setShowSettings(false)} className="px-16 py-6 bg-gradient-to-r from-accent-orange-dark to-accent-orange-light text-white rounded-full text-[12px] font-black uppercase shadow-xl hover:brightness-110 transition-all">Close Panel</button>
                  </div>
                )}
              </div>
          </div>
        </div>
      )}
      {/* EXPORT SETTINGS MODAL */}
      {exportSettings.showModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-6">
          <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-md" onClick={() => setExportSettings(prev => ({ ...prev, showModal: false }))}></div>
          <div className="relative w-full max-w-md bg-white rounded-[48px] shadow-2xl overflow-hidden animate-in zoom-in-95 duration-300">
            <div className="p-10">
              <div className="flex items-center gap-4 mb-8">
                <div className="h-12 w-12 bg-orange-100 text-orange-600 rounded-2xl flex items-center justify-center">
                  <i className="fa-solid fa-file-word text-xl"></i>
                </div>
                <div>
                  <h3 className="text-lg font-black uppercase tracking-tight text-slate-900">Export Settings</h3>
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Customize your Word document</p>
                </div>
              </div>

              <div className="space-y-6">
                <div className="space-y-2">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Filename</label>
                  <div className="relative">
                    <input 
                      type="text" 
                      value={exportSettings.filename} 
                      onChange={e => setExportSettings(prev => ({ ...prev, filename: e.target.value }))}
                      className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-orange-500 font-bold text-slate-700 pr-16"
                      placeholder="Enter filename..."
                    />
                    <span className="absolute right-6 top-1/2 -translate-y-1/2 text-[10px] font-black text-slate-300 uppercase">.doc</span>
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Document Title</label>
                  <input 
                    type="text" 
                    value={exportSettings.title} 
                    onChange={e => setExportSettings(prev => ({ ...prev, title: e.target.value }))}
                    className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-orange-500 font-bold text-slate-700"
                    placeholder="Enter title..."
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mt-10">
                <button 
                  onClick={() => setExportSettings(prev => ({ ...prev, showModal: false }))}
                  className="py-5 bg-slate-100 text-slate-500 rounded-3xl text-[11px] font-black uppercase tracking-widest hover:bg-slate-200 transition-all"
                >
                  Cancel
                </button>
                <button 
                  onClick={confirmExportWord}
                  className="py-5 bg-orange-600 text-white rounded-3xl text-[11px] font-black uppercase tracking-widest hover:brightness-110 shadow-xl shadow-orange-600/20 transition-all"
                >
                  Confirm Export
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
      {showAddExerciseModal && (
        <div className="fixed inset-0 z-[300] bg-slate-950/80 backdrop-blur-2xl flex items-center justify-center p-4">
          <div className="bg-white rounded-[48px] w-full max-w-xl overflow-hidden shadow-2xl flex flex-col border border-white/50">
            <div className="p-8 flex justify-between items-center border-b border-slate-100">
              <div className="flex items-center gap-4">
                <div className="h-4 w-4 bg-blue-500 rounded-full animate-pulse"></div>
                <h2 className="text-[12px] font-black uppercase text-slate-900 tracking-widest">Add Custom Exercise Type</h2>
              </div>
              <button onClick={() => setShowAddExerciseModal(false)} className="h-10 w-10 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-900">
                <i className="fa-solid fa-xmark text-xl"></i>
              </button>
            </div>
            <div className="p-8 space-y-6">
              <div>
                <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2 block">Exercise Name</label>
                <input 
                  type="text"
                  value={newExerciseName}
                  onChange={(e) => setNewExerciseName(e.target.value)}
                  placeholder="e.g., Circle the best answer"
                  className="w-full bg-slate-50 border border-slate-200 rounded-2xl px-6 py-4 text-sm font-bold text-slate-700 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all"
                  autoFocus
                />
              </div>
              <div>
                <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2 block">Category</label>
                <select
                  value={newExerciseCategory}
                  onChange={(e) => setNewExerciseCategory(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-200 rounded-2xl px-6 py-4 text-sm font-bold text-slate-700 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all appearance-none"
                >
                  <option value="All">All Modules</option>
                  <option value="Grammar">Grammar</option>
                  <option value="Vocabulary">Vocabulary</option>
                  <option value="Reading">Reading</option>
                  <option value="Mixed">Mixed</option>
                  <option value="Generals">Generals</option>
                  <option value="Custom">Custom</option>
                </select>
              </div>
              <button 
                onClick={saveCustomExerciseType}
                disabled={!newExerciseName.trim()}
                className="w-full py-4 bg-blue-600 text-white rounded-2xl text-[11px] font-bold uppercase tracking-widest hover:bg-blue-700 transition-all disabled:opacity-50"
              >
                Save Exercise Type
              </button>
            </div>
          </div>
        </div>
      )}
      {showSubjectModal && (
        <div className="fixed inset-0 z-[300] bg-slate-950/80 backdrop-blur-2xl flex items-center justify-center p-4">
          <div className="bg-white rounded-[48px] w-full max-w-2xl overflow-hidden shadow-2xl flex flex-col border border-white/50">
            <div className="p-8 flex justify-between items-center border-b border-slate-100">
              <div className="flex items-center gap-4">
                <div className="h-4 w-4 bg-amber-500 rounded-full animate-pulse"></div>
                <h2 className="text-[12px] font-black uppercase text-slate-900 tracking-widest">Localization Subjects</h2>
              </div>
              <button onClick={() => setShowSubjectModal(false)} className="h-10 w-10 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-900">
                <i className="fa-solid fa-xmark text-xl"></i>
              </button>
            </div>
            <div className="p-8 space-y-6">
              <div className="flex items-center justify-between">
                <p className="text-xs text-slate-500 uppercase font-bold tracking-wider">Select a country to localize names and places in your tests:</p>
                <button 
                  onClick={() => setIsCountriesHidden(!isCountriesHidden)}
                  className="px-4 py-2 bg-slate-100 text-slate-600 rounded-xl text-[10px] font-black uppercase hover:bg-slate-200 transition-all"
                >
                  {isCountriesHidden ? 'Show All' : 'Hide List'}
                </button>
              </div>
              
              {!isCountriesHidden && (
                <div className="grid grid-cols-2 gap-4 max-h-[400px] overflow-y-auto pr-2 no-scrollbar">
                  {SUBJECTS.map(subject => (
                    <button
                      key={subject.id}
                      onClick={() => {
                        setActiveSubject(subject.id);
                        setIsRandomSubject(false);
                        setShowSubjectModal(false);
                      }}
                      className={`p-6 rounded-3xl border-2 transition-all flex flex-col gap-2 items-start ${activeSubject === subject.id && !isRandomSubject ? 'border-amber-500 bg-amber-50 shadow-lg' : 'border-slate-100 bg-slate-50 hover:border-amber-200'}`}
                    >
                      <span className={`text-sm font-black uppercase ${activeSubject === subject.id && !isRandomSubject ? 'text-amber-600' : 'text-slate-600'}`}>{subject.name}</span>
                      <span className="text-[10px] text-slate-400 line-clamp-1">{subject.names.slice(0, 3).join(', ')}...</span>
                    </button>
                  ))}
                  <button
                    onClick={() => {
                      setIsRandomSubject(true);
                      setShowSubjectModal(false);
                    }}
                    className={`p-6 rounded-3xl border-2 transition-all flex flex-col gap-2 items-start col-span-2 ${isRandomSubject ? 'border-amber-500 bg-amber-50 shadow-lg' : 'border-slate-100 bg-slate-50 hover:border-amber-200'}`}
                  >
                    <span className={`text-sm font-black uppercase ${isRandomSubject ? 'text-amber-600' : 'text-slate-600'}`}>🎲 Randomize Countries</span>
                    <span className="text-[10px] text-slate-400">AI will pick one of the 10 countries randomly for each generation.</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      {/* HIDDEN FILE INPUTS */}
      <input 
        type="file" 
        ref={fileInputRef} 
        className="hidden" 
        accept=".pdf,.doc,.docx,.txt,.png,.jpg,.jpeg" 
        onChange={handleFileUpload} 
      />
      <input 
        type="file" 
        ref={logoUploadRef} 
        className="hidden" 
        accept="image/*" 
        onChange={handleLogoUpload} 
      />
    </div>
    </ErrorBoundary>
  );
}

export default App;
