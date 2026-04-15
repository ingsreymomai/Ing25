import React, { useState, useEffect } from 'react';
import { AcademicLevel, HistoryItem, BrandSettings } from '../types';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  curriculum: string[];
  activeModule: string;
  onModuleChange: (m: string) => void;
  activeLevel: AcademicLevel;
  onLevelChange: (l: AcademicLevel) => void;
  topic: string;
  onTopicChange: (t: string) => void;
  onClearCanvas: () => void;
  onToggleSettings: (tab?: string) => void;
  history: HistoryItem[];
  onLoadHistory: (item: HistoryItem) => void;
  onDeleteHistory: (id: string) => void;
  onRenameHistory?: (id: string, newTitle: string) => void;
  brandSettings: BrandSettings;
  templates: any[];
  activeTemplate: any;
  onTemplateSelect: (t: any) => void;
  isSingleReadingText: boolean;
  onSingleReadingTextChange: (val: boolean) => void;
  isRelaxingBackgroundEnabled: boolean;
  onRelaxingBackgroundChange: (val: boolean) => void;
  isPartBackgroundEnabled: boolean;
  onPartBackgroundChange: (val: boolean) => void;
  isInstructionBackgroundEnabled: boolean;
  onInstructionBackgroundChange: (val: boolean) => void;
  onRandomizeBackground: () => void;
  paperDesign: number;
  onPaperDesignChange: (val: number) => void;
  onDesignPaperClick: () => void;
  onPaperStyleClick: () => void;
  onHeaderFooterDesignClick: () => void;
  onInstructionDesignClick: () => void;
  onSubjectsClick: () => void;
  onFormatDesignClick: () => void;
  mcqLayout: 'single' | 'double' | 'quad';
  onMCQLayoutChange: (val: 'single' | 'double' | 'quad') => void;
  instructionCase: 'uppercase' | 'lowercase' | 'random';
  onInstructionCaseChange: (val: 'uppercase' | 'lowercase' | 'random') => void;
  width?: number;
  onWidthChange?: (width: number) => void;
  side?: 'left' | 'right';
  onSideChange?: (side: 'left' | 'right') => void;
  user: any;
  onLogin: () => void;
  onLogout: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  isOpen, onClose,
  activeModule, onModuleChange,
  activeLevel, onLevelChange,
  topic, onTopicChange,
  onClearCanvas, onToggleSettings,
  history, onLoadHistory, onDeleteHistory, onRenameHistory,
  brandSettings,
  templates, activeTemplate, onTemplateSelect,
  isSingleReadingText, onSingleReadingTextChange,
  isRelaxingBackgroundEnabled, onRelaxingBackgroundChange,
  isPartBackgroundEnabled, onPartBackgroundChange,
  isInstructionBackgroundEnabled, onInstructionBackgroundChange,
  onRandomizeBackground,
  paperDesign, onPaperDesignChange,
  onDesignPaperClick,
  onPaperStyleClick,
  onHeaderFooterDesignClick,
  onInstructionDesignClick,
  onSubjectsClick,
  onFormatDesignClick,
  mcqLayout,
  onMCQLayoutChange,
  instructionCase, onInstructionCaseChange,
  width = 280,
  onWidthChange,
  side = 'left',
  onSideChange,
  user,
  onLogin,
  onLogout
}) => {
  const [editingHistId, setEditingHistId] = useState<string | null>(null);
  const [tempTitle, setTempTitle] = useState('');
  const [isResizing, setIsResizing] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing || !onWidthChange) return;
      const newWidth = side === 'left' ? e.clientX : window.innerWidth - e.clientX;
      if (newWidth > 200 && newWidth < 600) {
        onWidthChange(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, onWidthChange, side]);

  const startRename = (e: React.MouseEvent, item: HistoryItem) => {
    e.stopPropagation();
    setEditingHistId(item.id);
    setTempTitle(item.title);
  };

  const submitRename = (id: string) => {
    if (onRenameHistory && tempTitle.trim()) {
      onRenameHistory(id, tempTitle);
    }
    setEditingHistId(null);
  };

  return (
    <aside 
      style={{ width: isOpen ? `${width}px` : '0px' }}
      className={`glass-panel h-full fixed ${side === 'left' ? 'left-0 border-r' : 'right-0 border-l'} top-0 text-slate-900 flex flex-col z-[110] border-white/20 transition-all duration-300 ease-in-out ${isOpen ? 'translate-x-0' : side === 'left' ? '-translate-x-full' : 'translate-x-full'}`}
    >
      {/* Resize Handle */}
      {isOpen && onWidthChange && (
        <div 
          onMouseDown={() => setIsResizing(true)}
          className={`absolute top-0 ${side === 'left' ? '-right-1' : '-left-1'} w-2 h-full cursor-col-resize hover:bg-orange-500/20 transition-colors z-[120]`}
        />
      )}

      <div className={`w-full h-full flex flex-col overflow-hidden ${isOpen ? 'opacity-100' : 'opacity-0'} transition-opacity duration-300`} style={{ width: `${width}px` }}>
        <div className="p-6 flex flex-col gap-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 bg-gradient-to-br from-orange-500 to-orange-600 rounded-xl flex items-center justify-center shadow-lg shadow-orange-200">
                <i className="fa-solid fa-sparkles text-white text-lg"></i>
              </div>
              <div>
                <h1 className="text-sm font-bold text-slate-800 leading-none">DPSS Virtues</h1>
                <span className="text-[9px] font-bold text-slate-400 uppercase tracking-wider">Version 2.0</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button 
                onClick={() => onSideChange?.(side === 'left' ? 'right' : 'left')}
                className="h-8 w-8 bg-slate-50 text-slate-400 rounded-lg flex items-center justify-center hover:bg-slate-100 hover:text-orange-600 transition-all"
                title="Switch Side"
              >
                <i className={`fa-solid ${side === 'left' ? 'fa-right-left' : 'fa-left-right'}`}></i>
              </button>
              <button 
                onClick={onClose}
                className="h-8 w-8 bg-slate-50 text-slate-400 rounded-lg flex items-center justify-center hover:bg-slate-100 hover:text-orange-600 transition-all"
                title="Hide Sidebar"
              >
                <i className="fa-solid fa-chevron-left"></i>
              </button>
            </div>
          </div>

          {/* User Profile / Login */}
          <div className="bg-slate-50 p-3 rounded-2xl border border-slate-100">
            {user ? (
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 overflow-hidden">
                  {user.photoURL ? (
                    <img src={user.photoURL} alt={user.displayName} className="h-8 w-8 rounded-full border border-white shadow-sm" referrerPolicy="no-referrer" />
                  ) : (
                    <div className="h-8 w-8 bg-orange-100 text-orange-600 rounded-full flex items-center justify-center font-bold text-xs uppercase">
                      {user.displayName?.charAt(0) || user.email?.charAt(0)}
                    </div>
                  )}
                  <div className="flex flex-col overflow-hidden">
                    <span className="text-[10px] font-bold text-slate-700 truncate">{user.displayName}</span>
                    <span className="text-[8px] text-slate-400 truncate">{user.email}</span>
                  </div>
                </div>
                <button 
                  onClick={onLogout}
                  className="h-7 w-7 bg-white text-slate-400 rounded-lg flex items-center justify-center hover:text-rose-600 transition-all shadow-sm"
                  title="Logout"
                >
                  <i className="fa-solid fa-right-from-bracket text-[10px]"></i>
                </button>
              </div>
            ) : (
              <button 
                onClick={onLogin}
                className="w-full py-2 bg-orange-600 text-white rounded-xl text-[10px] font-bold uppercase tracking-widest flex items-center justify-center gap-2 hover:bg-orange-700 transition-all shadow-md shadow-orange-200"
              >
                <i className="fa-solid fa-user"></i> Login with Google
              </button>
            )}
          </div>

        <div className="flex bg-slate-100 p-1 rounded-xl gap-1">
          {['GRAMMAR', 'READING', 'VOCABULARY', 'MIXED'].map(mod => (
            <button
              key={mod}
              onClick={() => onModuleChange(mod.charAt(0) + mod.slice(1).toLowerCase())}
              className={`flex-1 py-2 rounded-lg text-[8px] font-bold transition-all ${
                activeModule.toUpperCase() === mod 
                  ? 'bg-white text-orange-600 shadow-sm' 
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              {mod}
            </button>
          ))}
        </div>

        {activeModule.toUpperCase() === 'READING' && (
          <div className="flex items-center justify-between bg-blue-50 p-3 rounded-xl border border-blue-100">
            <span className="text-[8px] font-black text-blue-600 uppercase tracking-widest">One Reading Text for All Parts</span>
            <button 
              onClick={() => onSingleReadingTextChange(!isSingleReadingText)}
              className={`w-8 h-4 rounded-full transition-all relative ${isSingleReadingText ? 'bg-blue-600' : 'bg-slate-300'}`}
            >
              <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${isSingleReadingText ? 'left-[18px]' : 'left-[2px]'}`}></div>
            </button>
          </div>
        )}

        <div className="flex flex-col gap-2 bg-purple-50 p-3 rounded-xl border border-purple-100">
          <button 
            onClick={onDesignPaperClick}
            className="w-full flex items-center justify-between bg-white p-2 rounded-lg border border-purple-200 hover:bg-purple-100 hover:border-purple-300 transition-colors group shadow-sm"
          >
            <span className="text-[8px] font-black text-purple-600 uppercase tracking-widest">Design Test Style</span>
            <i className="fa-solid fa-chevron-right text-purple-400 group-hover:text-purple-600 transition-colors text-[7px]"></i>
          </button>
          <button 
            onClick={onFormatDesignClick}
            className="w-full flex items-center justify-between bg-white p-2 rounded-lg border border-orange-200 hover:bg-orange-100 hover:border-orange-300 transition-colors group shadow-sm"
          >
            <span className="text-[8px] font-black text-orange-600 uppercase tracking-widest">Format Design</span>
            <i className="fa-solid fa-chevron-right text-orange-400 group-hover:text-orange-600 transition-colors text-[7px]"></i>
          </button>

          <button 
            onClick={onSubjectsClick}
            className="w-full flex items-center justify-between bg-white p-2 rounded-lg border border-amber-200 hover:bg-amber-100 hover:border-amber-300 transition-colors group shadow-sm"
          >
            <span className="text-[8px] font-black text-amber-600 uppercase tracking-widest">Subjects</span>
            <i className="fa-solid fa-chevron-right text-amber-400 group-hover:text-amber-600 transition-colors text-[7px]"></i>
          </button>

          <div className="w-full flex items-center justify-between bg-white p-2 rounded-lg border border-emerald-200 shadow-sm">
            <span className="text-[8px] font-black text-emerald-600 uppercase tracking-widest">MCQ Lines</span>
            <div className="flex gap-1">
              {[
                { id: 'single', label: 'L1' },
                { id: 'double', label: 'L2' },
                { id: 'quad', label: 'L4' }
              ].map((l) => (
                <button
                  key={l.id}
                  onClick={() => onMCQLayoutChange(l.id as any)}
                  className={`h-5 px-2 rounded-md text-[7px] font-black transition-all ${mcqLayout === l.id ? 'bg-emerald-600 text-white' : 'bg-slate-50 border border-emerald-100 text-emerald-400 hover:bg-emerald-50'}`}
                >
                  {l.label}
                </button>
              ))}
            </div>
          </div>

          <button 
            onClick={onPaperStyleClick}
            className="w-full flex items-center justify-between bg-white p-2 rounded-lg border border-teal-200 hover:bg-teal-100 hover:border-teal-300 transition-colors group shadow-sm"
          >
            <span className="text-[8px] font-black text-teal-600 uppercase tracking-widest">Paper Style</span>
            <i className="fa-solid fa-chevron-right text-teal-400 group-hover:text-teal-600 transition-colors text-[7px]"></i>
          </button>

          <button 
            onClick={onHeaderFooterDesignClick}
            className="w-full flex items-center justify-between bg-white p-2 rounded-lg border border-blue-200 hover:bg-blue-100 hover:border-blue-300 transition-colors group shadow-sm"
          >
            <span className="text-[8px] font-black text-blue-600 uppercase tracking-widest">Header & Footer Styles</span>
            <i className="fa-solid fa-chevron-right text-blue-400 group-hover:text-blue-600 transition-colors text-[7px]"></i>
          </button>


        </div>

        <div className="flex flex-col gap-2 bg-indigo-50 p-3 rounded-xl border border-indigo-100">
          <div className="flex items-center justify-between">
            <span className="text-[8px] font-black text-indigo-600 uppercase tracking-widest">Instruction</span>
            <div className="flex gap-1">
              {[
                { id: 'uppercase', label: 'ABC' },
                { id: 'lowercase', label: 'Abc' },
                { id: 'random', label: 'Rand' }
              ].map((c) => (
                <button
                  key={c.id}
                  onClick={() => onInstructionCaseChange(c.id as any)}
                  className={`h-5 px-2 rounded-full text-[7px] font-black transition-all ${instructionCase === c.id ? 'bg-indigo-600 text-white' : 'bg-white border border-indigo-200 text-indigo-400'}`}
                >
                  {c.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* History Section */}
        <div className="flex-1 overflow-hidden flex flex-col">
          <label className="text-[9px] font-black text-slate-400 uppercase tracking-[0.15em] mb-4 block px-1">Neural History</label>
          <div className="flex-1 overflow-y-auto no-scrollbar space-y-3 pr-1">
            {(history || []).length === 0 && (
              <div className="p-8 border-2 border-dashed border-slate-100 rounded-3xl text-center space-y-2">
                <i className="fa-solid fa-clock-rotate-left text-slate-200 text-xl"></i>
                <p className="text-[8px] font-bold text-slate-300 uppercase">No History Yet</p>
              </div>
            )}
            {(history || []).map((item) => (
              <div 
                key={item.id} 
                onClick={() => onLoadHistory(item)}
                className="group bg-white border border-slate-100 rounded-2xl p-4 hover:border-orange-200 hover:shadow-md transition-all cursor-pointer relative overflow-hidden"
              >
                <div className="flex flex-col gap-1">
                  {editingHistId === item.id ? (
                    <input 
                      autoFocus
                      value={tempTitle}
                      onChange={e => setTempTitle(e.target.value)}
                      onBlur={() => submitRename(item.id)}
                      onKeyDown={e => e.key === 'Enter' && submitRename(item.id)}
                      className="text-[9px] font-black text-slate-900 uppercase bg-slate-50 border-none outline-none w-full"
                    />
                  ) : (
                    <span className="text-[9px] font-black text-slate-900 uppercase truncate pr-6">{item.title}</span>
                  )}
                  <span className="text-[7px] font-bold text-slate-400 uppercase">{new Date(item.timestamp).toLocaleDateString()}</span>
                </div>
                <div className="absolute right-2 top-1/2 -translate-y-1/2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button onClick={(e) => startRename(e, item)} className="h-6 w-6 bg-slate-50 text-slate-400 rounded-lg hover:text-orange-600"><i className="fa-solid fa-pen text-[8px]"></i></button>
                  <button onClick={(e) => { e.stopPropagation(); onDeleteHistory(item.id); }} className="h-6 w-6 bg-slate-50 text-slate-400 rounded-lg hover:text-rose-600"><i className="fa-solid fa-trash text-[8px]"></i></button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-auto p-6 border-t border-slate-100 space-y-4">
        <button 
          onClick={() => onToggleSettings()}
          className="w-full py-3 bg-slate-50 hover:bg-slate-100 text-slate-600 rounded-xl text-[9px] font-bold uppercase tracking-widest flex items-center justify-center gap-2 transition-all"
        >
          <i className="fa-solid fa-gear"></i> Workspace Settings
        </button>
        <div className="flex items-center gap-2 text-slate-400">
          <div className="h-5 w-5 rounded-full border border-emerald-500 flex items-center justify-center">
            <i className="fa-solid fa-check text-[7px] text-emerald-500"></i>
          </div>
          <span className="text-[9px] font-bold uppercase tracking-wider">AI Engine Ready</span>
        </div>
      </div>
    </div>
  </aside>
  );
};

export default Sidebar;