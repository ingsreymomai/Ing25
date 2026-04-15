import React, { useState, useRef, useEffect } from 'react';
import { RuleCategory } from '../types';

interface FormatDesignEditorProps {
  onSave: (design: { name: string; category: RuleCategory; style: any }) => void;
  currentStyle: any;
  initialCategory?: RuleCategory;
  designTargetTypeId?: string | null;
  initialName?: string;
}

const FormatDesignEditor: React.FC<FormatDesignEditorProps> = ({ onSave, currentStyle, initialCategory, designTargetTypeId, initialName }) => {
  console.log('FormatDesignEditor rendered with designTargetTypeId:', designTargetTypeId);
  const [activeRibbonTab, setActiveRibbonTab] = useState<'FILE' | 'HOME' | 'INSERT' | 'LAYOUT' | 'ADVANCED'>('HOME');
  const [designName, setDesignName] = useState(initialName || 'New Custom Design');
  const [selectedCategory, setSelectedCategory] = useState<RuleCategory>(initialCategory || 'General');
  
  // Font & Text Styles
  const [fontSize, setFontSize] = useState(currentStyle?.fontSize || '12');
  const [fontFamily, setFontFamily] = useState(currentStyle?.fontFamily || 'Times New Roman');
  const [textColor, setTextColor] = useState(currentStyle?.textColor || '#000000');
  const [fontWeight, setFontWeight] = useState(currentStyle?.fontWeight || '400');
  const [textDecoration, setTextDecoration] = useState(currentStyle?.textDecoration || 'none');
  
  // Layout & Container Styles
  const [columns, setColumns] = useState(currentStyle?.columns || 1);
  const [backgroundColor, setBackgroundColor] = useState(currentStyle?.backgroundColor || '#ffffff');
  const [containerPadding, setContainerPadding] = useState(currentStyle?.containerPadding || '40');
  const [containerMargin, setContainerMargin] = useState(currentStyle?.containerMargin || '0');
  const [containerBorderRadius, setContainerBorderRadius] = useState(currentStyle?.containerBorderRadius || '0');
  
  // Text Block Styles
  const [textBlockBg, setTextBlockBg] = useState(currentStyle?.textBlockBg || 'transparent');
  const [textBlockPadding, setTextBlockPadding] = useState(currentStyle?.textBlockPadding || '0');
  const [textBlockBorder, setTextBlockBorder] = useState(currentStyle?.textBlockBorder || 'none');
  
  // Table Styles
  const [showTable, setShowTable] = useState(currentStyle?.showTable !== undefined ? currentStyle.showTable : true);
  const [tablePadding, setTablePadding] = useState(currentStyle?.tablePadding || '8');
  const [tableBorderWidth, setTableBorderWidth] = useState(currentStyle?.tableBorderWidth || '1');
  const [tableBorderColor, setTableBorderColor] = useState(currentStyle?.tableBorderColor || '#000000');
  const [tableBorderStyle, setTableBorderStyle] = useState(currentStyle?.tableBorderStyle || 'solid');
  const [tableData, setTableData] = useState<string[][]>(currentStyle?.tableData || [
    ['Category', 'Value A', 'Value B'],
    ['Data 1', '100', '200'],
    ['Data 2', '150', '250']
  ]);
  const [cellAlignments, setCellAlignments] = useState<string[][]>(currentStyle?.cellAlignments || [
    ['left', 'left', 'left'],
    ['left', 'left', 'left'],
    ['left', 'left', 'left']
  ]);
  const [selectedCell, setSelectedCell] = useState<[number, number] | null>(null);

  // Advanced
  const [customCSS, setCustomCSS] = useState(currentStyle?.customCSS || '');
  const [showImages, setShowImages] = useState(currentStyle?.showImages || false);
  
  const getDefaultContent = () => {
    if (designTargetTypeId === 'mcq') {
      return 'Multiple Choice Questions\n\nQ1. This is a sample question designed with your custom format?\nA) Option One\nB) Option Two\nC) Option Three\nD) Option Four';
    } else if (designTargetTypeId === 'matching') {
      return 'Matching Exercise\n\nMatch the items on the left with the correct definitions on the right.\n1. Term A       A. Definition for Term A\n2. Term B       B. Definition for Term B';
    } else if (designTargetTypeId === 'true_false' || designTargetTypeId === 'tf') {
      return 'True / False Exercise\n\nRead the statements and mark them as True or False.\n1. The earth is round. ( T / F )\n2. Water boils at 50 degrees. ( T / F )';
    } else if (designTargetTypeId === 'correct_incorrect' || designTargetTypeId === 'correctIncorrect') {
      return 'Correct / Incorrect Exercise\n\nIdentify if the sentences are correct or incorrect.\n1. She don\'t like apples. ( C / I )\n2. He goes to school. ( C / I )';
    } else if (designTargetTypeId === 'table') {
      return 'Custom Table Design\n\nUse the ribbon above to add rows and columns. You can also style individual cells by clicking on them.';
    }
    return 'Custom Design Template\n\nThis is a sample text to preview your design. You can edit this text directly to see how your styles apply to different content types.';
  };

  const [editableContent, setEditableContent] = useState(currentStyle?.editableContent || {
    schoolName: 'Global Education Academy',
    footerText: 'Confidential • Academic Use Only',
    mainContent: getDefaultContent()
  });

  const contentRef = useRef<HTMLDivElement>(null);

  const emojis = [
    '📝', '✅', '⭐', '📚', '🍎', '🎨', '⚽', '🎸', '🚀', '🌈',
    '🐱', '🐶', '🦁', '🐘', '🦒', '🐧', '🦉', '🐝', '🦋', '🐢',
    '🍕', '🍔', '🍦', '🍩', '🍓', '🍉', '🍍', '🥑', '🥦', '🌽',
    '🏠', '🏫', '🏥', '🏦', '🗼', '🗽', '🎡', '🎢', '🚢', '✈️',
    '⌚', '📱', '💻', '⌨️', '🖱️', '📷', '🎥', '🔦', '🔋', '🔌',
    '💡', '📖', '📜', '🏷️', '💰', '✉️', '📦', '✏️', '✒️', '📏',
    '✂️', '📌', '📎', '🔑', '🔨', '🛠️', '⚙️', '⚖️', '🔗', '🧪',
    '🔭', '🧬', '🔬', '📡', '🛰️', '🌍', '🌙', '☀️', '☁️', '❄️',
    '🔥', '💧', '⚡', '☘️', '🍁', '🍄', '🐚', '💎', '🏆', '🥇',
    '🥈', '🥉', '🏅', '🎖️', '🎗️', '🎫', '🎟️', '🎭', '🎬',
    '🎤', '🎧', '🎼', '🎹', '🥁', '🎷', '🎺', '🎻', '🎲', '♟️',
    '🎯', '🎳', '🎮', '🎰', '🧩', '🧸', '🃏', '🀄', '🎴',
    '🖼️', '🧵', '🧶', '🛍️', '🛒', '🎁', '🎈', '🎏', '🎀',
    '🎊', '🎉', '🎎', '🏮', '🎐', '🧧', '📩', '📨', '📧',
    '🌎', '🌏', '🌐', '🗺️', '🗾', '🧭', '🏔️', '⛰️', '🌋',
    '🗻', '🏕️', '🏖️', '🏜️', '🏝️', '🏞️', '🏟️', '🏛️', '🏗️', '🧱'
  ];

  const insertEmoji = (emoji: string) => {
    // If a cell is selected, insert into that cell
    if (selectedCell) {
      const [r, c] = selectedCell;
      const newData = [...tableData];
      newData[r][c] += emoji;
      setTableData(newData);
    } else {
      // Otherwise insert into main content
      document.execCommand('insertText', false, emoji);
    }
  };

  const updateTableCell = (rowIndex: number, colIndex: number, value: string) => {
    const newData = [...tableData];
    newData[rowIndex][colIndex] = value;
    setTableData(newData);
  };

  const addTableRow = () => {
    setTableData(prev => [...prev, Array(prev[0]?.length || 1).fill('New Cell')]);
    setCellAlignments(prev => [...prev, Array(prev[0]?.length || 1).fill('left')]);
  };
  
  const addTableCol = () => {
    setTableData(prev => prev.map(row => [...row, 'New Cell']));
    setCellAlignments(prev => prev.map(row => [...row, 'left']));
  };
  
  const removeTableRow = () => {
    if (tableData.length > 1) {
      setTableData(prev => prev.slice(0, -1));
      setCellAlignments(prev => prev.slice(0, -1));
    }
  };
  
  const removeTableCol = () => {
    if (tableData[0]?.length > 1) {
      setTableData(prev => prev.map(row => row.slice(0, -1)));
      setCellAlignments(prev => prev.map(row => row.slice(0, -1)));
    }
  };

  const updateCellAlignment = (align: string) => {
    if (selectedCell) {
      const [r, c] = selectedCell;
      const newAligns = [...cellAlignments];
      newAligns[r][c] = align;
      setCellAlignments(newAligns);
    }
  };

  const [showRibbon, setShowRibbon] = useState(true);

  return (
    <div className="flex flex-col h-full bg-slate-50 rounded-[32px] overflow-hidden border border-slate-200 shadow-2xl relative">
      {/* MS Word Ribbon */}
      {showRibbon && (
        <div className="bg-white border-b border-slate-200 shrink-0">
          <div className="flex px-4 pt-1 gap-1 bg-slate-50">
            {['FILE', 'HOME', 'INSERT', 'LAYOUT', 'ADVANCED'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveRibbonTab(tab as any)}
                className={`px-4 py-1 text-[8px] font-black tracking-widest transition-all rounded-t-lg border-t border-x ${
                  activeRibbonTab === tab 
                    ? 'bg-white text-blue-600 border-slate-200' 
                    : 'bg-transparent text-slate-400 border-transparent hover:text-slate-600'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
          
          <div className="bg-white p-1 flex items-center gap-2 overflow-x-auto no-scrollbar border-t border-slate-200">
            {activeRibbonTab === 'HOME' && (
              <div className="flex items-center gap-3 border-r border-slate-200 pr-3">
                <div className="flex flex-col items-center gap-0.5">
                  <div className="flex bg-slate-50 border border-slate-200 rounded-lg p-0.5 gap-0.5">
                    <select 
                      value={fontFamily} 
                      onChange={(e) => setFontFamily(e.target.value)}
                      className="text-[9px] bg-white border border-slate-200 rounded px-1.5 py-0.5 outline-none"
                    >
                      <option>Times New Roman</option>
                      <option>Arial</option>
                      <option>Calibri</option>
                      <option>Inter</option>
                      <option>JetBrains Mono</option>
                    </select>
                    <select 
                      value={fontSize} 
                      onChange={(e) => setFontSize(e.target.value)}
                      className="text-[10px] bg-white border border-slate-200 rounded px-2 py-1 outline-none w-12"
                    >
                      {[8, 9, 10, 11, 12, 14, 16, 18, 20, 24].map(s => <option key={s}>{s}</option>)}
                    </select>
                  </div>
                  <span className="text-[8px] font-bold text-slate-400 uppercase">Font</span>
                </div>

                <div className="flex flex-col items-center gap-0.5">
                  <div className="flex bg-slate-50 border border-slate-200 rounded-lg p-0.5 gap-0.5">
                    <button onClick={() => setFontWeight(prev => prev === '900' ? '400' : '900')} className={`h-8 w-8 rounded flex items-center justify-center transition-all ${fontWeight === '900' ? 'bg-blue-600 text-white shadow-sm' : 'hover:bg-white text-slate-700'}`}><i className="fa-solid fa-bold"></i></button>
                    <button onClick={() => setTextDecoration(prev => prev === 'underline' ? 'none' : 'underline')} className={`h-8 w-8 rounded flex items-center justify-center transition-all ${textDecoration === 'underline' ? 'bg-blue-600 text-white shadow-sm' : 'hover:bg-white text-slate-700'}`}><i className="fa-solid fa-underline"></i></button>
                    <div className="flex items-center gap-1 px-2">
                      <input type="color" value={textColor} onChange={(e) => setTextColor(e.target.value)} className="w-6 h-6 p-0 border-none bg-transparent cursor-pointer" title="Text Color" />
                      <input type="color" value={backgroundColor} onChange={(e) => setBackgroundColor(e.target.value)} className="w-6 h-6 p-0 border-none bg-transparent cursor-pointer" title="Background Color" />
                    </div>
                  </div>
                  <span className="text-[8px] font-bold text-slate-400 uppercase">Style & Color</span>
                </div>
                
                <div className="flex flex-col items-center gap-0.5">
                  <div className="flex bg-slate-50 border border-slate-200 rounded-lg p-0.5 gap-0.5 max-w-[300px] overflow-x-auto no-scrollbar">
                    {emojis.map((e, idx) => (
                      <button key={`${e}-${idx}`} onClick={() => insertEmoji(e)} className="h-8 w-8 shrink-0 hover:bg-white hover:shadow-sm rounded flex items-center justify-center text-slate-700">{e}</button>
                    ))}
                  </div>
                  <span className="text-[8px] font-bold text-slate-400 uppercase">Emoji Picker</span>
                </div>
              </div>
            )}
            
            {activeRibbonTab === 'INSERT' && (
              <div className="flex items-center gap-4 border-r border-slate-200 pr-4">
                <div className="flex flex-col items-center gap-0.5 group cursor-pointer" onClick={() => setShowTable(!showTable)}>
                  <div className={`h-10 w-10 border rounded-xl flex items-center justify-center transition-all ${showTable ? 'bg-blue-600 border-blue-600 text-white' : 'bg-slate-50 border-slate-200 text-slate-600 group-hover:bg-blue-50'}`}>
                    <i className="fa-solid fa-table"></i>
                  </div>
                  <span className="text-[8px] font-black uppercase text-slate-400">Table</span>
                </div>
                {showTable && (
                  <>
                    <div className="flex flex-col items-center gap-0.5">
                      <div className="flex gap-1 bg-slate-50 p-0.5 rounded-lg border border-slate-200">
                        <button onClick={addTableRow} className="px-2 py-1 text-[8px] font-black uppercase bg-white rounded border border-slate-200 hover:bg-blue-50 text-blue-600">+Row</button>
                        <button onClick={addTableCol} className="px-2 py-1 text-[8px] font-black uppercase bg-white rounded border border-slate-200 hover:bg-blue-50 text-blue-600">+Col</button>
                        <button onClick={removeTableRow} className="px-2 py-1 text-[8px] font-black uppercase bg-white rounded border border-slate-200 hover:bg-red-50 text-red-600">-Row</button>
                        <button onClick={removeTableCol} className="px-2 py-1 text-[8px] font-black uppercase bg-white rounded border border-slate-200 hover:bg-red-50 text-red-600">-Col</button>
                      </div>
                      <span className="text-[8px] font-bold text-slate-400 uppercase">Structure</span>
                    </div>
                    <div className="flex flex-col items-center gap-0.5">
                      <div className="flex gap-1 bg-slate-50 p-0.5 rounded-lg border border-slate-200">
                        <button onClick={() => updateCellAlignment('left')} className="h-8 w-8 hover:bg-white rounded flex items-center justify-center text-slate-600"><i className="fa-solid fa-align-left"></i></button>
                        <button onClick={() => updateCellAlignment('center')} className="h-8 w-8 hover:bg-white rounded flex items-center justify-center text-slate-600"><i className="fa-solid fa-align-center"></i></button>
                        <button onClick={() => updateCellAlignment('right')} className="h-8 w-8 hover:bg-white rounded flex items-center justify-center text-slate-600"><i className="fa-solid fa-align-right"></i></button>
                      </div>
                      <span className="text-[8px] font-bold text-slate-400 uppercase">Cell Align</span>
                    </div>
                  </>
                )}
                <div className="flex flex-col items-center gap-0.5 group cursor-pointer" onClick={() => setShowImages(!showImages)}>
                  <div className={`h-10 w-10 border rounded-xl flex items-center justify-center transition-all ${showImages ? 'bg-blue-600 border-blue-600 text-white' : 'bg-slate-50 border-slate-200 text-slate-600 group-hover:bg-blue-50'}`}>
                    <i className="fa-solid fa-image"></i>
                  </div>
                  <span className="text-[8px] font-black uppercase text-slate-400">Pictures</span>
                </div>
              </div>
            )}
            
            {activeRibbonTab === 'LAYOUT' && (
              <div className="flex items-center gap-4 border-r border-slate-200 pr-4">
                <div className="flex flex-col items-center gap-0.5 group cursor-pointer" onClick={() => setColumns(prev => prev === 1 ? 2 : 1)}>
                  <div className={`h-10 w-10 border rounded-xl flex items-center justify-center transition-all ${columns === 2 ? 'bg-blue-600 border-blue-600 text-white' : 'bg-slate-50 border-slate-200 text-slate-600 group-hover:bg-blue-50'}`}>
                    <i className="fa-solid fa-columns"></i>
                  </div>
                  <span className="text-[8px] font-black uppercase text-slate-400">{columns} Column{columns > 1 ? 's' : ''}</span>
                </div>
                <div className="flex flex-col items-center gap-0.5">
                  <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-lg px-3 py-1">
                    <span className="text-[10px] font-bold text-slate-400">PAD</span>
                    <input type="range" min="0" max="100" value={containerPadding} onChange={(e) => setContainerPadding(e.target.value)} className="w-20 accent-blue-600" />
                    <span className="text-[10px] font-black text-slate-900 w-6">{containerPadding}</span>
                  </div>
                  <span className="text-[8px] font-bold text-slate-400 uppercase">Padding</span>
                </div>
                <div className="flex flex-col items-center gap-0.5">
                  <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-lg px-3 py-1">
                    <span className="text-[10px] font-bold text-slate-400">RAD</span>
                    <input type="range" min="0" max="50" value={containerBorderRadius} onChange={(e) => setContainerBorderRadius(e.target.value)} className="w-20 accent-blue-600" />
                    <span className="text-[10px] font-black text-slate-900 w-6">{containerBorderRadius}</span>
                  </div>
                  <span className="text-[8px] font-bold text-slate-400 uppercase">Radius</span>
                </div>
                <div className="flex flex-col items-center gap-0.5">
                  <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-lg px-3 py-1">
                    <input type="color" value={textBlockBg} onChange={(e) => setTextBlockBg(e.target.value)} className="w-6 h-6 p-0 border-none bg-transparent cursor-pointer" title="Text Block Background" />
                    <select value={textBlockPadding} onChange={(e) => setTextBlockPadding(e.target.value)} className="text-[10px] bg-white border border-slate-200 rounded px-1 outline-none">
                      {[0, 4, 8, 12, 16, 20, 30].map(p => <option key={p} value={p}>{p}px Pad</option>)}
                    </select>
                    <button onClick={() => setTextBlockBorder(prev => prev === 'none' ? 'solid' : 'none')} className={`h-6 w-6 rounded flex items-center justify-center border ${textBlockBorder !== 'none' ? 'bg-blue-600 border-blue-600 text-white' : 'bg-white border-slate-200 text-slate-400'}`}>
                      <i className="fa-solid fa-square-full text-[8px]"></i>
                    </button>
                  </div>
                  <span className="text-[8px] font-bold text-slate-400 uppercase">Text Block Style</span>
                </div>
                {showTable && (
                  <div className="flex flex-col items-center gap-1">
                    <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-lg px-3 py-1">
                      <input type="color" value={tableBorderColor} onChange={(e) => setTableBorderColor(e.target.value)} className="w-6 h-6 p-0 border-none bg-transparent cursor-pointer" />
                      <select value={tablePadding} onChange={(e) => setTablePadding(e.target.value)} className="text-[10px] bg-white border border-slate-200 rounded px-1 outline-none">
                        {[2, 4, 8, 12, 16].map(p => <option key={p} value={p}>{p}px</option>)}
                      </select>
                    </div>
                    <span className="text-[8px] font-bold text-slate-400 uppercase">Table Style</span>
                  </div>
                )}
              </div>
            )}

            {activeRibbonTab === 'ADVANCED' && (
              <div className="flex items-center gap-4 border-r border-slate-200 pr-4">
                <div className="flex flex-col items-center gap-1">
                  <textarea 
                    value={customCSS} 
                    onChange={(e) => setCustomCSS(e.target.value)}
                    className="text-[10px] w-64 h-10 p-2 rounded border border-slate-200 font-mono focus:ring-1 focus:ring-blue-500 outline-none"
                    placeholder=".preview-area { ... }"
                  />
                  <span className="text-[8px] font-bold text-slate-400 uppercase">Custom CSS Injector</span>
                </div>
              </div>
            )}

            <div className="flex-1"></div>
            
            <div className="flex items-center gap-3 bg-slate-50 p-2 rounded-2xl border border-slate-200">
              <div className="flex flex-col gap-1">
                <input 
                  type="text" 
                  value={designName} 
                  onChange={(e) => setDesignName(e.target.value)}
                  className="text-[10px] font-black uppercase text-slate-900 bg-transparent border-none focus:ring-0 p-0 w-40"
                  placeholder="Design Name"
                />
                <select 
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value as RuleCategory)}
                  className="text-[8px] font-bold text-slate-400 bg-transparent border-none focus:ring-0 p-0 uppercase cursor-pointer"
                >
                  {['General', 'Grammar', 'Vocabulary', 'Reading', 'Mixed', 'Custom', 'mcq', 'tf', 'correct_incorrect', 'vocabulary', 'circle', 'sentence_completion', 'word_box', 'reading_passage', 'matching', 'cloze', 'double_mcq'].map(cat => (
                    <option key={cat} value={cat}>{cat.replace('_', ' ').toUpperCase()} Category</option>
                  ))}
                </select>
              </div>
              <button 
                onClick={() => onSave({ 
                  name: designName, 
                  category: selectedCategory, 
                  style: { 
                    fontFamily, fontSize, textColor, fontWeight, textDecoration,
                    columns, backgroundColor, containerPadding, containerMargin, containerBorderRadius,
                    textBlockBg, textBlockPadding, textBlockBorder,
                    showTable, tablePadding, tableBorderWidth, tableBorderColor, tableBorderStyle,
                    tableData, cellAlignments, customCSS, showImages, editableContent
                  } 
                })}
                className="px-6 py-2 bg-blue-600 text-white rounded-xl text-[10px] font-black uppercase tracking-widest hover:bg-blue-700 shadow-lg shadow-blue-100 transition-all"
              >
                <i className="fa-solid fa-save mr-2"></i> Save Design
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Ribbon Toggle Button */}
      <button 
        onClick={() => setShowRibbon(!showRibbon)}
        className="absolute bottom-4 left-4 z-10 bg-white p-2 rounded-full shadow-lg border border-slate-200 text-slate-400 hover:text-blue-600"
      >
        <i className={`fa-solid ${showRibbon ? 'fa-compress' : 'fa-expand'}`}></i>
      </button>

      {/* Live Design Area */}
      <div className="flex-1 p-12 overflow-y-auto flex justify-center bg-slate-200/50 no-scrollbar">
        <style>{customCSS}</style>
        <div 
          className="preview-area bg-white w-full max-w-[800px] aspect-[1/1.414] shadow-2xl flex flex-col relative overflow-hidden transition-all duration-300"
          style={{ 
            fontFamily: fontFamily,
            fontSize: `${fontSize}px`,
            color: textColor,
            backgroundColor: backgroundColor,
            padding: `${containerPadding}px`,
            margin: `${containerMargin}px`,
            borderRadius: `${containerBorderRadius}px`,
            fontWeight: fontWeight,
            textDecoration: textDecoration
          }}
        >
          <div className="absolute top-0 right-0 p-4 pointer-events-none">
            <span className="text-[8px] font-black text-blue-600/20 uppercase tracking-[0.5em] rotate-45 origin-top-right">Live Design Engine v5.5</span>
          </div>
          
          {/* Content Simulation */}
          <div className={`flex-1 space-y-8 ${columns === 2 ? 'columns-2 gap-12' : ''}`}>
            <div
              ref={contentRef}
              contentEditable
              suppressContentEditableWarning
              onBlur={() => {
                if (contentRef.current) {
                  setEditableContent(prev => ({ ...prev, mainContent: contentRef.current!.innerHTML }));
                }
              }}
              className="w-full bg-transparent border-none focus:ring-0 p-0 outline-none leading-relaxed min-h-[100px] transition-all"
              style={{ 
                backgroundColor: textBlockBg,
                padding: `${textBlockPadding}px`,
                border: textBlockBorder === 'none' ? 'none' : `1px solid ${textColor}`,
                borderRadius: '4px'
              }}
              dangerouslySetInnerHTML={{ __html: editableContent.mainContent }}
            />

            {showTable && (
              <div className="w-full overflow-x-auto my-6 break-inside-avoid">
                <table 
                  className="w-full border-collapse" 
                  style={{ 
                    border: `${tableBorderWidth}pt ${tableBorderStyle} ${tableBorderColor}`,
                    padding: `${tablePadding}px`
                  }}
                >
                  <tbody>
                    {tableData.map((row, rIdx) => (
                      <tr key={rIdx}>
                        {row.map((cell, cIdx) => (
                          <td 
                            key={cIdx}
                            onClick={() => setSelectedCell([rIdx, cIdx])}
                            className={`p-3 border transition-all cursor-text ${selectedCell?.[0] === rIdx && selectedCell?.[1] === cIdx ? 'bg-blue-50 ring-2 ring-blue-500 ring-inset' : ''}`}
                            style={{ 
                              borderColor: tableBorderColor,
                              borderStyle: tableBorderStyle,
                              borderWidth: `${tableBorderWidth}pt`,
                              textAlign: cellAlignments[rIdx]?.[cIdx] as any || 'left'
                            }}
                          >
                            <input 
                              value={cell} 
                              onChange={(e) => updateTableCell(rIdx, cIdx, e.target.value)}
                              className="w-full bg-transparent border-none focus:ring-0 p-0 text-inherit font-inherit outline-none"
                            />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {showImages && (
              <div className="space-y-4 break-inside-avoid">
                <div className="grid grid-cols-2 gap-4">
                  <div className="aspect-video bg-slate-100 rounded-lg flex items-center justify-center border border-slate-200">
                    <i className="fa-solid fa-image text-slate-300 text-2xl"></i>
                  </div>
                  <div className="aspect-video bg-slate-100 rounded-lg flex items-center justify-center border border-slate-200">
                    <i className="fa-solid fa-image text-slate-300 text-2xl"></i>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          <div className="mt-auto pt-8 border-t border-slate-200 flex justify-between items-center text-[9px] font-bold text-slate-400 uppercase tracking-widest">
            <div>Page 1 of 2</div>
            <input 
              value={editableContent.footerText}
              onChange={(e) => setEditableContent(prev => ({ ...prev, footerText: e.target.value }))}
              className="text-right bg-transparent border-none focus:ring-0 p-0 w-64"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default FormatDesignEditor;
