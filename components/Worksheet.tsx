import React, { useMemo, useRef, useEffect } from 'react';
import { AcademicLevel, Theme, PaperType, BrandSettings } from '../types';
import { FONTS, PAPER_DESIGNS } from '../constants';
import { clsx } from 'clsx';

interface WorksheetProps {
  content: string;
  onContentChange: (val: string) => void;
  isGenerating: boolean;
  level: AcademicLevel;
  module: string;
  topic: string;
  paperType: PaperType;
  theme: Theme;
  brandSettings: BrandSettings;
  paperDesign: number;
  mcqStyle: number;
  isColorfulBackgroundEnabled: boolean;
  isInstructionBackgroundEnabled: boolean;
  globalLayout: number;
  baseLayout: number;
  instructionRulerStyle?: number;
  instructionHeaderStyle?: number;
  zoom?: number;
}

const Worksheet: React.FC<WorksheetProps> = ({
  content, onContentChange, isGenerating, brandSettings, paperDesign, mcqStyle, isColorfulBackgroundEnabled, isInstructionBackgroundEnabled, globalLayout, baseLayout, instructionRulerStyle = 0, instructionHeaderStyle = 0, zoom = 100
}) => {
  const editorRef = useRef<HTMLDivElement>(null);

  const placeholderHtml = useMemo(() => {
    return `
      <div class="flex flex-col items-center h-full min-h-[600px] text-center px-4 py-20 relative overflow-hidden bg-white text-black">
        ${brandSettings.logoData ? `<img src="${brandSettings.logoData}" class="max-h-32 w-auto mb-10" />` : ''}
        <div class="border-[12px] border-black p-12 max-w-2xl w-full">
            <div class="space-y-6 text-center">
                <h1 class="text-2xl md:text-3xl font-black uppercase tracking-widest">${brandSettings.schoolName}</h1>
                <hr class="border-black border-2 w-full" />
                <p class="text-sm md:text-lg font-bold italic tracking-[0.3em] uppercase">${brandSettings.schoolAddress}</p>
            </div>
        </div>
        ${instructionRulerStyle > 0 ? `
          <div class="w-full max-w-2xl mt-4">
            <div class="instruction-ruler-${instructionRulerStyle}"></div>
          </div>
        ` : ''}
        <div class="mt-20 w-full max-w-2xl text-left font-bold text-[10px] space-y-4">
            <div class="flex justify-between border-b border-slate-200 pb-2 text-slate-400">
                <span>NAME : _________________________________</span>
                <span>DATE : ____ / ____ / ____</span>
            </div>
        </div>
      </div>
    `;
  }, [brandSettings.schoolName, brandSettings.schoolAddress]);

  useEffect(() => {
    if (editorRef.current && content !== editorRef.current.innerHTML) {
      if (!isGenerating) {
        // Handle instruction rulers with decorative elements
        const rulers = editorRef.current.querySelectorAll('[class^="instruction-ruler-"]');
        rulers.forEach(ruler => {
          const styleNum = ruler.className.match(/instruction-ruler-(\d+)/)?.[1];
          if (styleNum === '5') {
            ruler.innerHTML = '<span>★</span><span>★</span><span>★</span>';
          } else if (styleNum === '6') {
            ruler.innerHTML = '<span>♥</span><span>♥</span><span>♥</span>';
          }
        });

        editorRef.current.innerHTML = content || placeholderHtml;
        
        // Clean up MCQ brackets in preview if MCQ Style is active
        if (mcqStyle > 0) {
          const boldElements = editorRef.current.querySelectorAll('b, strong');
          boldElements.forEach(el => {
            let text = el.textContent?.trim().toUpperCase() || '';
            if (/^[\(\[ ]*[A-D][\)\]\. ]*$/.test(text)) {
              const letter = text.match(/[A-D]/)?.[0];
              if (letter) {
                el.textContent = letter;
              }
            }
          });
        }
      }
    }
  }, [content, isGenerating, placeholderHtml, mcqStyle]);

  const activeFontFamily = useMemo(() => {
    const font = FONTS.find(f => f.name === brandSettings.activeFont);
    return font ? font.family : "'Times New Roman', serif";
  }, [brandSettings.activeFont]);

  const decorativeElements = useMemo(() => {
    if (globalLayout < 10 || globalLayout > 14) return null;
    
    const count = Math.floor(Math.random() * 6) + 5; // 5 to 10
    const elements = [];
    const types = {
      10: '★', // Stars
      11: '🌸', // Flowers
      12: '❤', // Hearts
      13: '🫧', // Bubbles
      14: '🍃'  // Leaves
    };
    const colors = {
      10: '#fcd34d',
      11: '#f9a8d4',
      12: '#fca5a5',
      13: '#bae6fd',
      14: '#86efac'
    };

    for (let i = 0; i < count; i++) {
      const top = Math.random() * 90 + 5;
      const left = Math.random() * 90 + 5;
      const size = Math.random() * 30 + 15;
      const rotation = Math.random() * 360;
      const opacity = Math.random() * 0.4 + 0.2;

      elements.push(
        <div
          key={i}
          className="absolute pointer-events-none select-none"
          style={{
            top: `${top}%`,
            left: `${left}%`,
            fontSize: `${size}pt`,
            color: colors[globalLayout as keyof typeof colors],
            opacity: opacity,
            transform: `rotate(${rotation}deg)`,
            zIndex: 5
          }}
        >
          {types[globalLayout as keyof typeof types]}
        </div>
      );
    }
    return elements;
  }, [globalLayout]);

  return (
    <div className="flex-1 overflow-auto p-2 md:p-12 no-scrollbar bg-slate-200/50">
      <div id="worksheet-container" className="w-full max-w-[210mm] mx-auto pb-64 shadow-2xl worksheet-paper transition-transform duration-300 ease-in-out" style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'top center' }}>
        <style>{`
          * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
          .prose { 
            font-family: ${activeFontFamily} !important; 
            font-size: ${brandSettings.fontSize || 12}pt !important; 
            line-height: 2 !important; /* Increased for better line alignment */
            padding: 0.33in 0.5in !important; /* Adjusted to align with 24pt lines */
            position: relative;
            z-index: 10;
            background: transparent !important;
          }
          @media (min-width: 768px) {
            .prose { padding: 0.66in 1in !important; }
          }
          .prose p, .prose div { margin-bottom: 0 !important; background: transparent !important; }
          .prose div[style*="background"] { padding: 15pt !important; border-radius: 8pt; margin-bottom: 20pt !important; background-color: rgba(255,255,255,0.7) !important; backdrop-filter: blur(2px); }
          .prose li, .prose ol, .prose ul { margin: 0 !important; padding: 0 !important; background: transparent !important; }
          .prose table { border-collapse: collapse !important; width: 100% !important; border: 1.5pt solid black !important; table-layout: fixed; margin-bottom: 0 !important; background: transparent !important; }
          .prose table table { border: none !important; margin-top: 5pt !important; width: auto !important; background: transparent !important; }
          .prose table table td { border: none !important; padding: 2pt 10pt !important; background: transparent !important; }
          .prose table table tr td:first-child { padding-left: 30pt !important; }
          .prose .options-table { width: 100% !important; table-layout: fixed !important; background: transparent !important; }
          .prose .options-table td { padding-left: 30pt !important; background: transparent !important; }
          .prose .options-table tr td:first-child { padding-left: 30pt !important; }
          .prose .options-table tr td:not(:first-child) { padding-left: 10pt !important; }
          .prose .blank-line { border-bottom: 1pt solid black; display: inline-block; min-width: 50pt; margin: 0 5pt; }
          .prose .checkbox-box { border: 1pt solid black; width: 12pt; height: 12pt; display: inline-block; margin-right: 5pt; vertical-align: middle; }
          .prose th, .prose td { border: 1pt solid black !important; padding: 6pt !important; vertical-align: top !important; background: transparent !important; }
          .prose .header-row, .prose tr:first-child td[colspan] {
            background-color: #dcfce7 !important;
            color: #064e3b !important;
            border-left: 6pt solid #059669 !important;
            text-align: left !important;
            padding-left: 15pt !important;
            font-weight: bold !important;
          }
          
          /* Instruction Header Styles */
          ${instructionHeaderStyle === 0 ? `
            .prose .header-row:not([style*="background"]), .prose tr:first-child td[colspan]:not([style*="background"]) { background-color: #334155 !important; color: white !important; }
          ` : ''}
          ${instructionHeaderStyle === 1 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { color: #1e3a8a !important; text-align: left !important; padding-left: 15pt !important; }
          ` : ''}
          ${instructionHeaderStyle === 2 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { color: #064e3b !important; text-align: left !important; padding-left: 15pt !important; }
          ` : ''}
          ${instructionHeaderStyle === 3 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { color: #7f1d1d !important; text-align: left !important; padding-left: 15pt !important; }
          ` : ''}
          ${instructionHeaderStyle === 4 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { border: 2pt solid #334155 !important; color: #334155 !important; background-color: transparent !important; }
          ` : ''}
          ${instructionHeaderStyle === 5 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { border: none !important; border-bottom: 3pt solid #334155 !important; color: #334155 !important; background-color: transparent !important; font-size: 14pt !important; font-weight: 900 !important; text-align: left !important; padding: 10pt 0 !important; }
          ` : ''}
          ${instructionHeaderStyle === 6 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { border: none !important; border-bottom: 4pt double #334155 !important; color: #334155 !important; background-color: transparent !important; text-align: left !important; padding: 8pt 0 !important; }
          ` : ''}
          ${instructionHeaderStyle === 7 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { color: #1e293b !important; border-radius: 8pt !important; border: 1pt solid #e2e8f0 !important; }
          ` : ''}
          ${instructionHeaderStyle === 8 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { color: #3730a3 !important; border-right: 6pt solid #3730a3 !important; text-align: right !important; padding-right: 15pt !important; }
          ` : ''}
          ${instructionHeaderStyle === 9 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { color: #92400e !important; border: 1.5pt dashed #92400e !important; }
          ` : ''}
          ${instructionHeaderStyle === 10 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { border: none !important; border-bottom: 1pt solid #e2e8f0 !important; color: #334155 !important; background-color: transparent !important; text-align: left !important; padding: 5pt 0 !important; }
          ` : ''}
          ${instructionHeaderStyle === 11 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { color: white !important; border-radius: 4pt !important; text-align: center !important; padding: 12pt !important; }
          ` : ''}
          ${instructionHeaderStyle === 12 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { border: 2pt solid #10b981 !important; color: #065f46 !important; text-align: center !important; font-weight: 900 !important; }
          ` : ''}
          ${instructionHeaderStyle === 13 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { border: 3pt solid black !important; color: black !important; text-transform: uppercase !important; font-weight: 900 !important; }
          ` : ''}
          ${instructionHeaderStyle === 15 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { background-color: #581c87 !important; color: white !important; border: 2pt solid #fbbf24 !important; text-align: center !important; }
          ` : ''}
          ${instructionHeaderStyle === 16 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { background-color: #14532d !important; color: white !important; border-radius: 20pt 0 20pt 0 !important; padding-left: 20pt !important; }
          ` : ''}
          ${instructionHeaderStyle === 17 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { background: linear-gradient(90deg, #0ea5e9, #38bdf8) !important; color: white !important; text-align: center !important; }
          ` : ''}
          ${instructionHeaderStyle === 18 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { border: 2pt dotted #64748b !important; color: #475569 !important; background: transparent !important; }
          ` : ''}
          ${instructionHeaderStyle === 19 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { background-color: #ea580c !important; color: white !important; border-bottom: 4pt solid #9a3412 !important; font-weight: 900 !important; }
          ` : ''}
          ${instructionHeaderStyle === 20 ? `
            .prose .header-row, .prose tr:first-child td[colspan] { background: linear-gradient(90deg, #f8fafc 50%, #f1f5f9 50%) !important; background-size: 40px 100% !important; border: 1pt solid #cbd5e1 !important; color: #1e293b !important; text-align: center !important; }
          ` : ''}
          ${instructionHeaderStyle === 14 ? `
            /* Mix Styles - No global override, let AI generate specific styles */
          ` : ''}

          /* Zebra Striping - Disabled for decorative papers */
          ${globalLayout === 0 ? `
            .prose tr:nth-child(even) td { background-color: #f8fafc !important; }
            .prose tr:nth-child(odd) td { background-color: #ffffff !important; }
          ` : ''}
          
          .prose .options-table tr td { background-color: transparent !important; }
          
          .prose .professional-table {
            border-collapse: collapse !important;
            width: 100% !important;
            margin-bottom: 20pt !important;
            background: transparent !important;
          }
          .prose .professional-table thead th {
            background-color: #334155 !important;
            color: white !important;
            text-align: center !important;
            font-weight: bold !important;
            padding: 10pt !important;
          }
          .prose .professional-table tbody td {
            padding: 8pt !important;
            border: 1pt solid #e2e8f0 !important;
            background: transparent !important;
          }

          /* Teacher Answer Key Styling */
          .prose .answer-key-section {
            margin-top: 40pt !important;
            padding: 20pt !important;
            border-top: 2pt dashed #cbd5e1 !important;
            background-color: rgba(248, 250, 252, 0.8) !important;
            border-radius: 8pt !important;
          }
          .prose .answer-key-section h2 {
            color: #1e293b !important;
            font-size: 14pt !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            margin-bottom: 10pt !important;
          }
          .prose .answer-key-section b, .prose .answer-key-section strong {
            border: none !important;
            background: transparent !important;
            min-width: auto !important;
            height: auto !important;
            display: inline !important;
            margin-right: 0.2em !important;
            font-size: inherit !important;
          }

          /* Ruler Table Styling */
          .prose .ruler-table { border: none !important; border-collapse: collapse !important; }
          .prose .ruler-table td { border: none !important; padding: 15pt !important; }
          .prose .ruler-table tr td:first-child { border-right: 1.5pt solid #ff0000 !important; padding-right: 15pt !important; }
          .prose .ruler-table tr td:last-child { padding-left: 15pt !important; }
          .prose .ruler-table tr:first-child td[colspan] { border-right: none !important; border-bottom: 2pt solid #334155 !important; }

          /* Native Column Layout */
          .prose .column-layout {
            column-count: 2;
            column-gap: 40pt;
            column-rule: 1.5pt solid black;
            margin-top: 10pt;
            padding: 10pt 0;
          }
          .prose .column-layout p, .prose .column-layout div {
            break-inside: avoid-column;
          }

          /* Paper Designs */
          .design-modern-blue { border: 4pt solid #2563eb !important; border-radius: 0 !important; box-shadow: 15pt 15pt 0 #dbeafe !important; }
          .design-modern-blue .header-row { background-color: #2563eb !important; color: white !important; text-transform: uppercase; }
          
          .design-classic { border: 1.5pt solid black !important; padding: 1.2in !important; outline: 4pt double black !important; outline-offset: -15pt !important; }
          .design-classic .header-row { background-color: black !important; color: white !important; font-family: 'Georgia', serif !important; }
          
          .design-minimalist { border: none !important; box-shadow: none !important; padding: 1.5in !important; }
          .design-minimalist .header-row { background: transparent !important; color: black !important; border-bottom: 3pt solid black !important; text-transform: uppercase; letter-spacing: 0.4em; font-weight: 900 !important; }
          
          .design-playful { border: 4pt dashed #f97316 !important; border-radius: 40pt !important; background-color: #fff7ed !important; }
          .design-playful .header-row { background-color: #f97316 !important; border-radius: 20pt 20pt 0 0 !important; font-size: 1.2em !important; }
          .design-playful td { border: 2pt dashed #fdba74 !important; border-radius: 15pt !important; }
          
          .design-professional { border-left: 25pt solid #1e293b !important; border-right: 1pt solid #e2e8f0 !important; border-top: 1pt solid #e2e8f0 !important; border-bottom: 1pt solid #e2e8f0 !important; background-color: #f8fafc !important; }
          .design-professional .header-row { background-color: #334155 !important; text-align: left !important; padding-left: 20pt !important; }

          .design-bold-red { border: 6pt solid #991b1b !important; border-radius: 0 !important; }
          .design-bold-red .header-row { background-color: #991b1b !important; color: white !important; font-weight: 900 !important; }

          .design-royal-gold { border: 4pt solid #854d0e !important; background-color: #fefce8 !important; }
          .design-royal-gold .header-row { background-color: #854d0e !important; color: #fefce8 !important; }

          .design-deep-ocean { border-top: 20pt solid #1e3a8a !important; border-bottom: 20pt solid #1e3a8a !important; }
          .design-deep-ocean .header-row { background-color: #1e3a8a !important; color: white !important; }

          .design-sunset-vibrant { border: 4pt solid #be123c !important; background: linear-gradient(180deg, #fff1f2 0%, #ffffff 100%) !important; }
          .design-sunset-vibrant .header-row { background: linear-gradient(90deg, #be123c, #fb7185) !important; color: white !important; }

          .design-cyberpunk { border: 2pt solid #000 !important; box-shadow: 10pt 10pt 0 #facc15 !important; }
          .design-cyberpunk .header-row { background-color: #000 !important; color: #facc15 !important; text-transform: uppercase !important; font-family: monospace !important; }

          .design-academic-heavy { border: 2pt solid black !important; padding: 1in !important; }
          .design-academic-heavy .header-row { border-bottom: 4pt solid black !important; background: transparent !important; color: black !important; font-size: 16pt !important; }

          .design-art-deco { border: 8pt double #4338ca !important; }
          .design-art-deco .header-row { background-color: #4338ca !important; color: white !important; letter-spacing: 0.3em !important; }

          .design-futuristic { border: 1pt solid #0ea5e9 !important; background-color: #f0f9ff !important; clip-path: polygon(0% 0%, 100% 0%, 100% 95%, 95% 100%, 0% 100%); }
          .design-futuristic .header-row { background-color: #0ea5e9 !important; color: white !important; }

          /* Column/Table Designs 21-30 */
          .design-col-table-1 { background: #f8fafc; border: 1pt solid #e2e8f0; }
          .design-col-table-2 { background: #fff7ed; border: 1pt solid #fdba74; }
          .design-col-table-3 { background: #f0fdf4; border: 1pt solid #86efac; }
          .design-col-table-4 { background: #eff6ff; border: 1pt solid #93c5fd; }
          .design-col-table-5 { background: #faf5ff; border: 1pt solid #d8b4fe; }
          .design-col-table-6 { background: #fff1f2; border: 1pt solid #fda4af; }
          .design-col-table-7 { background: #fdf2f8; border: 1pt solid #f9a8d4; }
          .design-col-table-8 { background: #f0f9ff; border: 1pt solid #7dd3fc; }
          .design-col-table-9 { background: #f5f3ff; border: 1pt solid #c4b5fd; }
          .design-col-table-10 { background: #ecfdf5; border: 1pt solid #6ee7b7; }

          .design-col-table-1 .professional-table { border: 2pt solid #334155; }
          .design-col-table-2 .professional-table { border: 2pt solid #c2410c; }
          .design-col-table-3 .professional-table { border: 2pt solid #15803d; }
          .design-col-table-4 .professional-table { border: 2pt solid #1d4ed8; }
          .design-col-table-5 .professional-table { border: 2pt solid #7e22ce; }
          .design-col-table-6 .professional-table { border: 2pt solid #be123c; }
          .design-col-table-7 .professional-table { border: 2pt solid #be185d; }
          .design-col-table-8 .professional-table { border: 2pt solid #0369a1; }
          .design-col-table-9 .professional-table { border: 2pt solid #5b21b6; }
          .design-col-table-10 .professional-table { border: 2pt solid #047857; }
          
          .design-elegant { border: 1pt solid #92400e !important; background-color: #fffbeb !important; position: relative; }
          .design-elegant::before { content: ''; position: absolute; inset: 10pt; border: 1pt solid #d97706; pointer-events: none; }
          .design-elegant .header-row { background-color: #92400e !important; font-family: 'Garamond', serif !important; font-style: italic !important; }
          
          .design-technical { border: 2pt solid #0f172a !important; background-image: linear-gradient(#e2e8f0 1px, transparent 1px), linear-gradient(90deg, #e2e8f0 1px, transparent 1px) !important; background-size: 30px 30px !important; }
          .design-technical .header-row { background-color: #0f172a !important; font-family: 'Courier New', monospace !important; }
          
          .design-eco { border: 6pt solid #166534 !important; border-radius: 60pt 10pt 60pt 10pt !important; background-color: #f0fdf4 !important; }
          .design-eco .header-row { background-color: #166534 !important; border-radius: 40pt 0 0 0 !important; }
          
          .design-contrast { border: 12pt solid black !important; padding: 0.5in !important; }
          .design-contrast .header-row { background-color: black !important; color: white !important; font-size: 1.5em !important; letter-spacing: -0.05em !important; }

          .design-two-fold { 
            border: 1pt solid #cbd5e1 !important; 
            background: linear-gradient(90deg, #ffffff 49.5%, #e2e8f0 50%, #ffffff 50.5%) !important;
            box-shadow: 0 20px 50px rgba(0,0,0,0.1) !important;
            padding: 1in !important;
          }
          .design-two-fold .header-row { background-color: #64748b !important; }

          .design-projector {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 5pt solid #ffffff !important;
            padding: 1.5in !important;
          }
          .design-projector .header-row { background-color: #ffffff !important; color: #000000 !important; font-size: 2em !important; font-weight: 900 !important; }
          .design-projector td { border: 1pt solid #ffffff !important; color: #ffffff !important; }
          
          .design-modern-round {
            border: 2pt solid #6366f1 !important;
            border-radius: 50pt !important;
            padding: 1.2in !important;
            background-color: #f5f3ff !important;
          }
          .design-modern-round .header-row { background-color: #6366f1 !important; border-radius: 40pt 40pt 0 0 !important; }

          /* MCQ Design Variety Support */
          .prose .mcq-blank-start {
            display: inline-block;
            margin-right: 10pt;
            border-bottom: 1pt solid black;
            min-width: 40pt;
            text-align: center;
          }
          .prose u {
            text-decoration: none !important;
            border-bottom: 1pt solid black;
            display: inline-block;
            min-width: 25pt;
            text-align: center;
            margin-right: 5pt;
          }

          /* Matching Variety Support */
          .prose .matching-blank {
            display: inline-block;
            width: 80pt;
            border-bottom: 1pt solid #94a3b8;
            margin-right: 10pt;
            vertical-align: bottom;
          }
          .prose .word-bank-box {
            border: 1.5pt solid #334155;
            padding: 15pt;
            margin: 15pt 0;
            background-color: #f8fafc;
            border-radius: 4pt;
            display: flex;
            flex-wrap: wrap;
            gap: 15pt;
            justify-content: center;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.05em;
          }
          .prose .word-bank-box-alt {
            border: 2pt double #1e293b;
            padding: 12pt;
            margin: 15pt 0;
            background-color: #ffffff;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(80pt, 1fr));
            gap: 10pt;
            text-align: center;
            font-family: 'Courier New', monospace;
          }
          .prose .word-bank-item {
            padding: 2pt 8pt;
          }
          .prose .word-bank-item-box {
            border: 1pt solid #cbd5e1;
            padding: 4pt;
            background: #f1f5f9;
          }

          /* Ruler Table Style */
          .prose .ruler-table {
            border-collapse: collapse !important;
            width: 100% !important;
            border: none !important;
            margin-bottom: 20pt !important;
          }
          .prose .ruler-table td {
            border: none !important;
            padding: 15pt !important;
            vertical-align: top !important;
          }
          .prose .ruler-table td:first-child:not([colspan="2"]) {
            border-right: 1.5pt solid #ff0000 !important;
          }
          .prose .ruler-table td[colspan="2"] {
            border-right: none !important;
            border-bottom: 1.5pt solid black !important;
          }

          /* Relaxing Part Backgrounds */
          .prose .bg-relax-blue { background-color: #f0f9ff !important; border-radius: 8pt; padding: 15pt !important; margin-bottom: 15pt !important; }
          .prose .bg-relax-green { background-color: #f0fdf4 !important; border-radius: 8pt; padding: 15pt !important; margin-bottom: 15pt !important; }
          .prose .bg-relax-yellow { background-color: #fffbeb !important; border-radius: 8pt; padding: 15pt !important; margin-bottom: 15pt !important; }
          .prose .bg-relax-purple { background-color: #f5f3ff !important; border-radius: 8pt; padding: 15pt !important; margin-bottom: 15pt !important; }
          .prose .bg-relax-rose { background-color: #fff1f2 !important; border-radius: 8pt; padding: 15pt !important; margin-bottom: 15pt !important; }
          .prose .bg-relax-ocean { 
            background-image: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%) !important; 
            border-left: 5pt solid #0ea5e9 !important;
            border-radius: 0 8pt 8pt 0;
            padding: 15pt !important; 
            margin-bottom: 15pt !important; 
          }
          .prose .bg-relax-mist { 
            background-image: linear-gradient(135deg, #f1f5f9 0%, #f8fafc 100%) !important; 
            border: 1pt solid #e2e8f0 !important;
            border-radius: 12pt;
            padding: 15pt !important; 
            margin-bottom: 15pt !important; 
          }
          .prose .bg-relax-forest { 
            background-image: linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%) !important; 
            border-right: 5pt solid #166534 !important;
            border-radius: 8pt 0 0 8pt;
            padding: 15pt !important; 
            margin-bottom: 15pt !important; 
          }

          /* MCQ Style Support - Base Styles */
          .prose b, .prose strong {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 1.2em;
            height: 1.2em;
            margin-right: 0.4em;
            transition: all 0.2s;
            border: none;
            background: transparent;
            font-weight: 700 !important;
          }

          /* MCQ Style 0: Standard (No special styling) */
          ${mcqStyle === 0 ? `
            .prose b, .prose strong { 
              display: inline; 
              min-width: auto; 
              height: auto; 
              margin-right: 0.2em;
              border: none !important;
              background: transparent !important;
              border-radius: 0 !important;
              padding: 0 !important;
            }
          ` : ''}

          /* MCQ Style 1: Rounded Badge (Styled by Design) */
          ${mcqStyle === 1 ? `
            .design-modern-blue b, .design-modern-blue strong { background: #eff6ff; border: 0.5pt solid #2563eb; border-radius: 50% !important; color: #1e40af; }
            .design-classic b, .design-classic strong { background: transparent; border: 0.5pt solid black; border-radius: 50% !important; font-family: 'Georgia', serif; }
            .design-minimalist b, .design-minimalist strong { background: #f8fafc; border: none; border-bottom: 1pt solid black; border-radius: 0 !important; }
            .design-playful b, .design-playful strong { background: #ffedd5; border: 1pt dashed #f97316; border-radius: 50% !important; color: #9a3412; }
            .design-professional b, .design-professional strong { background: #f1f5f9; border-left: 3pt solid #334155; border-radius: 0 !important; padding-left: 4px; width: auto; min-width: 1.8em; }
            .design-elegant b, .design-elegant strong { background: #fef3c7; border: 0.5pt solid #92400e; border-radius: 50% 0 50% 0 !important; color: #78350f; }
            .design-technical b, .design-technical strong { background: #0f172a; color: white; border-radius: 0 !important; font-family: 'Courier New', monospace; }
            .design-eco b, .design-eco strong { background: #dcfce7; border: 1pt solid #166534; border-radius: 12px 4px !important; color: #14532d; }
            .design-contrast b, .design-contrast strong { background: black; color: white; border-radius: 0 !important; transform: rotate(-3deg); }
            .design-two-fold b, .design-two-fold strong { background: white; border: 0.5pt solid #cbd5e1; border-radius: 50% !important; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
            .design-projector b, .design-projector strong { background: white; color: black; border-radius: 50% !important; font-weight: 900 !important; }
            .design-modern-round b, .design-modern-round strong { background: #e0e7ff; border: 1pt solid #6366f1; border-radius: 50% !important; color: #4338ca; }
            
            /* Green Circles for Notebook and Emerald Layouts */
            .layout-notebook b, .layout-notebook strong,
            .layout-modern-emerald b, .layout-modern-emerald strong,
            .layout-mint b, .layout-mint strong,
            .layout-leaves b, .layout-leaves strong {
              background: #ecfdf5 !important;
              border: 1pt solid #059669 !important;
              color: #059669 !important;
              border-radius: 50% !important;
            }
            
            /* Default if no design class matches */
            .worksheet-page:not([class*="design-"]) b, .worksheet-page:not([class*="design-"]) strong {
              background: transparent;
              border-radius: 50% !important;
              border: 0.5pt solid #cbd5e1;
              font-size: 0.9em;
              color: inherit;
              font-weight: 700 !important;
            }
          ` : ''}

          /* MCQ Style 2: Boxed Letters */
          ${mcqStyle === 2 ? `
            .prose b, .prose strong {
              border: 1pt solid #475569;
              border-radius: 4px !important;
              background: #f8fafc;
              color: #1e293b;
              padding: 0 4px;
              min-width: 1.4em;
            }
          ` : ''}

          /* MCQ Style 3: Parentheses */
          ${mcqStyle === 3 ? `
            .prose b, .prose strong {
              display: inline;
              background: transparent !important;
              border: none !important;
              padding: 0 !important;
              margin-right: 0.2em;
            }
            .prose b::before, .prose strong::before { content: '('; }
            .prose b::after, .prose strong::after { content: ')'; }
          ` : ''}

          /* MCQ Style 4: Underlined Letter */
          ${mcqStyle === 4 ? `
            .prose b, .prose strong {
              display: inline;
              background: transparent !important;
              border: none !important;
              border-bottom: 1.5pt solid currentColor !important;
              border-radius: 0 !important;
              padding: 0 !important;
              min-width: auto;
              height: auto;
              margin-right: 0.3em;
            }
          ` : ''}

          /* MCQ Style 5: Bold Letter */
          ${mcqStyle === 5 ? `
            .prose b, .prose strong {
              display: inline;
              background: transparent !important;
              border: none !important;
              padding: 0 !important;
              min-width: auto;
              height: auto;
              font-weight: 900 !important;
              font-size: 1.1em;
              margin-right: 0.2em;
            }
          ` : ''}

          /* MCQ Style 6: Diamond */
          ${mcqStyle === 6 ? `
            .prose b, .prose strong {
              display: inline-flex;
              align-items: center;
              justify-content: center;
              width: 1.6em;
              height: 1.6em;
              border: 1pt solid currentColor !important;
              transform: rotate(45deg);
              margin-right: 0.5em;
              background: transparent !important;
            }
            .prose b span, .prose strong span { transform: rotate(-45deg); }
          ` : ''}

          /* MCQ Style 7: Bracketed Letter */
          ${mcqStyle === 7 ? `
            .prose b, .prose strong {
              display: inline;
              background: transparent !important;
              border: none !important;
              padding: 0 !important;
              margin-right: 0.2em;
            }
            .prose b::before, .prose strong::before { content: '['; }
            .prose b::after, .prose strong::after { content: ']'; }
          ` : ''}

          /* MCQ Style 8: Circle Fill */
          ${mcqStyle === 8 ? `
            .prose b, .prose strong {
              display: inline-flex;
              align-items: center;
              justify-content: center;
              width: 1.6em;
              height: 1.6em;
              background-color: currentColor !important;
              color: #fff !important;
              border-radius: 50% !important;
              font-weight: bold !important;
              font-size: 0.9em;
              margin-right: 0.4em;
            }
          ` : ''}

          /* MCQ Style 9: Square Fill */
          ${mcqStyle === 9 ? `
            .prose b, .prose strong {
              display: inline-flex;
              align-items: center;
              justify-content: center;
              width: 1.6em;
              height: 1.6em;
              background-color: currentColor !important;
              color: #fff !important;
              font-weight: bold !important;
              font-size: 0.9em;
              margin-right: 0.4em;
            }
          ` : ''}

          /* MCQ Style 10: Double Parentheses */
          ${mcqStyle === 10 ? `
            .prose b, .prose strong {
              display: inline;
              background: transparent !important;
              border: none !important;
              padding: 0 !important;
              margin-right: 0.2em;
            }
            .prose b::before, .prose strong::before { content: '(('; }
            .prose b::after, .prose strong::after { content: '))'; }
          ` : ''}

          /* Design-Specific Table Overrides */
          .design-minimalist table { border: none !important; border-top: 2pt solid black !important; border-bottom: 2pt solid black !important; }
          .design-minimalist td, .design-minimalist th { border: none !important; border-bottom: 1pt solid #e2e8f0 !important; }
          
          .design-playful table { border: 3pt dashed #f97316 !important; border-radius: 20pt !important; overflow: hidden; }
          .design-playful td { border: 1.5pt dashed #fdba74 !important; }
          
          .design-technical table { border: 1.5pt solid #0f172a !important; font-family: 'Courier New', monospace !important; }
          .design-technical td { border: 1pt solid #334155 !important; }

          .design-elegant table { border: 1pt solid #92400e !important; }
          .design-elegant td { border: 0.5pt solid #d97706 !important; }

          .design-contrast table { border: 4pt solid black !important; }
          .design-contrast td { border: 2pt solid black !important; }

          .design-modern-blue table { border: 1.5pt solid #2563eb !important; }
          .design-modern-blue td { border: 0.5pt solid #bfdbfe !important; }
          .design-modern-blue tr:nth-child(even) td { background-color: #eff6ff; }

          .design-classic table { border: 2pt solid black !important; outline: 0.5pt solid black !important; outline-offset: 2pt !important; }
          .design-classic td { border: 0.5pt solid black !important; font-family: 'Georgia', serif !important; }

          .design-professional table { border: 1pt solid #334155 !important; border-left: 6pt solid #334155 !important; }
          .design-professional td { border: 0.5pt solid #e2e8f0 !important; }

          .design-eco table { border: 2pt solid #166534 !important; border-radius: 10pt !important; overflow: hidden; }
          .design-eco td { border: 1pt solid #bbf7d0 !important; background-color: #f0fdf4; }

          .design-modern-round table { border: 2pt solid #6366f1 !important; border-radius: 15pt !important; overflow: hidden; }
          .design-modern-round td { border: 1pt solid #e0e7ff !important; }
          .design-modern-round tr:nth-child(even) td { background-color: #f5f3ff; }

          .design-modern-blue .professional-table thead th { background-color: #2563eb !important; }
          .design-classic .professional-table thead th { background-color: black !important; font-family: 'Georgia', serif !important; }
          .design-playful .professional-table thead th { background-color: #f97316 !important; border-radius: 10pt 10pt 0 0 !important; }
          .design-technical .professional-table thead th { background-color: #0f172a !important; font-family: 'Courier New', monospace !important; }
          .design-eco .professional-table thead th { background-color: #166534 !important; }
          .design-modern-round .professional-table thead th { background-color: #6366f1 !important; border-radius: 15pt 15pt 0 0 !important; }

          .worksheet-footer {
            margin-top: 40pt;
            border-top: 1pt solid #e2e8f0;
            padding-top: 10pt;
            text-align: center;
            font-size: 8pt;
            color: #94a3b8;
            font-style: italic;
          }

          @media print {
            .no-print { display: none !important; }
            .bg-white { background-color: white !important; }
          /* Global Layouts - Structural Fix */
          .layout-lined p, .layout-lined div:not([style*="background"]) { 
            border-bottom: 1pt solid #cbd5e1 !important; 
            margin-bottom: 0 !important;
            padding-bottom: 4pt !important;
          }
          .layout-grid {
            background-image: linear-gradient(#e5e7eb 1px, transparent 1px), linear-gradient(90deg, #e5e7eb 1px, transparent 1px) !important;
            background-size: 20pt 20pt !important;
            background-attachment: local !important;
          }
          .layout-rulers {
            border-left: 40pt solid #f1f5f9 !important;
            position: relative;
          }
          .layout-rulers::before {
            content: '';
            position: absolute;
            left: -40pt;
            top: 0;
            bottom: 0;
            width: 2pt;
            background-color: #fca5a5;
            margin-left: 35pt;
          }

          .layout-vertical-middle {
            position: relative;
          }
          /* Removed ::after red line overlay */

          .layout-s1 p, .layout-s1 div:not([style*="background"]) {
            border-bottom: 1pt solid #cbd5e1 !important;
          }
          .layout-s2 {
            border-left: 2pt solid #94a3b8 !important;
          }
          .layout-s2 p, .layout-s2 div:not([style*="background"]) {
            border-bottom: 1pt solid #e2e8f0 !important;
          }
          .layout-s3 p, .layout-s3 div:not([style*="background"]) {
            border-bottom: 1pt solid #94a3b8 !important;
          }

          }
          .layout-s4 {
            background-image: linear-gradient(90deg, #f1f5f9 1px, transparent 1px), linear-gradient(#f1f5f9 1px, transparent 1px) !important;
            background-size: 40pt 40pt !important;
          }

          /* Instruction Ruler Styles */
          .instruction-ruler-1 { height: 1pt; background: #000; width: 100%; margin: 15pt 0; }
          .instruction-ruler-2 { height: 2pt; border-top: 2pt dashed #000; width: 100%; margin: 15pt 0; }
          .instruction-ruler-3 { height: 4pt; border-top: 1pt solid #000; border-bottom: 1pt solid #000; width: 100%; margin: 15pt 0; }
          .instruction-ruler-4 { height: 6pt; background: linear-gradient(90deg, #334155 0%, #94a3b8 50%, #334155 100%); border-radius: 3pt; width: 100%; margin: 15pt 0; }
          .instruction-ruler-5 { 
            height: 20pt; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            gap: 10pt; 
            color: #fcd34d; 
            font-size: 14pt; 
            margin: 15pt 0;
            position: relative;
          }
          .instruction-ruler-5::before, .instruction-ruler-5::after {
            content: '';
            flex: 1;
            height: 1pt;
            background: #e2e8f0;
          }
          .instruction-ruler-6 { 
            height: 20pt; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            gap: 10pt; 
            color: #fca5a5; 
            font-size: 14pt; 
            margin: 15pt 0;
            position: relative;
          }
          .instruction-ruler-6::before, .instruction-ruler-6::after {
            content: '';
            flex: 1;
            height: 1pt;
            background: #e2e8f0;
          }

          /* Colorful Layouts */
          .layout-clean-white { background-color: #ffffff !important; border: 2pt solid #f1f5f9 !important; box-shadow: inset 0 0 0 10pt #f8fafc !important; }
          .layout-orange-mix { background-color: #ffffff !important; border-left: 15pt solid #f97316 !important; border-right: 2pt solid #fff7ed !important; }
          .layout-modern-emerald { background-color: #f0fdf4 !important; border: 2pt solid #059669 !important; border-left: 10pt solid #059669 !important; }
          .layout-soft-lavender { background-color: #faf5ff !important; border: 1pt solid #d8b4fe !important; border-top: 12pt solid #9333ea !important; }
          
          .layout-mint { background-color: #f0fdf4 !important; border: 1pt solid #dcfce7 !important; }
          .layout-peach { background-color: #fff7ed !important; border: 1pt solid #ffedd5 !important; }
          .layout-sky { background-color: #f0f9ff !important; border: 1pt solid #e0f2fe !important; }
          .layout-lavender { background-color: #f5f3ff !important; border: 1pt solid #ede9fe !important; }
          .layout-citrus { 
            background: linear-gradient(135deg, #f0fdf4 0%, #fff7ed 100%) !important; 
            border: 1pt solid #dcfce7 !important;
          }
          .layout-rose { background-color: #fff1f2 !important; border: 1pt solid #ffe4e6 !important; }

          /* Decorative Layouts - Frame Styles */
          .layout-stars { 
            background-color: #fff !important; 
            border: 15pt solid #fff !important;
            outline: 2pt solid #fcd34d !important;
            outline-offset: -10pt !important;
          }
          .layout-flowers { 
            background-color: #fff !important; 
            border: 20pt solid #fff !important;
            outline: 3pt double #f9a8d4 !important;
            outline-offset: -12pt !important;
          }
          .layout-hearts { 
            background-color: #fff !important; 
            border: 15pt solid #fff !important;
            outline: 2pt dashed #fca5a5 !important;
            outline-offset: -8pt !important;
          }
          .layout-bubbles { 
            background-color: #f0f9ff !important; 
            border: 20pt solid #f0f9ff !important;
            outline: 4pt solid #bae6fd !important;
            outline-offset: -15pt !important;
          }
          .layout-leaves { 
            background-color: #f0fdf4 !important; 
            border: 15pt solid #f0fdf4 !important;
            outline: 2pt solid #86efac !important;
            outline-offset: -10pt !important;
          }
          .layout-rainbow { 
            background: linear-gradient(to bottom, #fee2e2, #fef3c7, #ecfdf5, #eff6ff, #f5f3ff) !important; 
            opacity: 0.8;
          }
          .layout-galaxy { 
            background-color: #0f172a !important; 
            color: #f8fafc !important;
            background-image: radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 40px), radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 30px), radial-gradient(white, rgba(255,255,255,.1) 2px, transparent 40px) !important;
            background-size: 550px 550px, 350px 350px, 250px 250px !important;
          }
          .layout-notebook { 
            background-color: #fff !important; 
            background-image: linear-gradient(#e5e7eb 1px, transparent 1px) !important;
            background-size: 100% 24pt !important;
            border-left: 4pt double #ef4444 !important;
            padding-left: 45pt !important;
          }
          .layout-vintage { 
            background-color: #fef3c7 !important; 
            background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100' height='100' filter='url(%23noise)' opacity='0.05'/%3E%3C/svg%3E") !important;
          }
          .layout-modern { 
            background: linear-gradient(45deg, #f8fafc 25%, #f1f5f9 25%, #f1f5f9 50%, #f8fafc 50%, #f8fafc 75%, #f1f5f9 75%, #f1f5f9 100%) !important;
            background-size: 100px 100px !important;
          }
        `}</style>
        
        <div 
          className={clsx(
            "worksheet-page prose min-h-[297mm] p-[0.8in_1in] relative rounded-sm",
            PAPER_DESIGNS[paperDesign] || '',
            isColorfulBackgroundEnabled ? 'bg-blue-50/50' : 'bg-white',
            baseLayout === 1 && 'layout-lined',
            baseLayout === 2 && 'layout-grid',
            baseLayout === 3 && 'layout-vertical-middle',
            baseLayout === 4 && 'layout-rulers',
            baseLayout === 5 && 'layout-s1',
            baseLayout === 6 && 'layout-s2',
            baseLayout === 7 && 'layout-s3',
            baseLayout === 8 && 'layout-s4',
            globalLayout === 0 && 'layout-clean-white',
            globalLayout === 1 && 'layout-orange-mix',
            globalLayout === 2 && 'layout-modern-emerald',
            globalLayout === 3 && 'layout-soft-lavender',
            globalLayout === 4 && 'layout-mint',
            globalLayout === 5 && 'layout-peach',
            globalLayout === 6 && 'layout-sky',
            globalLayout === 7 && 'layout-lavender',
            globalLayout === 8 && 'layout-citrus',
            globalLayout === 9 && 'layout-rose',
            globalLayout === 10 && 'layout-stars',
            globalLayout === 11 && 'layout-flowers',
            globalLayout === 12 && 'layout-hearts',
            globalLayout === 13 && 'layout-bubbles',
            globalLayout === 14 && 'layout-leaves',
            globalLayout === 15 && 'layout-rainbow',
            globalLayout === 16 && 'layout-galaxy',
            globalLayout === 17 && 'layout-notebook',
            globalLayout === 18 && 'layout-vintage',
            globalLayout === 19 && 'layout-modern'
          )}
               ref={editorRef}
               contentEditable={!isGenerating}
               onInput={(e) => onContentChange(e.currentTarget.innerHTML)}
               dangerouslySetInnerHTML={{ __html: content || placeholderHtml }}
          >
        </div>
        <div id="decorative-elements-container" className="absolute inset-0 pointer-events-none overflow-hidden">
          {decorativeElements}
        </div>
      </div>
    </div>
  );
};

export default Worksheet;
