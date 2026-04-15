import { saveAs } from 'file-saver';
import { jsPDF } from 'jspdf';
import { toPng } from 'html-to-image';

export interface ExportMetadata {
  author?: string;
  date?: string;
  title?: string;
}

export const exportToWord = async (
  htmlContent: string, 
  filename: string, 
  headerHtml: string = '', 
  marginValue: string = '0.4in 0.6in 0.4in 0.6in',
  fontFamily: string = 'Times New Roman',
  lineHeight: string = '1.15',
  metadata?: ExportMetadata,
  isFrameEnabled: boolean = false,
  activeDesign: string = '',
  paperStyles?: any,
  mcqStyle: number = 0,
  globalLayout: number = 0,
  baseLayout: number = 0,
  instructionRulerStyle: number = 0,
  instructionHeaderStyle: number = 0
) => {
  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = htmlContent;

  const headerDiv = document.createElement('div');
  headerDiv.innerHTML = headerHtml;

  // Randomize Ruler Color if it's the middle ruler layout
  const rulerColors = ['#ff0000', '#2563eb', '#16a34a', '#d97706', '#7c3aed', '#db2777', '#0f172a'];
  const activeRulerColor = rulerColors[Math.floor(Math.random() * rulerColors.length)];

  const linePercentage = `200%`;
  const exactLineHeight = `24pt`;

  // 1. FIX: Convert all images to Base64 (This prevents "Empty Boxes")
  const images = [...Array.from(tempDiv.querySelectorAll('img')), ...Array.from(headerDiv.querySelectorAll('img'))];
  for (const img of images) {
    try {
      const response = await fetch(img.src);
      const blob = await response.blob();
      const base64 = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.readAsDataURL(blob);
      });
      img.src = base64 as string;
      
      // Force fixed size so they don't overlap
      // We use inches for Word compatibility
      const originalWidth = img.width || 550;
      
      // Check if it's a logo (often has specific styles or is in header)
      const isLogo = img.style.maxHeight === '80pt' || img.classList.contains('logo') || headerDiv.contains(img);

      if (isLogo) {
        img.setAttribute('width', '120');
        img.style.width = '1.25in';
        img.style.height = 'auto';
      } else if (originalWidth > 200) {
        // Large images (like Quest Lab images)
        img.setAttribute('width', '550'); 
        img.style.width = '6.5in'; // Adjusted for 1in margins (8.5 - 2 = 6.5)
        img.style.height = 'auto';
      } else if (originalWidth < 50) {
        // Small icons
        img.setAttribute('width', '45');
        img.style.width = '0.45in';
        img.style.height = 'auto';
      } else {
        // Medium images - preserve relative size
        const inWidth = (originalWidth / 96).toFixed(2);
        img.setAttribute('width', originalWidth.toString());
        img.style.width = `${inWidth}in`;
        img.style.height = 'auto';
      }
      img.style.display = 'block';
      if (!isLogo) img.style.margin = '5px auto';
    } catch (e) {
      console.warn("Could not convert image to base64", e);
    }
  }

  // 2. FIX: Handle Round MCQ Badges for Word
  const designClass = activeDesign || '';

  const badges = tempDiv.querySelectorAll('b, strong, span');
  badges.forEach(badge => {
    const text = badge.textContent?.trim() || "";
    // Match "A", "B", "C", "D" or "A.", "B." or "(A)", "[A]" etc.
    const isOptionLetter = /^[\[\(]?[A-D][\]\)]?\.?$/.test(text);
    
    if (isOptionLetter) {
      (badge as HTMLElement).style.display = 'inline-block';
      (badge as HTMLElement).style.width = '22pt';
      (badge as HTMLElement).style.height = '22pt';
      (badge as HTMLElement).style.lineHeight = '22pt';
      (badge as HTMLElement).style.textAlign = 'center';
      (badge as HTMLElement).style.marginRight = '6pt';
      (badge as HTMLElement).style.fontWeight = 'bold';
      (badge as HTMLElement).style.fontSize = '10pt';
      (badge as HTMLElement).style.verticalAlign = 'middle';

      // Force Boxed/Circled based on paperStyles if mcqStyle > 0
      const forceBadge = mcqStyle > 0;

      if (forceBadge) {
        // Design-Specific Word Fallbacks
        if (designClass === 'design-modern-blue') {
          (badge as HTMLElement).style.border = '1.5pt solid #2563eb';
          (badge as HTMLElement).style.backgroundColor = '#eff6ff';
          (badge as HTMLElement).style.color = '#1e40af';
          (badge as HTMLElement).style.borderRadius = '11pt';
        } else if (designClass === 'design-classic') {
          (badge as HTMLElement).style.border = '1pt solid black';
          (badge as HTMLElement).style.backgroundColor = 'transparent';
          (badge as HTMLElement).style.borderRadius = '0';
        } else if (designClass === 'design-playful') {
          (badge as HTMLElement).style.border = '2pt solid #f97316';
          (badge as HTMLElement).style.backgroundColor = '#ffedd5';
          (badge as HTMLElement).style.color = '#9a3412';
          (badge as HTMLElement).style.borderRadius = '11pt';
        } else if (designClass === 'design-technical') {
          (badge as HTMLElement).style.backgroundColor = '#0f172a';
          (badge as HTMLElement).style.color = '#ffffff';
          (badge as HTMLElement).style.border = 'none';
          (badge as HTMLElement).style.borderRadius = '0';
        } else if (designClass === 'design-elegant') {
          (badge as HTMLElement).style.border = '1pt solid #92400e';
          (badge as HTMLElement).style.backgroundColor = '#fef3c7';
          (badge as HTMLElement).style.borderRadius = '11pt';
        } else if (designClass === 'design-contrast') {
          (badge as HTMLElement).style.backgroundColor = 'black';
          (badge as HTMLElement).style.color = 'white';
          (badge as HTMLElement).style.border = 'none';
          (badge as HTMLElement).style.borderRadius = '0';
        } else if (designClass === 'design-modern-round' || designClass === 'design-projector') {
          (badge as HTMLElement).style.border = '1.5pt solid #6366f1';
          (badge as HTMLElement).style.backgroundColor = '#e0e7ff';
          (badge as HTMLElement).style.color = '#4338ca';
          (badge as HTMLElement).style.borderRadius = '11pt';
        } else {
          // Default fallback for forced badges
          // Style 1 (Round) should NOT have a square border
          if (mcqStyle === 1 || mcqStyle === 8) {
            (badge as HTMLElement).style.border = 'none';
            (badge as HTMLElement).style.backgroundColor = 'transparent';
          } else if (mcqStyle === 2) {
            (badge as HTMLElement).style.border = '1pt solid black';
            (badge as HTMLElement).style.backgroundColor = '#f8fafc';
          } else {
            (badge as HTMLElement).style.border = 'none';
            (badge as HTMLElement).style.backgroundColor = 'transparent';
          }
          
          if (mcqStyle === 3 || mcqStyle === 1 || mcqStyle === 8) {
            (badge as HTMLElement).style.borderRadius = '11pt';
          } else {
            (badge as HTMLElement).style.borderRadius = '0';
          }
        }
      }
    }
  });

  // 2.1 FIX: Handle MCQ Blank Start and Underlined Letters for Word
  const specialElements = tempDiv.querySelectorAll('.mcq-blank-start, u, .blank-line, .checkbox-box, b, strong');
  specialElements.forEach(el => {
    if (el.classList.contains('mcq-blank-start') || el.classList.contains('blank-line')) {
      (el as HTMLElement).style.display = 'inline-block';
      (el as HTMLElement).style.width = '50pt';
      (el as HTMLElement).style.borderBottom = '1pt solid black';
      (el as HTMLElement).style.marginRight = '10pt';
      (el as HTMLElement).style.textAlign = 'center';
      (el as HTMLElement).style.textDecoration = 'none';
      (el as HTMLElement).innerHTML = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;';
    }
    if (el.classList.contains('checkbox-box')) {
      (el as HTMLElement).style.display = 'inline-block';
      (el as HTMLElement).style.width = '12pt';
      (el as HTMLElement).style.height = '12pt';
      (el as HTMLElement).style.border = '1pt solid black';
      (el as HTMLElement).style.marginRight = '5pt';
      (el as HTMLElement).style.verticalAlign = 'middle';
    }
    // If it's a <u> used for the "Letter on Line" style
    if (el.tagName === 'U' && el.textContent?.includes('\u00A0')) {
      (el as HTMLElement).style.borderBottom = '1pt solid black';
      (el as HTMLElement).style.textDecoration = 'none';
      (el as HTMLElement).style.padding = '0 5pt';
    }
    // MCQ Styling for Word
    if (mcqStyle > 0 && (el.tagName === 'B' || el.tagName === 'STRONG' || el.tagName === 'SPAN')) {
      let text = el.textContent?.trim().toUpperCase() || '';
      // AGGRESSIVE STRIP: Remove brackets, periods, and spaces
      text = text.replace(/[\(\)\[\]\.\s]/g, '');
      
      if (['A', 'B', 'C', 'D'].includes(text) && text.length === 1) {
        // Apply Style 1: Round (Unicode Circled)
        if (mcqStyle === 1) {
          if (text === 'A') el.innerHTML = 'Ⓐ';
          else if (text === 'B') el.innerHTML = 'Ⓑ';
          else if (text === 'C') el.innerHTML = 'Ⓒ';
          else if (text === 'D') el.innerHTML = 'Ⓓ';
          (el as HTMLElement).style.fontSize = '14pt';
          (el as HTMLElement).style.setProperty('mso-text-raise', '1pt');
        } 
        // Apply Style 2: Boxed
        else if (mcqStyle === 2) {
          el.innerHTML = `[${text}]`;
          (el as HTMLElement).style.border = '0.5pt solid #475569';
          (el as HTMLElement).style.padding = '0 2pt';
          (el as HTMLElement).style.backgroundColor = '#f8fafc';
        }
        // Apply Style 3: Parentheses
        else if (mcqStyle === 3) {
          el.innerHTML = `(${text})`;
        }
        // Apply Style 4: Underlined
        else if (mcqStyle === 4) {
          el.innerHTML = text;
          (el as HTMLElement).style.textDecoration = 'underline';
          (el as HTMLElement).style.borderBottom = '1pt solid black';
        }
        // Apply Style 5: Bold
        else if (mcqStyle === 5) {
          el.innerHTML = text;
          (el as HTMLElement).style.fontWeight = 'bold';
          (el as HTMLElement).style.fontSize = '13pt';
        }
        // Apply Style 6: Diamond
        else if (mcqStyle === 6) {
          el.innerHTML = `◆${text}`;
          (el as HTMLElement).style.fontSize = '12pt';
        }
        // Apply Style 7: Bracket
        else if (mcqStyle === 7) {
          el.innerHTML = `[${text}]`;
        }
        // Apply Style 8: Circle Fill
        else if (mcqStyle === 8) {
          if (text === 'A') el.innerHTML = '❶';
          else if (text === 'B') el.innerHTML = '❷';
          else if (text === 'C') el.innerHTML = '❸';
          else if (text === 'D') el.innerHTML = '❹';
          (el as HTMLElement).style.fontSize = '14pt';
        }
        // Apply Style 9: Square Fill
        else if (mcqStyle === 9) {
          el.innerHTML = `■${text}`;
          (el as HTMLElement).style.fontSize = '12pt';
        }
        // Apply Style 10: Double Paren
        else if (mcqStyle === 10) {
          el.innerHTML = `((${text}))`;
        }

        // Common cleanup for styled MCQs
        if (mcqStyle !== 0) {
          (el as HTMLElement).style.borderRadius = '0';
          (el as HTMLElement).style.display = 'inline-block';
          (el as HTMLElement).style.width = 'auto';
          
          // Apply colors based on layout (for Style 1, 2, 8, 9 mostly)
          if (mcqStyle === 1 || mcqStyle === 2 || mcqStyle === 8 || mcqStyle === 9) {
            if (globalLayout === 0) (el as HTMLElement).style.color = '#2563eb';
            else if (globalLayout === 1) (el as HTMLElement).style.color = '#ea580c';
            else if (globalLayout === 2 || globalLayout === 4 || globalLayout === 14 || globalLayout === 17) (el as HTMLElement).style.color = '#059669';
            else if (globalLayout === 3 || globalLayout === 7) (el as HTMLElement).style.color = '#9333ea';
            else if (globalLayout === 5 || globalLayout === 8) (el as HTMLElement).style.color = '#f97316';
            else if (globalLayout === 6 || globalLayout === 13) (el as HTMLElement).style.color = '#0284c7';
            else if (globalLayout === 9 || globalLayout === 11 || globalLayout === 12) (el as HTMLElement).style.color = '#db2777';
            else if (globalLayout === 10) (el as HTMLElement).style.color = '#d97706';
            else (el as HTMLElement).style.color = '#334155';
          }
        }
      }
    }
  });

  // 2.2 FIX: Handle Header for Middle Ruler (Option 4)
  // REMOVED: Global master table for baseLayout === 3 to prevent "ruler everywhere" mistake.
  
  // This is the "Magic Fix" for Word
  let sections = Array.from(tempDiv.children);
  
  // If the only child is the prose div, unwrap it to process its children
  if (sections.length === 1 && sections[0].classList.contains('prose')) {
    sections = Array.from(sections[0].children);
  } else if (sections.length > 1) {
    // Filter out decorative container if it's at the top level
    sections = sections.filter(el => el.id !== 'decorative-elements-container');
    // If we have a prose div among others, we might need to unwrap it too
    const proseIndex = sections.findIndex(el => el.classList.contains('prose'));
    if (proseIndex !== -1) {
      const prose = sections[proseIndex];
      sections.splice(proseIndex, 1, ...Array.from(prose.children));
    }
  }

  let finalHtml = "";
  for (let i = 0; i < sections.length; i++) {
    const el = sections[i] as HTMLElement;
    
    if (el.id === 'decorative-elements-container' || el.classList.contains('decorative-element')) {
      continue;
    }

    // If it's a new Set title, force a page break
    const isNewSet = el.textContent?.toUpperCase().includes('(SET');
    const pageBreak = isNewSet && i > 0 ? 'style="page-break-before:always"' : '';
    
    if (el.style.backdropFilter || el.getAttribute('style')?.includes('backdrop-filter')) {
      el.style.backgroundColor = '#f8fafc';
      el.style.setProperty('mso-shading', 'windowtext 0% #f8fafc');
    }
    
    // If it's a "Part" header or instruction block
    if (el.classList.contains('bg-relax-blue') || el.style.backgroundColor === 'rgb(240, 249, 255)') {
      el.style.backgroundColor = '#f0f9ff';
      el.style.setProperty('mso-shading', 'windowtext 0% #f0f9ff');
    }
    // ... add more as needed

    finalHtml += `
      <table border="0" cellspacing="0" cellpadding="0" width="100%" ${pageBreak} style="margin: 0; padding: 0; border-collapse: collapse; mso-table-lspace:0pt; mso-table-rspace:0pt; width: 100%;">
        <tr>
          <td align="left" style="padding: 0; margin: 0; font-family: '${fontFamily}', serif; font-size: 12pt; line-height: ${exactLineHeight}; mso-line-height-rule: exactly; border: none;">
            ${el.outerHTML}
          </td>
        </tr>
      </table>`;
  }

  // 2.3 FIX: Handle existing instruction ruler divs
  const existingRulers = tempDiv.querySelectorAll('[class*="instruction-ruler-"]');
  existingRulers.forEach(ruler => {
    const el = ruler as HTMLElement;
    const match = el.className.match(/instruction-ruler-(\d+)/);
    const styleNum = match ? parseInt(match[1]) : 0;
    
    el.style.width = '100%';
    el.style.marginTop = '5pt';
    el.style.marginBottom = '10pt';
    el.style.height = '1pt';
    el.style.fontSize = '1pt';
    el.innerHTML = '&nbsp;';
    
    if (styleNum === 1) {
      el.style.borderBottom = '1pt solid #000000';
      el.style.setProperty('mso-border-bottom-alt', '1pt solid #000000');
    } else if (styleNum === 2) {
      el.style.borderBottom = '2pt dashed #000000';
      el.style.setProperty('mso-border-bottom-alt', '2pt dashed #000000');
    } else if (styleNum === 3) {
      el.style.borderBottom = '4pt double #000000';
      el.style.setProperty('mso-border-bottom-alt', '4pt double #000000');
    } else if (styleNum === 4) {
      el.style.borderBottom = '4pt solid #334155';
      el.style.setProperty('mso-border-bottom-alt', '4pt solid #334155');
    } else if (styleNum === 5) {
      el.style.textAlign = 'center';
      el.style.color = '#fcd34d';
      el.style.fontSize = '14pt';
      el.style.height = 'auto';
      el.innerHTML = '★ ★ ★';
    } else if (styleNum === 6) {
      el.style.textAlign = 'center';
      el.style.color = '#fca5a5';
      el.style.fontSize = '14pt';
      el.style.height = 'auto';
      el.innerHTML = '♥ ♥ ♥';
    }
  });

  // 3. FIX: Table Formatting (Ensures grids and nested MCQ tables are correct)
  const tables = tempDiv.querySelectorAll('table');
  tables.forEach(table => {
    const isNested = table.parentElement?.closest('table') !== null;
    
    if (isNested) {
      // Nested tables (like MCQ options) should have NO borders and 100% width
      table.setAttribute('border', '0');
      table.style.border = 'none';
      table.style.width = '100%';
      table.style.borderCollapse = 'collapse';
      const cells = table.querySelectorAll('td');
      cells.forEach((c) => {
        (c as HTMLElement).style.border = 'none';
        (c as HTMLElement).style.padding = '2pt';
        
        // Check if it's the first cell in its row
        const isFirstInRow = c.previousElementSibling === null;
        if (isFirstInRow) {
          (c as HTMLElement).style.paddingLeft = '30pt'; // Indent "A." by approx 7 spaces
        }
        (c as HTMLElement).style.verticalAlign = 'top';
        (c as HTMLElement).style.width = '25%'; // Default for 4-column MCQ
      });
    } else {
      // Top-level tables
      // Robust detection: check for class OR if it's a 2-column table with specific border styles
      const hasRulerClass = table.classList.contains('ruler-table') || table.className.includes('ruler-table');
      const isTwoCol = table.rows.length > 0 && table.rows[0].cells.length === 2;
      // Also check if any cell has a border-right style already
      const hasVerticalBorder = Array.from(table.querySelectorAll('td')).some(td => {
        const style = td.getAttribute('style') || '';
        return style.includes('border-right') && style.includes('solid');
      });
      
      const isRulerTable = hasRulerClass || hasVerticalBorder || (isTwoCol && table.getAttribute('data-type') === 'ruler') || (baseLayout === 3 && isTwoCol);
      
      // If it's a ruler table, we MUST force the border
      if (isRulerTable) {
        table.setAttribute('border', '0');
        table.style.border = 'none';
        table.style.width = '100%';
        table.style.borderCollapse = 'collapse'; // Changed from separate to collapse for better Word support
        (table.style as any).msoTableLspace = '0pt';
        (table.style as any).msoTableRspace = '0pt';
        table.style.margin = '0';
        
        const rows = Array.from(table.rows);
        rows.forEach((row) => {
          const cells = Array.from(row.cells);
          cells.forEach((c, idx) => {
            const isFirstCol = idx === 0;
            const isHeader = c.getAttribute('colspan') === '2' || (row === table.rows[0] && row.cells.length === 1);
            
            const cell = c as HTMLElement;
            cell.style.padding = '15pt';
            cell.style.verticalAlign = 'top';
            cell.style.border = 'none'; 
            
            if (isFirstCol && !isHeader && row.cells.length === 2) {
              // THIS IS THE CRITICAL RULER LINE - Using Randomized Color
              // We apply it to EVERY cell in the first column to ensure a continuous line
              cell.style.borderRight = `1.5pt solid ${activeRulerColor}`; 
              cell.style.setProperty('mso-border-right-alt', `1.5pt solid ${activeRulerColor}`);
              cell.style.width = '50%'; 
              cell.style.paddingRight = '15pt';
            }
            if (!isFirstCol && !isHeader && row.cells.length === 2) {
              cell.style.width = '50%';
              cell.style.paddingLeft = '15pt';
            }
            if (isHeader) {
              const cell = c as HTMLElement;
              
              // Base Style: Soft Green + Left Border (Applied to all regardless of style)
              cell.style.backgroundColor = '#dcfce7';
              cell.style.setProperty('mso-shading', 'windowtext 0% #dcfce7');
              cell.style.color = '#064e3b';
              cell.style.borderLeft = '6pt solid #059669';
              cell.style.setProperty('mso-border-left-alt', '6pt solid #059669');
              cell.style.textAlign = 'left';
              cell.style.paddingLeft = '15pt';
              cell.style.fontWeight = 'bold';

              // Apply Instruction Header Style Overrides
              if (instructionHeaderStyle === 0) {
                cell.style.backgroundColor = '#334155';
                cell.style.color = '#ffffff';
                cell.style.setProperty('mso-shading', 'windowtext 0% #334155');
                cell.style.textAlign = 'center';
              } else if (instructionHeaderStyle === 1) {
                cell.style.backgroundColor = '#dbeafe';
                cell.style.color = '#1e3a8a';
                cell.style.setProperty('mso-shading', 'windowtext 0% #dbeafe');
                cell.style.borderLeft = '6pt solid #1e3a8a';
                cell.style.setProperty('mso-border-left-alt', '6pt solid #1e3a8a');
                cell.style.textAlign = 'left';
                cell.style.paddingLeft = '15pt';
              } else if (instructionHeaderStyle === 2) {
                cell.style.backgroundColor = '#dcfce7';
                cell.style.color = '#064e3b';
                cell.style.setProperty('mso-shading', 'windowtext 0% #dcfce7');
                cell.style.borderLeft = '6pt solid #064e3b';
                cell.style.setProperty('mso-border-left-alt', '6pt solid #064e3b');
                cell.style.textAlign = 'left';
                cell.style.paddingLeft = '15pt';
              } else if (instructionHeaderStyle === 3) {
                cell.style.backgroundColor = '#fee2e2';
                cell.style.color = '#7f1d1d';
                cell.style.setProperty('mso-shading', 'windowtext 0% #fee2e2');
                cell.style.borderLeft = '6pt solid #7f1d1d';
                cell.style.setProperty('mso-border-left-alt', '6pt solid #7f1d1d');
                cell.style.textAlign = 'left';
                cell.style.paddingLeft = '15pt';
              } else if (instructionHeaderStyle === 4) {
                cell.style.border = '2pt solid #334155';
                cell.style.setProperty('mso-border-alt', '2pt solid #334155');
                cell.style.color = '#334155';
                cell.style.backgroundColor = 'transparent';
                cell.style.textAlign = 'center';
              } else if (instructionHeaderStyle === 5) {
                cell.style.border = 'none';
                cell.style.borderBottom = '3pt solid #334155';
                cell.style.setProperty('mso-border-bottom-alt', '3pt solid #334155');
                cell.style.color = '#334155';
                cell.style.fontSize = '14pt';
                cell.style.fontWeight = '900';
                cell.style.textAlign = 'left';
                cell.style.padding = '10pt 0';
              } else if (instructionHeaderStyle === 6) {
                cell.style.border = 'none';
                cell.style.borderBottom = '4pt double #334155';
                cell.style.setProperty('mso-border-bottom-alt', '4pt double #334155');
                cell.style.color = '#334155';
                cell.style.textAlign = 'left';
                cell.style.padding = '8pt 0';
              } else if (instructionHeaderStyle === 7) {
                cell.style.backgroundColor = '#f1f5f9';
                cell.style.setProperty('mso-shading', 'windowtext 0% #f1f5f9');
                cell.style.color = '#1e293b';
                cell.style.border = '1pt solid #e2e8f0';
                cell.style.setProperty('mso-border-alt', '1pt solid #e2e8f0');
              } else if (instructionHeaderStyle === 8) {
                cell.style.backgroundColor = '#e0e7ff';
                cell.style.setProperty('mso-shading', 'windowtext 0% #e0e7ff');
                cell.style.color = '#3730a3';
                cell.style.borderRight = '6pt solid #3730a3';
                cell.style.setProperty('mso-border-right-alt', '6pt solid #3730a3');
                cell.style.textAlign = 'right';
                cell.style.paddingRight = '15pt';
              } else if (instructionHeaderStyle === 9) {
                cell.style.backgroundColor = '#fffbeb';
                cell.style.setProperty('mso-shading', 'windowtext 0% #fffbeb');
                cell.style.color = '#92400e';
                cell.style.border = '1.5pt dashed #92400e';
                cell.style.setProperty('mso-border-alt', '1.5pt dashed #92400e');
              } else if (instructionHeaderStyle === 10) {
                cell.style.border = 'none';
                cell.style.borderBottom = '1pt solid #e2e8f0';
                cell.style.setProperty('mso-border-bottom-alt', '1pt solid #e2e8f0');
                cell.style.color = '#334155';
                cell.style.textAlign = 'left';
                cell.style.padding = '5pt 0';
              } else if (instructionHeaderStyle === 11) {
                cell.style.backgroundColor = '#1e293b';
                cell.style.setProperty('mso-shading', 'windowtext 0% #1e293b');
                cell.style.color = '#ffffff';
                cell.style.textAlign = 'center';
                cell.style.padding = '12pt';
              } else if (instructionHeaderStyle === 12) {
                cell.style.backgroundColor = '#ecfdf5';
                cell.style.setProperty('mso-shading', 'windowtext 0% #ecfdf5');
                cell.style.color = '#065f46';
                cell.style.border = '2pt solid #10b981';
                cell.style.setProperty('mso-border-alt', '2pt solid #10b981');
                cell.style.textAlign = 'center';
              } else if (instructionHeaderStyle === 13) {
                cell.style.backgroundColor = '#facc15';
                cell.style.setProperty('mso-shading', 'windowtext 0% #facc15');
                cell.style.color = '#000000';
                cell.style.border = '3pt solid #000000';
                cell.style.setProperty('mso-border-alt', '3pt solid #000000');
                cell.style.fontWeight = '900';
              } else if (instructionHeaderStyle === 15) {
                cell.style.backgroundColor = '#581c87';
                cell.style.setProperty('mso-shading', 'windowtext 0% #581c87');
                cell.style.color = '#ffffff';
                cell.style.border = '2pt solid #fbbf24';
                cell.style.setProperty('mso-border-alt', '2pt solid #fbbf24');
                cell.style.textAlign = 'center';
              } else if (instructionHeaderStyle === 16) {
                cell.style.backgroundColor = '#14532d';
                cell.style.setProperty('mso-shading', 'windowtext 0% #14532d');
                cell.style.color = '#ffffff';
                cell.style.setProperty('mso-border-left-alt', '10pt solid #14532d');
                cell.style.paddingLeft = '20pt';
              } else if (instructionHeaderStyle === 17) {
                cell.style.backgroundColor = '#0ea5e9';
                cell.style.setProperty('mso-shading', 'windowtext 0% #0ea5e9');
                cell.style.color = '#ffffff';
                cell.style.textAlign = 'center';
              } else if (instructionHeaderStyle === 18) {
                cell.style.border = '2pt dotted #64748b';
                cell.style.setProperty('mso-border-alt', '2pt dotted #64748b');
                cell.style.color = '#475569';
              } else if (instructionHeaderStyle === 19) {
                cell.style.backgroundColor = '#ea580c';
                cell.style.setProperty('mso-shading', 'windowtext 0% #ea580c');
                cell.style.color = '#ffffff';
                cell.style.borderBottom = '4pt solid #9a3412';
                cell.style.setProperty('mso-border-bottom-alt', '4pt solid #9a3412');
                cell.style.fontWeight = '900';
              } else if (instructionHeaderStyle === 20) {
                cell.style.backgroundColor = '#f8fafc';
                cell.style.setProperty('mso-shading', 'windowtext 0% #f8fafc');
                cell.style.color = '#1e293b';
                cell.style.border = '1pt solid #cbd5e1';
                cell.style.setProperty('mso-border-alt', '1pt solid #cbd5e1');
                cell.style.textAlign = 'center';
              } else {
                cell.style.borderBottom = '2.5pt solid #334155';
                cell.style.setProperty('mso-border-bottom-alt', '2.5pt solid #334155');
                cell.style.textAlign = 'center';
              }
            }
          });
        });
      } else {
        table.setAttribute('border', '1');
        table.style.borderCollapse = 'collapse';
        table.style.margin = '0 auto';
        table.style.width = '100%';
      }
      
      // Check if it's a Word Search (many small cells with single letters)
      const cells = table.querySelectorAll('td');
      const isWordSearch = cells.length > 20 && Array.from(cells).every(c => (c.textContent?.trim().length || 0) <= 1);
      
      if (isWordSearch) {
        cells.forEach(c => {
          (c as HTMLElement).style.width = '25pt';
          (c as HTMLElement).style.height = '25pt';
          (c as HTMLElement).style.textAlign = 'center';
          (c as HTMLElement).style.lineHeight = exactLineHeight;
          (c as HTMLElement).style.setProperty('mso-line-height-rule', 'exactly');
        });
      }
    }
  });

  let metadataHtml = "";
  if (metadata) {
    metadataHtml = `
      <div style="margin-bottom: 20pt; border-bottom: 1pt solid #ccc; padding-bottom: 10pt; font-size: 9pt; color: #666; font-family: '${fontFamily}', serif;">
        ${metadata.title ? `<div style="font-size: 14pt; font-weight: bold; color: #000; margin-bottom: 5pt;">${metadata.title}</div>` : ''}
        ${metadata.author ? `<div><strong>Author:</strong> ${metadata.author}</div>` : ''}
        ${metadata.date ? `<div><strong>Date:</strong> ${metadata.date}</div>` : `<div><strong>Exported on:</strong> ${new Date().toLocaleDateString()}</div>`}
      </div>
    `;
  }

  const mTop = '0.4in';
  const mRight = '0.6in';
  const mBottom = '0.4in';
  const mLeft = '0.6in';

  const frameStyle = isFrameEnabled ? 'border: 1.5pt solid black; padding: 10pt;' : '';

  // Apply Paper Style (globalLayout) to the main container
  let paperStyleCss = '';
  let bodyBgColor = '#ffffff';
  let containerStyle = '';
  
  if (globalLayout === 0) { // Clean White
    paperStyleCss = 'background-color: #ffffff; border: 1.5pt solid #f1f5f9; padding: 10pt;';
    bodyBgColor = '#ffffff';
  } else if (globalLayout === 1) { // Orange Mix
    paperStyleCss = 'background-color: #ffffff; border-left: 15pt solid #f97316; padding-left: 15pt;';
    bodyBgColor = '#ffffff';
  } else if (globalLayout === 2) { // Modern Emerald
    paperStyleCss = 'background-color: #f0fdf4; border-left: 15pt solid #059669; padding-left: 15pt;';
    bodyBgColor = '#f0fdf4';
  } else if (globalLayout === 3) { // Soft Lavender
    paperStyleCss = 'background-color: #faf5ff; border-top: 15pt solid #9333ea; padding-top: 15pt;';
    bodyBgColor = '#faf5ff';
  } else if (globalLayout === 4) { // Mint
    paperStyleCss = 'background-color: #f0fdf4; border: 1pt solid #dcfce7;';
    bodyBgColor = '#f0fdf4';
  } else if (globalLayout === 5) { // Peach
    paperStyleCss = 'background-color: #fff7ed; border: 1pt solid #ffedd5;';
    bodyBgColor = '#fff7ed';
  } else if (globalLayout === 6) { // Sky
    paperStyleCss = 'background-color: #f0f9ff; border: 1pt solid #e0f2fe;';
    bodyBgColor = '#f0f9ff';
  } else if (globalLayout === 7) { // Lavender
    paperStyleCss = 'background-color: #f5f3ff; border: 1pt solid #ede9fe;';
    bodyBgColor = '#f5f3ff';
  } else if (globalLayout === 8) { // Citrus
    paperStyleCss = 'background-color: #f0fdf4; border-right: 10pt solid #f97316; padding-right: 15pt;';
    containerStyle += ' mso-border-right-alt: 10pt solid #f97316;';
    bodyBgColor = '#f0fdf4';
  } else if (globalLayout === 9) { // Rose
    paperStyleCss = 'background-color: #fff1f2; border: 6pt solid #fb7185; padding: 15pt;';
    bodyBgColor = '#fff1f2';
  } else if (globalLayout === 10) { // Stars
    paperStyleCss = 'background-color: #ffffff; border: 8pt double #fbbf24; padding: 20pt;';
    bodyBgColor = '#ffffff';
  } else if (globalLayout === 11) { // Flowers
    paperStyleCss = 'background-color: #ffffff; border: 10pt solid #fce7f3; padding: 10pt; border-style: double;';
    bodyBgColor = '#ffffff';
  } else if (globalLayout === 12) { // Hearts
    paperStyleCss = 'background-color: #ffffff; border: 10pt solid #fee2e2; padding: 10pt; border-style: dashed;';
    bodyBgColor = '#ffffff';
  } else if (globalLayout === 13) { // Bubbles
    paperStyleCss = 'background-color: #f0f9ff; border: 10pt solid #e0f2fe; padding: 10pt;';
    bodyBgColor = '#f0f9ff';
  } else if (globalLayout === 14) { // Leaves
    paperStyleCss = 'background-color: #f0fdf4; border: 10pt solid #dcfce7; padding: 10pt;';
    bodyBgColor = '#f0fdf4';
  } else if (globalLayout === 15) { // Rainbow
    paperStyleCss = 'background-color: #fff5f5; border: 2pt solid #ffe3e3;';
    bodyBgColor = '#fff5f5';
  } else if (globalLayout === 16) { // Galaxy
    paperStyleCss = 'background-color: #0f172a; color: #ffffff; border: 2pt solid #1e293b;';
    bodyBgColor = '#0f172a';
  } else if (globalLayout === 17) { // Notebook
    // Red margin line - using a double border to mimic a real notebook margin
    // We use a wider padding and a double line for the "Notebook" feel
    paperStyleCss = 'background-color: #ffffff; border-left: 4.5pt double #ef4444; padding-left: 35pt;';
    bodyBgColor = '#ffffff';
  } else if (globalLayout === 18) { // Vintage
    paperStyleCss = 'background-color: #fef3c7; border: 1pt solid #fde68a;';
    bodyBgColor = '#fef3c7';
  } else if (globalLayout === 19) { // Modern
    paperStyleCss = 'background-color: #f8fafc; border: 2pt solid #e2e8f0;';
    bodyBgColor = '#f8fafc';
  }

  const shadingStyle = `mso-shading: windowtext 0% ${bodyBgColor};`;

  // 4. Structural Layout Enhancements (Borders for Paper Styles)
  containerStyle = `padding: 10pt; min-height: 10in; ${shadingStyle} ${frameStyle} ${paperStyleCss}` + containerStyle;
  
  // Ensure all borders in paperStyleCss have mso-border-alt equivalents
  if (paperStyleCss.includes('border-left')) {
    const match = paperStyleCss.match(/border-left:\s*([^;]+)/);
    if (match) containerStyle += ` mso-border-left-alt: ${match[1]};`;
  }
  if (paperStyleCss.includes('border-right')) {
    const match = paperStyleCss.match(/border-right:\s*([^;]+)/);
    if (match) containerStyle += ` mso-border-right-alt: ${match[1]};`;
  }
  if (paperStyleCss.includes('border-top')) {
    const match = paperStyleCss.match(/border-top:\s*([^;]+)/);
    if (match) containerStyle += ` mso-border-top-alt: ${match[1]};`;
  }
  if (paperStyleCss.includes('border-bottom')) {
    const match = paperStyleCss.match(/border-bottom:\s*([^;]+)/);
    if (match) containerStyle += ` mso-border-bottom-alt: ${match[1]};`;
  }
  if (paperStyleCss.includes('border:') && !paperStyleCss.includes('border-')) {
    const match = paperStyleCss.match(/border:\s*([^;]+)/);
    if (match) containerStyle += ` mso-border-alt: ${match[1]};`;
  }

  // 5. Lined Paper & Notebook Structural Fix (Apply border-bottom to paragraphs)
  // This is NATIVE Word borders, not visuals.
  if (baseLayout === 1 || baseLayout === 3 || baseLayout === 4 || baseLayout >= 5 || globalLayout === 17) {
    const pElements = tempDiv.querySelectorAll('p, div.item, li, td, span, h1, h2, h3');
    pElements.forEach(p => {
      const el = p as HTMLElement;
      // Don't apply to header rows or empty spans
      if (el.closest('.header-row') || el.classList.contains('header-row')) return;
      if (el.tagName === 'SPAN' && !el.textContent?.trim()) return;
      
      // For Notebook/Lined, we want the text to sit "on" the line
      // We use a slightly darker blue for notebook lines if globalLayout is 17 for better visibility in Word
      const lineColor = globalLayout === 17 ? '#93c5fd' : '#cbd5e1';
      const lineWidth = globalLayout === 17 ? '1.0pt' : '0.5pt';
      
      el.style.borderBottom = `${lineWidth} solid ${lineColor}`;
      el.style.paddingBottom = '2pt'; // Reduced to make text sit "on" the line
      el.style.marginBottom = '10pt'; // Adjusted for 24pt line height feel
      el.style.setProperty('mso-border-bottom-alt', `${lineWidth} solid ${lineColor}`);
    });
    
    // Also apply to headerDiv elements
    const headerElements = headerDiv.querySelectorAll('p, h1, h2, h3, div');
    headerElements.forEach(el => {
      const element = el as HTMLElement;
      if (element.classList.contains('header-row')) return;
      
      const lineColor = globalLayout === 17 ? '#93c5fd' : '#cbd5e1';
      const lineWidth = globalLayout === 17 ? '1.0pt' : '0.5pt';
      element.style.borderBottom = `${lineWidth} solid ${lineColor}`;
      element.style.paddingBottom = '2pt';
      element.style.marginBottom = '10pt';
      element.style.setProperty('mso-border-bottom-alt', `${lineWidth} solid ${lineColor}`);
    });
  }

  // 5.5 Decorative Elements for Word Export (Stars, Hearts, etc.)
  if (globalLayout >= 10 && globalLayout <= 14) {
    const symbols: Record<number, string> = { 10: '★', 11: '🌸', 12: '❤', 13: '🫧', 14: '🍃' };
    const colors: Record<number, string> = { 10: '#fcd34d', 11: '#f9a8d4', 12: '#fca5a5', 13: '#bae6fd', 14: '#86efac' };
    const symbol = symbols[globalLayout];
    const color = colors[globalLayout];
    
    // Add decorative symbols to the header corners
    const decoTable = document.createElement('table');
    decoTable.style.width = '100%';
    decoTable.style.marginBottom = '10pt';
    decoTable.innerHTML = `
      <tr>
        <td style="font-size: 24pt; color: ${color}; opacity: 0.5; text-align: left;">${symbol} ${symbol}</td>
        <td style="font-size: 24pt; color: ${color}; opacity: 0.5; text-align: right;">${symbol} ${symbol}</td>
      </tr>
    `;
    headerDiv.insertBefore(decoTable, headerDiv.firstChild);

    // Inject decorative elements more systematically to ensure they appear "on each page"
    const contentNodes = Array.from(tempDiv.children);
    // Insert every 3-4 nodes to simulate per-page presence
    for (let i = 0; i < contentNodes.length; i += 3) {
        const decoDiv = document.createElement('div');
        const align = i % 2 === 0 ? 'left' : 'right';
        const size = 18;
        decoDiv.style.textAlign = align;
        decoDiv.style.fontSize = `${size}pt`;
        decoDiv.style.color = color;
        decoDiv.style.opacity = '0.3';
        decoDiv.style.margin = '10pt 0';
        decoDiv.innerHTML = symbol.repeat(3);
        
        if (contentNodes[i]) {
          tempDiv.insertBefore(decoDiv, contentNodes[i]);
        }
    }
    
    // Also add one at the very bottom
    const bottomDeco = document.createElement('div');
    bottomDeco.style.textAlign = 'center';
    bottomDeco.style.fontSize = '24pt';
    bottomDeco.style.color = color;
    bottomDeco.style.opacity = '0.5';
    bottomDeco.innerHTML = symbol.repeat(5);
    tempDiv.appendChild(bottomDeco);
  }

  // 6. Instruction Ruler Structural Fix - Apply to ALL headers
  if (instructionRulerStyle > 0) {
    const headers = tempDiv.querySelectorAll('.header-row, tr:first-child td[colspan="2"], tr:first-child td[colspan="1"]');
    headers.forEach(header => {
      const rulerDiv = document.createElement('div');
      rulerDiv.style.width = '100%';
      rulerDiv.style.marginTop = '5pt';
      rulerDiv.style.marginBottom = '10pt';
      rulerDiv.style.height = '1pt';
      rulerDiv.style.fontSize = '1pt';
      rulerDiv.innerHTML = '&nbsp;';
      
      if (instructionRulerStyle === 1) {
        rulerDiv.style.borderBottom = '1pt solid #000000';
        rulerDiv.style.setProperty('mso-border-bottom-alt', '1pt solid #000000');
      } else if (instructionRulerStyle === 2) {
        rulerDiv.style.borderBottom = '2pt dashed #000000';
        rulerDiv.style.setProperty('mso-border-bottom-alt', '2pt dashed #000000');
      } else if (instructionRulerStyle === 3) {
        rulerDiv.style.borderBottom = '4pt double #000000';
        rulerDiv.style.setProperty('mso-border-bottom-alt', '4pt double #000000');
      } else if (instructionRulerStyle === 4) {
        rulerDiv.style.borderBottom = '4pt solid #334155';
        rulerDiv.style.setProperty('mso-border-bottom-alt', '4pt solid #334155');
      } else if (instructionRulerStyle === 5) {
        rulerDiv.style.textAlign = 'center';
        rulerDiv.style.color = '#fcd34d';
        rulerDiv.style.fontSize = '14pt';
        rulerDiv.style.height = 'auto';
        rulerDiv.innerHTML = '★ ★ ★';
      } else if (instructionRulerStyle === 6) {
        rulerDiv.style.textAlign = 'center';
        rulerDiv.style.color = '#fca5a5';
        rulerDiv.style.fontSize = '14pt';
        rulerDiv.style.height = 'auto';
        rulerDiv.innerHTML = '♥ ♥ ♥';
      }
      
      // Find the parent row or table to insert after
      const parentTable = header.closest('table');
      if (parentTable) {
        parentTable.parentNode?.insertBefore(rulerDiv, parentTable.nextSibling);
      }
    });
  }

  const content = `
    <html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40">
    <head>
      <meta charset='utf-8'>
      <style>
        @page Section1 { 
          size: 8.5in 11.0in; 
          margin: ${mTop} ${mRight} ${mBottom} ${mLeft}; 
          mso-header-margin: 0.5in; 
          mso-footer-margin: 0.5in; 
          mso-paper-source: 0;
        }
        div.Section1 { 
          page: Section1; 
        }
        body { 
          font-family: "${fontFamily}", serif; 
          font-size: 12pt; 
          line-height: ${exactLineHeight}; 
          mso-line-height-rule: exactly; 
          margin: 0;
          padding: 0;
          background-color: ${bodyBgColor};
        }
        p, div, li, span, ol, ul { 
          margin: 0pt; 
          padding: 0pt; 
          line-height: ${exactLineHeight}; 
          mso-line-height-rule: exactly;
        }
        table { 
          mso-table-lspace:0pt; 
          mso-table-rspace:0pt; 
          border-collapse: collapse; 
          margin: 0; 
          width: 100%;
        }
        td { 
          font-family: "${fontFamily}", serif; 
          font-size: 12pt; 
          line-height: ${exactLineHeight}; 
          mso-line-height-rule: exactly; 
          padding: 0;
          vertical-align: top;
        }
        .header-row { background-color: #334155; color: white; text-align: center; font-weight: bold; padding: 10px; mso-shading: windowtext 0% #334155; }
      </style>
    </head>
    <body>
      <div class="Section1">
        <!-- Master Table for Paper Style Border -->
        <table border="0" cellspacing="0" cellpadding="0" width="100%" style="width: 100%; border-collapse: collapse; ${containerStyle}">
          <tr>
            <td style="padding: 10pt; ${shadingStyle}">
              ${headerDiv.innerHTML}
              ${metadataHtml}
              ${finalHtml}
            </td>
          </tr>
        </table>
      </div>
    </body>
    </html>`;

  const blob = new Blob(['\ufeff', content], { type: 'application/msword' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.doc`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

export const exportToHTML = (htmlContent: string, filename: string, headerHtml: string = '') => {
  const fullHtml = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <title>${filename}</title>
      <style>
        body { font-family: 'Times New Roman', serif; padding: 1in; line-height: 1.5; }
        .header { margin-bottom: 40px; }
        .content { white-space: pre-wrap; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        th, td { border: 1px solid black; padding: 8px; text-align: left; }
        img { max-width: 100%; height: auto; }
      </style>
    </head>
    <body>
      <div class="header">${headerHtml}</div>
      <div class="content">${htmlContent}</div>
    </body>
    </html>
  `;
  const blob = new Blob([fullHtml], { type: 'text/html;charset=utf-8' });
  saveAs(blob, `${filename}.html`);
};

export const exportToPDF = async (elementId: string, filename: string) => {
  const element = document.getElementById(elementId);
  if (!element) return;

  try {
    const dataUrl = await toPng(element, { quality: 0.95, pixelRatio: 2 });
    const pdf = new jsPDF('p', 'mm', 'a4');
    const imgProps = pdf.getImageProperties(dataUrl);
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
    
    pdf.addImage(dataUrl, 'PNG', 0, 0, pdfWidth, pdfHeight);
    pdf.save(`${filename}.pdf`);
  } catch (error) {
    console.error('Error exporting to PDF:', error);
    // Fallback to print
    window.print();
  }
};