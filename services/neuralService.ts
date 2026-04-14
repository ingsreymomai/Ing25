import { GoogleGenAI, GenerateContentResponse, Type } from "@google/genai";
import { NeuralEngine, QuickSource, OutlineItem, ExternalKeys } from "../types";

export interface NeuralResult {
  text: string;
  thought?: string;
  keyUsed?: string;
}

// ==========================================
//  KEY ROTATION ENGINE (Supports 2+ Keys)
// ==========================================
const getGeminiKeys = (userKey?: string): string[] => {
    // 1. If user provided a custom key in UI settings, use only that
    if (userKey && userKey.trim().length > 0) {
        return [userKey.trim()];
    }

    // 2. Look for the comma-separated list from Vercel/Vite
    let envKeys = "";
    try {
        // Support both Vite (Client) and Process (Server/Node) environments
        const metaEnv = (import.meta as any).env;
        envKeys = metaEnv?.VITE_GEMINI_API_KEYS || metaEnv?.GEMINI_API_KEY || "";
        
        if (!envKeys && typeof process !== 'undefined') {
            envKeys = process.env.VITE_GEMINI_API_KEYS || process.env.GEMINI_API_KEY || "";
        }
    } catch (e) {
        console.error("Environment key lookup failed", e);
    }

    // Clean and split the keys into an array
    return envKeys.split(',').map(k => k.trim()).filter(k => k.length > 0);
};

function isQuotaError(error: any): boolean {
    const msg = error?.message?.toLowerCase() || "";
    return msg.includes("quota") || msg.includes("429") || msg.includes("resource_exhausted") || msg.includes("limit");
}

const withRetry = async <T>(
  fn: () => Promise<T>,
  retries: number = 1,
  delay: number = 1500
): Promise<T> => {
  try {
    return await fn();
  } catch (error) {
    if (retries <= 0) throw error;
    await new Promise(resolve => setTimeout(resolve, delay));
    return withRetry(fn, retries - 1, delay * 2);
  }
};

// ==========================================
//  MAIN AI GENERATOR (With Rotation)
// ==========================================
export const callNeuralEngine = async (
  engine: NeuralEngine,
  prompt: string,
  systemInstruction: string,
  file?: QuickSource | null,
  userKeys: ExternalKeys = {}
): Promise<NeuralResult> => {
  
  // FIX: Check if the engine name contains "gemini" (handles all versions: 1.5, 2.0, 3.1, etc.)
  const isGeminiModel = engine.toLowerCase().includes("gemini");

  if (isGeminiModel) {
    const availableKeys = getGeminiKeys(userKeys[engine]);

    if (availableKeys.length === 0) {
      return { 
        text: `<div class="p-6 bg-orange-50 text-orange-700 border border-orange-200 rounded-xl">
                <strong>Configuration Required:</strong> No Gemini API Keys found. 
                Please add <code>VITE_GEMINI_API_KEYS</code> to your Vercel Environment Variables.
               </div>` 
      };
    }

    // Loop through all keys (Rotation logic)
    for (let i = 0; i < availableKeys.length; i++) {
      try {
        return await withRetry(async () => {
          const ai = new GoogleGenAI({ apiKey: availableKeys[i] });
          const parts: any[] = [{ text: prompt }];
          
          if (file) {
            parts.push({ inlineData: { data: file.data, mimeType: file.mimeType } });
          }

          const response: GenerateContentResponse = await ai.models.generateContent({
            model: engine, // This passes the exact string (e.g. "gemini-3.1-flash-lite-preview")
            contents: { parts },
            config: {
              systemInstruction,
              temperature: 0.7,
              topP: 0.95,
              topK: 64
            },
          });

          return {
            text: response.text || "No content generated.",
            thought: `Neural synthesis complete via ${engine} (Key #${i + 1}/${availableKeys.length})`
          };
        });
      } catch (error: any) {
        // If Quota Error and we have more keys, try next key
        if (isQuotaError(error) && i < availableKeys.length - 1) {
          console.warn(`Gemini Key #${i + 1} exhausted (Quota). Rotating to next key...`);
          continue; 
        }
        // If it's the last key or not a quota error, show the specific error
        return { text: `<div class="p-6 bg-red-50 text-red-600 rounded-xl border border-red-200"><strong>Neural Error:</strong> ${error.message}</div>` };
      }
    }
  }

  // ==========================================
  //  NON-GEMINI ENGINES (GPT, Grok, Deepseek)
  // ==========================================
  const userKey = userKeys[engine];
  if (!userKey) {
      return { text: `<div class="p-6 bg-orange-50 text-orange-600 border border-orange-200 rounded-xl">API Key required for ${engine} in Settings.</div>` };
  }

  return withRetry(async () => {
    let endpoint = "";
    if (engine.includes("gpt")) endpoint = "https://api.openai.com/v1/chat/completions";
    else if (engine.includes("grok")) endpoint = "https://api.x.ai/v1/chat/completions";
    else if (engine.includes("deepseek")) endpoint = "https://api.deepseek.com/chat/completions";

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json', 
        'Authorization': `Bearer ${userKey}` 
      },
      body: JSON.stringify({
        model: engine,
        messages: [
            { role: "system", content: systemInstruction }, 
            { role: "user", content: prompt }
        ],
        temperature: 0.7
      })
    });

    if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error?.message || "External API call failed");
    }

    const data = await response.json();
    return { 
        text: data.choices[0].message.content, 
        thought: `External synthesis via ${engine}.` 
    };
  }).catch((error: any) => ({ 
      text: `<div class="p-6 bg-red-50 text-red-600 border border-red-200 rounded-xl"><strong>${engine} Error:</strong> ${error.message}</div>` 
  }));
};

// ==========================================
//  OUTLINE GENERATOR (With Rotation)
// ==========================================
export const generateNeuralOutline = async (
  prompt: string
): Promise<OutlineItem[]> => {
  const availableKeys = getGeminiKeys();

  for (let i = 0; i < availableKeys.length; i++) {
    try {
      const ai = new GoogleGenAI({ apiKey: availableKeys[i] });
      const response = await ai.models.generateContent({
        model: 'gemini-1.5-flash', // Outline is light, always use flash
        contents: prompt,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                title: { type: Type.STRING },
                children: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      title: { type: Type.STRING },
                      children: { 
                          type: Type.ARRAY, 
                          items: { type: Type.OBJECT, properties: { title: { type: Type.STRING } } } 
                      }
                    }
                  }
                }
              },
              required: ["title"]
            }
          }
        }
      });

      const jsonStr = response.text || "[]";
      const data = JSON.parse(jsonStr);
      
      const addIds = (items: any[]): OutlineItem[] => {
        return items.map((item, index) => ({
          id: `outline-${Date.now()}-${index}-${Math.random()}`,
          title: item.title,
          expanded: true,
          children: item.children ? addIds(item.children) : []
        }));
      };

      return addIds(data);
    } catch (error: any) {
      if (isQuotaError(error) && i < availableKeys.length - 1) {
        continue; // Try next key
      }
      console.error(`Outline generation failed:`, error.message);
      return [];
    }
  }
  return [];
};
