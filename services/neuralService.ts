
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
    if (userKey && typeof userKey === 'string' && userKey.trim().length > 0 && !userKey.includes('TODO')) {
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
        console.warn("[Neural] Environment key lookup failed", e);
    }

    // Clean and split the keys into an array
    return envKeys.split(',').map(k => k.trim()).filter(k => k.length > 0 && !k.includes('TODO'));
};

function isQuotaError(error: any): boolean {
    const msg = error?.message?.toLowerCase() || "";
    return msg.includes("quota") || msg.includes("rate limit") || msg.includes("429") || msg.includes("resource_exhausted") || msg.includes("limit") || msg.includes("503") || msg.includes("unavailable") || msg.includes("high demand");
}

const withRetry = async <T>(
  fn: () => Promise<T>,
  retries: number = 2,
  delay: number = 2000
): Promise<T> => {
  try {
    return await fn();
  } catch (error: any) {
    const errorMsg = error?.message?.toLowerCase() || "";
    const isPermissionDenied = errorMsg.includes("permission denied") || errorMsg.includes("403") || errorMsg.includes("unauthorized") || errorMsg.includes("api_key_invalid");
    const isModelNotFound = errorMsg.includes("model not found") || errorMsg.includes("404");
    
    if (isPermissionDenied) {
      console.error("[Neural] Permission Denied (403). API key is invalid or restricted.");
      throw new Error("Gemini API Permission Denied: Your API key may be invalid or restricted. Please check your Engine settings.");
    }

    if (isModelNotFound) {
      console.error("[Neural] Model Not Found (404). The selected model might not be available in your region.");
      throw new Error("Gemini API Model Not Found: The selected engine is not available. Try switching to a different Flash or Pro model in settings.");
    }

    if (retries <= 0) throw error;
    
    console.warn(`[Neural] Retrying in ${delay}ms... (${retries} attempts left). Error: ${error.message}`);
    await new Promise(resolve => setTimeout(resolve, delay));
    return withRetry(fn, retries - 1, delay * 1.5);
  }
};

const TIMEOUT_MS = 60000; // 60 seconds timeout for AI calls

const withTimeout = <T>(promise: Promise<T>, ms: number, message: string): Promise<T> => {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) => setTimeout(() => reject(new Error(message)), ms))
  ]);
};

export const callNeuralEngine = async (
  engine: NeuralEngine,
  prompt: string,
  systemInstruction: string,
  file?: QuickSource | null,
  userKeys: ExternalKeys = {}
): Promise<NeuralResult> => {
  
  const isGemini = 
    engine === NeuralEngine.GEMINI_3_FLASH_LITE || 
    engine === NeuralEngine.GEMINI_3_FLASH || 
    engine === NeuralEngine.GEMINI_3_PRO ||
    (engine as string).toLowerCase().includes('gemini');

  if (isGemini) {
    const availableKeys = getGeminiKeys(userKeys[engine]);

    if (availableKeys.length === 0) {
        return { text: `Error: No Gemini API Keys found. Please add VITE_GEMINI_API_KEYS to your Vercel Environment Variables or provide a key in settings.` };
    }

    let lastError: any = null;
    // Loop through all keys (Rotation logic)
    for (let i = 0; i < availableKeys.length; i++) {
        try {
            const result = await withRetry(async () => {
                let modelName = engine as string;
                // Map legacy and internal names to latest Gemini 3 models as per skill guidelines
                if (modelName.includes('flash-lite') || modelName === 'gemini-1.5-flash-lite') {
                    modelName = 'gemini-3.1-flash-lite-preview';
                } else if (modelName.includes('flash') || modelName === 'gemini-1.5-flash') {
                    modelName = 'gemini-3-flash-preview';
                } else if (modelName.includes('pro') || modelName === 'gemini-1.5-pro') {
                    modelName = 'gemini-3.1-pro-preview';
                }

                try {
                    console.log(`[Neural] Calling Gemini SDK: ${modelName} (Key #${i + 1}/${availableKeys.length})`);
                    const ai = new GoogleGenAI({ apiKey: availableKeys[i] });
                    const parts: any[] = [{ text: prompt }];
                    if (file) {
                        parts.push({ inlineData: { data: file.data, mimeType: file.mimeType } });
                    }

                    const response = await withTimeout(
                        ai.models.generateContent({
                            model: modelName,
                            contents: { parts },
                            config: { systemInstruction, temperature: 0.7, topP: 0.95, topK: 64 },
                        }),
                        TIMEOUT_MS,
                        `Synthesis timed out after ${TIMEOUT_MS/1000}s. The request might be too large or the service is slow.`
                    );

                    if (response.text) return { text: response.text, thought: `Neural synthesis complete via ${modelName} node.` };
                } catch (sdkError: any) {
                    console.warn("[Neural] SDK failed, trying REST fallback...", sdkError.message);
                    
                    // REST Fallback with v1beta and v1 support
                    const restModelName = modelName.startsWith('models/') ? modelName : `models/${modelName}`;
                    const versions = ['v1beta', 'v1'];
                    
                    for (const version of versions) {
                        try {
                            const url = `https://generativelanguage.googleapis.com/${version}/${restModelName}:generateContent?key=${availableKeys[i]}`;
                            const contents: any[] = [{ role: 'user', parts: [{ text: prompt }] }];
                            if (file) contents[0].parts.push({ inlineData: { data: file.data, mimeType: file.mimeType } });

                            const body: any = {
                                contents,
                                generationConfig: { temperature: 0.7, topP: 0.95, topK: 64 }
                            };
                            if (systemInstruction) body.systemInstruction = { parts: [{ text: systemInstruction }] };

                            const fetchResponse = await withTimeout(
                                fetch(url, {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify(body)
                                }),
                                30000, // 30s for REST fallback
                                `REST ${version} timed out.`
                            );

                            if (fetchResponse.ok) {
                                const data = await fetchResponse.json();
                                const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
                                if (text) return { text, thought: `Neural synthesis complete via Direct REST (${version}) Fallback.` };
                            } else {
                                const errorData = await fetchResponse.json().catch(() => ({}));
                                console.warn(`[Neural] REST ${version} error:`, fetchResponse.status, errorData);
                                if (fetchResponse.status === 429 || fetchResponse.status === 503) {
                                    throw new Error(`REST API Error: ${fetchResponse.status} ${errorData.error?.message || ''}`);
                                }
                            }
                        } catch (e) {
                            console.warn(`[Neural] REST ${version} failed:`, e);
                            throw e; // Bubble up to outer try/catch
                        }
                    }
                    throw sdkError;
                }
                throw new Error("No content generated.");
            });
            return result;
        } catch (error: any) {
            lastError = error;
            if (isQuotaError(error) && i < availableKeys.length - 1) {
                console.warn(`Gemini Key #${i + 1} exhausted (Quota/503). Rotating to next key...`);
                continue; 
            }
            // If it's the last key or not a quota error, return the specific error
            if (i === availableKeys.length - 1) {
                return { text: `Error: ${error.message || "Synthesis failed"}` };
            }
        }
    }
    return { text: `Error: ${lastError?.message || "Synthesis failed"}` };
  }

  const userKey = userKeys[engine];
  if (!userKey) return { text: `Error: Key required for ${engine}` };

  return withRetry(async () => {
    let endpoint = "";
    if (engine === NeuralEngine.GPT_4O) endpoint = "https://api.openai.com/v1/chat/completions";
    else if (engine === NeuralEngine.GROK_3) endpoint = "https://api.x.ai/v1/chat/completions";
    else if (engine === NeuralEngine.DEEPSEEK_V3) endpoint = "https://api.deepseek.com/chat/completions";

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${userKey}` },
      body: JSON.stringify({
        model: engine,
        messages: [{ role: "system", content: systemInstruction }, { role: "user", content: prompt }],
        temperature: 0.7
      })
    });

    const data = await response.json();
    return { text: data.choices[0].message.content, thought: `External synthesis via ${engine}.` };
  }).catch((error: any) => ({ text: `Error: ${error.message}` }));
};

// Fixed initialization and property access for generateNeuralOutline.
export const generateNeuralOutline = async (
  prompt: string
): Promise<OutlineItem[]> => {
  const addIds = (items: any[]): OutlineItem[] => {
    return items.map((item, index) => ({
      id: `outline-${Date.now()}-${index}-${Math.random()}`,
      title: item.title,
      expanded: true,
      children: item.children ? addIds(item.children) : []
    }));
  };

  const availableKeys = getGeminiKeys();
  if (availableKeys.length === 0) {
    console.warn("No API key available for outline generation.");
    return [];
  }

  for (let i = 0; i < availableKeys.length; i++) {
    try {
      // Attempt 1: SDK
      const ai = new GoogleGenAI({ apiKey: availableKeys[i] });
      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
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
                      children: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { title: { type: Type.STRING } } } }
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
      return addIds(data);
    } catch (sdkError: any) {
      console.warn(`Outline SDK failed (Key #${i + 1}), falling back to REST...`, sdkError.message);
      
      try {
        // Attempt 2: REST Fallback
        const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key=${availableKeys[i]}`;
        const body = {
          contents: [{ role: 'user', parts: [{ text: prompt }] }],
          generationConfig: {
            responseMimeType: "application/json",
            temperature: 0.1
          }
        };

        const fetchResponse = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });

        if (!fetchResponse.ok) {
            const errorData = await fetchResponse.json().catch(() => ({}));
            throw new Error(`REST Outline failed: ${fetchResponse.status} ${errorData.error?.message || ''}`);
        }
        
        const data = await fetchResponse.json();
        const jsonStr = data.candidates?.[0]?.content?.parts?.[0]?.text || "[]";
        return addIds(JSON.parse(jsonStr));
      } catch (restError: any) {
        if (isQuotaError(restError) && i < availableKeys.length - 1) {
            console.warn(`Gemini Key #${i + 1} exhausted for outline. Rotating...`);
            continue;
        }
        console.error(`Outline generation failed completely.`, restError.message);
        if (i === availableKeys.length - 1) {
            return [];
        }
      }
    }
  }
  return [];
};
