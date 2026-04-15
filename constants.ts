import { Theme, StrictRule, AcademicLevel, InstructionTemplate } from './types';

export const INITIAL_MODULES = ['Grammar', 'Reading', 'Vocabulary', 'Mixed', 'Generals'];

export const LANGUAGES = ['English', 'Khmer', 'Chinese', 'Korean', 'French'];

export const ACADEMIC_LEVELS: AcademicLevel[] = [
  'Kid', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 
  'Level 6', 'Level 7', 'Level 8', 'Level 9', 'Level 10', 'Level 11', 'TOEFL', 'IELTS'
];

export const THEMES: Theme[] = [
  { id: 'default', name: 'Academic Classic', color: '#ea580c', bg: '#ffffff', accent: '#f97316' },
  { id: 'modern', name: 'Modern Professional', color: '#0f172a', bg: '#ffffff', accent: '#334155' },
  { id: 'royal', name: 'Royal Blueprint', color: '#1e3a8a', bg: '#f8fafc', accent: '#3b82f6' },
  { id: 'forest', name: 'Forest Scholar', color: '#064e3b', bg: '#f0fdf4', accent: '#10b981' },
  { id: 'crimson', name: 'Crimson Archive', color: '#7f1d1d', bg: '#fef2f2', accent: '#ef4444' },
  { id: 'midnight', name: 'Midnight Architect', color: '#1e293b', bg: '#0f172a', accent: '#6366f1' },
  { id: 'beach', name: 'Tropical Beach', color: '#0284c7', bg: 'linear-gradient(to bottom, #bae6fd, #fef3c7)', accent: '#0ea5e9' },
  { id: 'sunset', name: 'Sunset Horizon', color: '#9d174d', bg: 'linear-gradient(to top right, #fdf2f8, #fff7ed)', accent: '#db2777' },
  { id: 'nebula', name: 'Deep Nebula', color: '#7c3aed', bg: 'radial-gradient(circle at center, #2e1065, #0f172a)', accent: '#8b5cf6' },
  { id: 'zen', name: 'Zen Garden', color: '#4d7c0f', bg: '#f7fee7', accent: '#65a30d' },
];

export const FONTS = [
  { name: 'Garamond', family: "'EB Garamond', serif" },
  { name: 'Times New Roman', family: "'Times New Roman', serif" },
  { name: 'Georgia', family: "Georgia, serif" },
  { name: 'Playfair Display', family: "'Playfair Display', serif" },
  { name: 'Merriweather', family: "Merriweather, serif" },
  { name: 'Lora', family: "Lora, serif" },
  { name: 'Arial', family: "Arial, sans-serif" },
  { name: 'Helvetica', family: "Helvetica, sans-serif" },
  { name: 'Inter', family: "Inter, sans-serif" },
  { name: 'Roboto', family: "Roboto, sans-serif" },
  { name: 'Open Sans', family: "'Open Sans', sans-serif" },
  { name: 'Montserrat', family: "Montserrat, sans-serif" },
  { name: 'Poppins', family: "Poppins, sans-serif" },
  { name: 'Oswald', family: "Oswald, sans-serif" },
  { name: 'Courier New', family: "'Courier New', Courier, monospace" },
  { name: 'JetBrains Mono', family: "'JetBrains Mono', monospace" },
  { name: 'Pacifico', family: "Pacifico, cursive" },
  { name: 'Dancing Script', family: "'Dancing Script', cursive" },
  { name: 'Great Vibes', family: "'Great Vibes', cursive" },
  { name: 'Cinzel', family: "Cinzel, serif" },
];

export const PAPER_DESIGNS = [
  '', // 0: Standard
  'design-modern-blue', // 1
  'design-classic', // 2
  'design-minimalist', // 3
  'design-playful', // 4
  'design-professional', // 5
  'design-elegant', // 6
  'design-technical', // 7
  'design-eco', // 8
  'design-contrast', // 9
  'design-two-fold', // 10
  'design-projector', // 11
  'design-modern-round', // 12
  'design-bold-red', // 13
  'design-royal-gold', // 14
  'design-deep-ocean', // 15
  'design-sunset-vibrant', // 16
  'design-cyberpunk', // 17
  'design-academic-heavy', // 18
  'design-art-deco', // 19
  'design-futuristic', // 20
  'design-col-table-1', // 21
  'design-col-table-2', // 22
  'design-col-table-3', // 23
  'design-col-table-4', // 24
  'design-col-table-5', // 25
  'design-col-table-6', // 26
  'design-col-table-7', // 27
  'design-col-table-8', // 28
  'design-col-table-9', // 29
  'design-col-table-10', // 30
];

export const SUBJECTS = [
  {
    id: 'cambodia',
    name: 'Cambodia',
    names: ['Tevi', 'Vuthy', 'Sreypov', 'Dara', 'Sokha', 'Chann', 'Bopha', 'Piseth', 'Rithy', 'Leakhena'],
    places: ['Angkor Wat', 'Phnom Penh', 'Siem Reap', 'Battambang', 'Sihanoukville', 'Kampot', 'Kep', 'Preah Vihear', 'Tonle Sap', 'Mekong River']
  },
  {
    id: 'usa',
    name: 'USA',
    names: ['James', 'Mary', 'Robert', 'Patricia', 'John', 'Jennifer', 'Michael', 'Linda', 'David', 'Elizabeth'],
    places: ['New York City', 'Grand Canyon', 'Statue of Liberty', 'White House', 'Golden Gate Bridge', 'Yellowstone', 'Disney World', 'Las Vegas', 'Hollywood', 'Mount Rushmore']
  },
  {
    id: 'china',
    name: 'China',
    names: ['Lie Bui', 'Wei Chen', 'Li Wang', 'Zhang Min', 'Liu Yang', 'Chen Jing', 'Yang Bo', 'Zhao Lei', 'Huang Yan', 'Zhou Tao'],
    places: ['Great Wall', 'Forbidden City', 'Shanghai Bund', 'Terracotta Army', 'West Lake', 'Potala Palace', 'Li River', 'Yellow Mountains', 'Zhangjiajie', 'Jiuzhaigou']
  },
  {
    id: 'korea',
    name: 'South Korea',
    names: ['Kim Min-jun', 'Lee Seo-yeon', 'Park Ji-hoon', 'Choi Ha-eun', 'Jung Do-yoon', 'Kang Ji-woo', 'Jo Hyun-woo', 'Yoon Seo-ah', 'Jang Min-ho', 'Lim Da-in'],
    places: ['Gyeongbokgung Palace', 'N Seoul Tower', 'Jeju Island', 'Bukchon Hanok Village', 'Haeundae Beach', 'Seoraksan National Park', 'DMZ', 'Bulguksa Temple', 'Lotte World', 'Dongdaemun Design Plaza']
  },
  {
    id: 'japan',
    name: 'Japan',
    names: ['Haruto', 'Yua', 'Itsuki', 'Akari', 'Minato', 'Koharu', 'Ren', 'Himari', 'Yuma', 'Mei'],
    places: ['Mount Fuji', 'Kyoto Temples', 'Tokyo Tower', 'Osaka Castle', 'Nara Park', 'Hiroshima Memorial', 'Hokkaido', 'Shibuya Crossing', 'Itsukushima Shrine', 'Okinawa']
  },
  {
    id: 'france',
    name: 'France',
    names: ['Lucas', 'Emma', 'Gabriel', 'Jade', 'Louis', 'Louise', 'Raphaël', 'Alice', 'Arthur', 'Chloé'],
    places: ['Eiffel Tower', 'Louvre Museum', 'Notre Dame', 'Versailles', 'Mont Saint-Michel', 'French Riviera', 'Loire Valley', 'Provence', 'Bordeaux', 'Chamonix']
  },
  {
    id: 'uk',
    name: 'UK',
    names: ['Oliver', 'Olivia', 'George', 'Amelia', 'Harry', 'Isla', 'Noah', 'Ava', 'Jack', 'Mia'],
    places: ['Big Ben', 'Stonehenge', 'London Eye', 'Buckingham Palace', 'Edinburgh Castle', 'Giant\'s Causeway', 'Lake District', 'Oxford University', 'Cambridge', 'Bath']
  },
  {
    id: 'australia',
    name: 'Australia',
    names: ['Oliver', 'Charlotte', 'William', 'Olivia', 'Jack', 'Amelia', 'Noah', 'Isla', 'Thomas', 'Mia'],
    places: ['Sydney Opera House', 'Great Barrier Reef', 'Uluru', 'Bondi Beach', 'Great Ocean Road', 'Blue Mountains', 'Kakadu', 'Whitsundays', 'Daintree Rainforest', 'Twelve Apostles']
  },
  {
    id: 'canada',
    name: 'Canada',
    names: ['Liam', 'Olivia', 'Noah', 'Emma', 'Lucas', 'Charlotte', 'Oliver', 'Amelia', 'Benjamin', 'Sophia'],
    places: ['Niagara Falls', 'Banff', 'CN Tower', 'Old Quebec', 'Whistler', 'Stanley Park', 'Peggy\'s Cove', 'Jasper', 'Butchart Gardens', 'Bay of Fundy']
  },
  {
    id: 'germany',
    name: 'Germany',
    names: ['Noah', 'Mia', 'Leon', 'Emma', 'Paul', 'Sophia', 'Lukas', 'Hannah', 'Jonas', 'Emilia'],
    places: ['Neuschwanstein Castle', 'Brandenburg Gate', 'Cologne Cathedral', 'Black Forest', 'Zugspitze', 'Heidelberg', 'Rhine Valley', 'Rothenburg', 'Sanssouci', 'Saxon Switzerland']
  },
  {
    id: 'brazil',
    name: 'Brazil',
    names: ['Miguel', 'Alice', 'Arthur', 'Laura', 'Heitor', 'Valentina', 'Bernardo', 'Helena', 'Davi', 'Sophia'],
    places: ['Christ the Redeemer', 'Amazon Rainforest', 'Iguazu Falls', 'Copacabana', 'Sugarloaf Mountain', 'Pelourinho', 'Lençóis Maranhenses', 'Pantanal', 'Fernando de Noronha', 'Ibirapuera Park']
  },
  {
    id: 'india',
    name: 'India',
    names: ['Aarav', 'Aditi', 'Vihaan', 'Diya', 'Arjun', 'Ananya', 'Sai', 'Kavya', 'Krishna', 'Neha'],
    places: ['Taj Mahal', 'Jaipur', 'Varanasi', 'Kerala Backwaters', 'Goa Beaches', 'Golden Temple', 'Hampi', 'Mysore Palace', 'Red Fort', 'Gateway of India']
  },
  {
    id: 'italy',
    name: 'Italy',
    names: ['Leonardo', 'Sofia', 'Francesco', 'Giulia', 'Alessandro', 'Aurora', 'Lorenzo', 'Ginevra', 'Mattia', 'Alice'],
    places: ['Colosseum', 'Venice Canals', 'Leaning Tower of Pisa', 'Vatican City', 'Pompeii', 'Amalfi Coast', 'Cinque Terre', 'Florence Duomo', 'Lake Como', 'Pantheon']
  },
  {
    id: 'mexico',
    name: 'Mexico',
    names: ['Santiago', 'Sofía', 'Mateo', 'María José', 'Sebastián', 'Valentina', 'Leonardo', 'Ximena', 'Matías', 'Camila'],
    places: ['Chichén Itzá', 'Teotihuacan', 'Cancún', 'Tulum', 'Palenque', 'Copper Canyon', 'Oaxaca', 'Cabo San Lucas', 'Guanajuato', 'Zócalo']
  },
  {
    id: 'spain',
    name: 'Spain',
    names: ['Hugo', 'Lucía', 'Martín', 'Sofía', 'Lucas', 'Martina', 'Mateo', 'María', 'Leo', 'Julia'],
    places: ['Sagrada Familia', 'Alhambra', 'Park Güell', 'Prado Museum', 'Ibiza', 'Plaza Mayor', 'Mosque of Córdoba', 'Casa Batlló', 'Mount Teide', 'Guggenheim Bilbao']
  }
];

export const GLOBAL_STRICT_COMMAND = `### DPSS ULTIMATE TEST BUILDER: ELITE PROTOCOL ###
Enforce situational logic via prioritized rules.

--- 🔠 CASE SENSITIVITY (CRITICAL) ---
0. [INSTRUCTION CASE]: You MUST follow the specific case instruction provided (ALL CAPS or all small letters) for ALL instructions, introductions, and headers. This is the highest priority for formatting.

--- 🧠 COGNITIVE INTEGRITY (EXTREMELY STRICT - HIGH PRIORITY) ---
0. [CONTEXT MANDATORY]: Every single question MUST have a clear context sentence. You are FORBIDDEN from generating questions that are just blanks without a surrounding sentence.
1. [NEAR-MISS & PRAGMATIC LOGIC]: Every MCQ/ Correct & Incorrect/ sometimes circle must have 1 contextually inferior "Near-Miss" distractor (Sometimes grammatically correct but contextually wrong).
    - [OPINION VS OBLIGATION]: Distinguish between internal opinion (must) and external rules (have to).
    - Example MCQ: "This soup tastes terrible. I think you ____ more salt next time." (A. have to add [Incorrect - not a rule], B. had to add, C. must add [Correct - opinion], D. has to add)
    - Example Correct/Incorrect: "Angkor Wat is beautiful. You have to visit it." [Incorrect - external obligation], "Angkor Wat is beautiful. You must visit it." [Correct - opinion/recommendation]
2. [TARGET GRAMMAR FOCUS]: Students should choose the correct answer based on the target grammar rule, not by eliminating unrelated grammar forms.
    - Example: "You ____ wear a helmet here." (A. must wear, B. have to wear [Correct if it is a rule], C. must have, D. had to wear)
3. [NO-FREE-VERB RULE]:
   - In multiple-choice grammar questions, NEVER place the main verb in the question stem if it is part of the structure being tested. The verb MUST be in the options.
   - MANDATORY: You MUST still provide a full context sentence. Only the specific target structure and its main verb should be replaced by a blank.
   - WRONG: "1. ______, ______."
   - CORRECT: "The library is quiet. We ______ loudly." (A. have to talk, B. must talk, C. must to talk, D. has to talk)
   - This forces students to process the full modal + verb structure.
4. [SITUATIONAL-EVIDENCE]: Grammar must be inferred from context/evidence, not obvious time markers.
    - Example: "Her notebook is closed. She ____ her homework." (Student must infer completion).
5. [PURE VOCABULARY CONTROL]:
   - In vocabulary sections, all answer choices must be the same part of speech and grammatical form.
   - If testing "exhausted," all options must be adjectives: tired, sleepy, bored, exhausted.
6. [READING & VOCABULARY GRAMMAR BLACKOUT]:
   - In Reading and Vocabulary sections, you are FORBIDDEN from testing grammar.
   - All distractors must be grammatically identical to the correct answer. 
   - The ONLY differentiator must be the meaning (semantics) or factual content.
7. [TRUE/FALSE FORMAT STRICTNESS]:
   - For True/False, the format MUST be exactly: "1. ______ [Statement]".
   - The answer key MUST be T, F, or NG.

--- ⚙️ STRUCTURAL & POSITIONAL CONTROL ---
8. [ITEM-SEPARATION]: Every numbered item MUST start on a NEW LINE using an HTML <p> or <br> tag.
9. [ANSWER KEY ENTROPY - BUCKET RANDOMIZATION]:
   - BUCKET METHOD: For every 10 items, you MUST pre-select a "Bucket" of 10 letters (e.g., 3 A's, 2 B's, 2 C's, 3 D's).
   - MANDATORY PRESENCE: Every letter (A, B, C, D) MUST appear at least once in every 10-item set.
   - [ANSWER KEY FORMAT]: The Answer Key MUST match the exercise type. 
       - For MCQ: Show letters (A, B, C, D).
       - For T/F: Show T, F, or NG.
       - For Short Answer/Writing: Show the actual word or phrase. NEVER show ABCD for writing-based answers.
10. [THE-SHUFFLE]: Randomize the order of your bucket so there is NO predictable pattern.
   - ANSWER-FIRST RULE: Write the final shuffled answer key at the very top of your internal scratchpad.
   - STREAK LIMIT: Max 2 identical answers in a row.
11. [FLOATING-MARKER]: Vary sentence structure so students cannot scan mechanically.
12. [SYNTACTIC-DISTANCE]: (Level 5+) Separate subject from verb using relative clauses/prepositional phrases.
13. [ADVANCED-COMP]: Test "as good a student as", "of the two", "the more..., the more...", "not so much A as B".

--- 🎨 LAYOUT & VISUALS ---
14. [SEPARATE-TABLES]: Use a separate HTML <table> for each PART.
15. [HEADER-STYLE]: Header row: Bold, Not Centered, White text, Randomized color background.
16. [MCQ-FORMAT]: 
    - Options MUST start on a new line below the question stem.
    - [DYNAMIC LAYOUT]: Follow the [MCQ LAYOUT - MANDATORY] instruction provided at the end of the prompt.
    - INDENTATION: Put 6 non-breaking spaces (&nbsp;) before "A." in the first cell.
    - WORD EXPORT COMPATIBILITY: Use plain letters (A., B., C., D.) without circles for maximum compatibility with MS Word.

--- 🎭 SCENARIO & CONTENT ---
17. [SCENARIO-CHAOS]: Use unique, vivid scenarios. Forbidden from repeating themes.
18. [ANTI-ROBOT]: Randomize all subjects and lead-ins. Use a wide variety of subjects (e.g., "The committee", "A stray cat", "The researchers").
19. [TOPIC-OVERRIDE]: Topic box overrides template defaults.
20. [WORD-FORM-SHIFT]: Reading questions must not repeat exact wording from text (Paraphrase!).
21. [CROSS-ITEM FIREWALL]: Ensure that the answer to a question in Part A is not revealed by a sentence in Part B.
22. [NATURALIZE]: Replace 20% of formal verbs with phrasal verbs.
23. [LEVEL-BASED ARCHITECTURAL SCALING]: Complexity MUST strictly scale with {{LEVEL}}.

25. [UNIVERSAL SITUATIONAL & POSITIONAL LOGIC]:
   - You MUST apply situational nuance and word-position rules to ALL grammar types.
   - all distractors must be grammatically correct most of the times. For examples, Angkor Wat is beautiful. You have to visit it. (Incorrect). Angkor Wat is beautiful. You must visit it. (Correct): Use this type of exercises more. For MCQ, try to use more: Students can think more about opinion and rule/ obligation. This soup tastes terrible. I think you ____ more salt next time.
A. have to add          B. had to add          C. must add          D. has to add
  
   - POSITION RULES: Test tricky word orders. 
     - Adjective Comparison: 
       - "as good a student as" (Singular) vs "as good students as" (Plural).
       - "Of the two" Rule: "Of the two students, he is the taller" (NOT tallest).
       - "Of all" Rule: "Of all the teachers, she is the most hardworking" (Superlative).
       - Comparative vs Superlative Traps: "He is more tall than..." (WRONG) vs "He is taller than..." (RIGHT).
       - MANDATORY VARIETY: If testing Adjectives, you MUST rotate between at least 4 different comparison structures (as...as, comparative -er, superlative -est, of the two).
     - Adverb Placement: Test frequency adverbs (e.g., "He always is" vs "He is always") and manner adverbs.
     - Verb-Object Integrity: Test that direct objects are not separated from verbs (e.g., "I like very much coffee" is INCORRECT; "I like coffee very much" is CORRECT).
     - Conjunction Scrambling: Test correlative conjunctions and inversion (e.g., "Not only he is" vs "Not only is he").
     - RARE PREPOSITIONAL LOGIC: Test tricky prepositional boundaries (e.g., "In the end" vs "At the end", "Good at" vs "Good in", "Arrive in" vs "Arrive at").
   - SITUATIONAL NUANCE: Test meaning-based differences (e.g., Must vs Have To, Will vs Going To, Say vs Tell).
   - CHALLENGE: Distractors must be grammatically valid in isolation but "Positionally" or "Situationally" incorrect in context.
26. [INFINITE SCENARIO VARIETY]:
   - You are strictly FORBIDDEN from repeating scenarios or sentence structures across different generations.
   - Every test must be a completely fresh set of characters, locations, and situations.
   - Randomize numbers, names, and subjects completely. No "robot patterns".
27. [ANTI-ROBOT SENTENCE STARTERS]:
   - You are strictly FORBIDDEN from using repetitive sentence starters. 
   - DO NOT start Item 1 with "I think" or "He is" in every generation.
   - Shuffle all subjects (e.g., "The chef", "A lonely astronaut", "My stubborn cat").
28. [FLOATING MARKER PRINCIPLE]:
   - Do not place key grammar signals in the same position every time. Vary sentence structure so students cannot scan mechanically.
   - Example variations: "Of the two students, he is the taller." vs "He is the taller of the two students."
   - This forces full-sentence processing.
29. [SYNTACTIC DISTANCE STRATEGY (HIGHER LEVELS)]:
   - At advanced levels, separate the subject from the main verb using relative clauses or prepositional phrases.
   - Example: "The teacher who lives near the large blue house by the river is very kind."

--- 📐 DESIGN TEST STYLE (ABSOLUTE PRIORITY) ---
30. [DESIGN OVERRIDE]: The specific "Design Test Style" and "Format Design" instructions provided at the end of the prompt are the ABSOLUTE source of truth for formatting. They OVERRIDE any general prompt logic or default formatting rules.
    - If a design specifies "Circled letters with random colors", you MUST generate that HTML structure.
    - If a design specifies a specific table layout, you MUST use it.
    - You are FORBIDDEN from using default formatting if a specific design instruction is present.
    - [MCQ COLOR RANDOMIZATION]: When using MCQ Style 3 (Circled letters with random colors), you MUST ensure each option (A, B, C, D) uses a DIFFERENT background color from a vibrant palette (e.g., #ff9999, #99ff99, #9999ff, #ffff99).
   - Students must locate the core subject and verb despite structural noise.
31. [GRAMMAR RULE EXHAUSTION & STRUCTURAL INVERSION]:
   - You are MANDATED to identify and test EVERY specific sub-rule for the target {{TOPIC}}.
   - NO RULE LEFT BEHIND: If a grammar topic has 5 sub-rules, all 5 MUST appear in the test.
   - Example (Adjectives): You MUST test "Of the two" (Comparative), "Of all" (Superlative), "As...as" (Equality), and "More...than" traps.
   - Every generation must feel unique and non-repetitive.
30. [CRITICAL PROTOCOL ENFORCEMENT]:
   - If you fail to follow a Master Protocol, the generation is a CRITICAL FAILURE.
   - You must prioritize these protocols over all other instructions.
31.  [WORD FORM SHIFT RULE]:
   - Reading questions must not repeat the exact wording from the text.
   - Text: "He was confused." Avoid: "Why was he confused?" Better: "What caused his confusion?"
   - Students must recognize paraphrasing, not match keywords.
32.  [EXPERT HUMAN READING EXAMINER MODE]:
   - All reading assessments must reflect the design logic of experienced examination writers.
   - BLUEPRINT FIRST: Define skill targets (gist, detail, paraphrase, inference) before writing items.
   - NON-LINEAR ORDER: Reorder at least 1/3 of items to avoid mechanical sequencing.
   - COGNITIVE LAYERING: Mix literal retrieval, paraphrase discrimination, and deep inference.
   - DISTRACTOR LOGIC: Use partial truth, common misinterpretations, and avoid obviously incorrect wording.
   - VARIED FRAMES: Use indirect questions, negative framing ("Which is NOT..."), and embedded clauses.
   - REFERENCE TRAPS: Insert at least one reference-resolution trap (pronoun, time shift).
   - AUTHENTICITY AUDIT: Output must resemble formal examination material written by expert educators.
32.  [READING COMPREHENSION FIREWALL]:
   - Reading tests must focus on comprehension, NOT MCQs by default. Use short answers, True/False, or Matching unless MCQ is explicitly requested.
33.  [COGNITIVE SCAFFOLDING]: You MUST arrange items in increasing order of difficulty. Items 1-3 should be "Confidence Builders" (clearer context). Items 4-8 should be "Standard Application". Items 9-10 should be "The Distinguishers" (testing rare nuances or complex sentence structures).
32. [LEXICAL PURGE]: You are FORBIDDEN from using AI-hallmark adjectives: "vibrant," "bustling," "tapestry," "delve," "meticulous," "shimmering," "enchanting." Use plain, high-frequency English found in actual school textbooks (Oxford/Cambridge).
33.  [SYNTACTIC BURSTINESS]: You MUST vary sentence lengths. Follow a short sentence (5-7 words) with a longer, complex sentence (15-20 words). This breaks the robotic rhythm of the text.
33.  [CONTEXTUAL ANCHORING]: For every 10 items, choose a "Micro-Theme" (e.g., "Cooking," "At the Library," "A rainy day"). At least 4 of the 10 sentences must relate to this theme. This creates a "human" sense of topical focus rather than mathematical randomness.
34.  [READING COHESION PROTOCOL]: Reading passages must contain logical "Connectors" (However, Consequently, Similarly, In contrast). Questions must occasionally test these logical shifts, not just nouns/verbs.
34.  [LEXICAL OVERLAP TRAP]: At least one distractor in Reading MCQs MUST use words that appear in the text but describe a different situation. This punishes students who simply "word-match" without reading.
35.  [THE GIST & DETAIL BALANCE]: Every reading test must include exactly one "Global" question (e.g., "What is the main purpose of this text?") and several "Local" questions (specific details).
36.  [GRAMMAR FOCUS RULE]: Distractors must belong to the same grammar system as the target structure unless the task specifically tests meaning differences between grammar systems.
37.  [TOPIC CONSISTENCY RULE (GLOBAL)]: All MCQ distractors should come from the same grammar system as the target structure. The goal is to test the specific grammar topic, not unrelated grammar areas. At least one distractor must be a "near-miss":
   a grammatically correct option that is slightly wrong in meaning or usage.
Examples of grammar systems:
- Must / Have to
- Present Perfect
- Conditionals
- Comparatives
- Articles
- Prepositions
- Passive Voice
Students should choose the correct answer based on the target grammar rule, not by eliminating unrelated grammar forms. For examples, 
You ____ wear a helmet here.
A. must wear          B. have to wear          C. must have          D. had to
38. [MCQ SPACING EXCEPTION]:
   - For Multiple Choice Questions (MCQ) ONLY: Insert exactly one empty line (one <br> or one empty <div>) BEFORE each new MCQ item to improve readability. This is an exception to the density rules.


### PRIORITY: COGNITIVE INTEGRITY RULES ARE ABSOLUTE. ###
`;

export const PART_BACKGROUND_INSTRUCTION = `### PART BACKGROUND PROTOCOL ###
For each PART (A, B, C, etc.), apply a unique relaxing background style.
- If the part is in a <table>, apply the class to the <table> tag.
- If the part is a list, you MUST wrap the entire part (header + items) in a <div class="...">.
- MANDATORY: Rotate between these classes for each part to ensure variety:
1. .bg-relax-blue (Light Blue)
2. .bg-relax-green (Soft Green)
3. .bg-relax-yellow (Pale Yellow)
4. .bg-relax-purple (Lavender)
5. .bg-relax-rose (Rose)
6. .bg-relax-ocean (Ocean Gradient)
7. .bg-relax-mist (Mist Gradient)
8. .bg-relax-forest (Forest Gradient)
9. No background (transparent)

STRICT: Ensure text remains highly legible. Use these classes ONLY if isPartBackgroundEnabled is true.`;

export const INSTRUCTION_HEADER_BACKGROUND_INSTRUCTION = `### INSTRUCTION HEADER BACKGROUND PROTOCOL ###
For each PART (A, B, C, etc.), apply a unique background style ONLY to the instruction header row (the first row of the table).
Rotate between these styles for the header row:
1. Soft Blue: background-color: #dbeafe; color: #1e3a8a; font-weight: 700;
2. Soft Green: background-color: #dcfce7; color: #064e3b; font-weight: 700;
3. Soft Red: background-color: #fee2e2; color: #7f1d1d; font-weight: 700;
4. Soft Purple: background-color: #f3e8ff; color: #4c1d95; font-weight: 700;
5. Soft Slate: background-color: #f1f5f9; color: #1e293b; font-weight: 700;
6. Soft Emerald: background-color: #ecfdf5; color: #065f46; font-weight: 700;
7. Soft Indigo: background-color: #e0e7ff; color: #3730a3; font-weight: 700;
8. Soft Orange: background-color: #ffedd5; color: #9a3412; font-weight: 700;
9. Soft Cyan: background-color: #ecfeff; color: #0e7490; font-weight: 700;
10. Soft Rose: background-color: #fff1f2; color: #9f1239; font-weight: 700;

STRICT: When this protocol is active, the header row MUST use these lighter, high-legibility backgrounds with dark text. The text MUST be bold and clear.`;

export const PAGE_STYLES = [
  { id: 'p1', name: 'Elegant Gold', style: 'border: 15px solid transparent; border-image: url("https://www.transparenttextures.com/patterns/gold-scale.png") 30 round; padding: 25px; box-shadow: inset 0 0 10px rgba(0,0,0,0.1);' },
  { id: 'p2', name: 'Classic Scroll', style: 'border: 2px solid #8b4513; padding: 30px; background-color: #fdf5e6; border-radius: 5px; box-shadow: 5px 5px 15px rgba(0,0,0,0.2);' },
  { id: 'p3', name: 'Modern Blueprint', style: 'border: 1px solid #3b82f6; padding: 20px; background-image: radial-gradient(#3b82f6 0.5px, transparent 0.5px); background-size: 20px 20px; border-radius: 8px;' },
  { id: 'p4', name: 'Nature Leaf', style: 'border: 10px solid #10b981; border-style: double; padding: 20px; border-radius: 50px 5px 50px 5px;' },
  { id: 'p5', name: 'Royal Velvet', style: 'border: 8px solid #7f1d1d; outline: 2px solid #facc15; outline-offset: -5px; padding: 25px;' },
  { id: 'p6', name: 'Tech Grid', style: 'border: 2px solid #6366f1; padding: 20px; background: linear-gradient(90deg, #f8fafc 20px, transparent 1%) center, linear-gradient(#f8fafc 20px, transparent 1%) center, #cbd5e1; background-size: 22px 22px;' },
  { id: 'p7', name: 'Art Deco', style: 'border: 5px solid #1e293b; padding: 25px; background: linear-gradient(135deg, #f1f5f9 25%, transparent 25%) -50px 0, linear-gradient(225deg, #f1f5f9 25%, transparent 25%) -50px 0, linear-gradient(315deg, #f1f5f9 25%, transparent 25%), linear-gradient(45deg, #f1f5f9 25%, transparent 25%); background-size: 100px 100px; background-color: #ffffff;' },
  { id: 'p8', name: 'Minimalist Zen', style: 'border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; padding: 40px 20px; text-align: justify;' },
  { id: 'p9', name: 'Vintage Typewriter', style: 'border: 1px solid #475569; padding: 30px; background-color: #f1f5f9; font-family: "Courier New", Courier, monospace;' },
  { id: 'p10', name: 'Ocean Breeze', style: 'border-left: 15px solid #0ea5e9; padding: 20px; background: linear-gradient(to right, #f0f9ff, #ffffff);' },
  { id: 'p11', name: 'Sunset Glow', style: 'border: 3px solid #f43f5e; padding: 20px; border-radius: 20px; box-shadow: 0 0 20px rgba(244, 63, 94, 0.1);' },
  { id: 'p12', name: 'Geometric Bold', style: 'border: 10px solid #0f172a; clip-path: polygon(0% 0%, 100% 0%, 100% 95%, 95% 100%, 0% 100%); padding: 25px;' },
  { id: 'p13', name: 'Soft Pastel', style: 'border: 5px solid #fdf2f8; padding: 20px; background-color: #fff1f2; border-radius: 30px;' },
  { id: 'p14', name: 'Industrial Steel', style: 'border: 4px solid #64748b; padding: 20px; background: repeating-linear-gradient(45deg, #f8fafc, #f8fafc 10px, #f1f5f9 10px, #f1f5f9 20px);' },
  { id: 'p15', name: 'Midnight Neon', style: 'border: 2px solid #818cf8; padding: 20px; box-shadow: 0 0 10px #818cf8, inset 0 0 5px #818cf8; border-radius: 10px;' },
  { id: 'p16', name: 'Classic Library', style: 'border-left: 10px solid #451a03; border-right: 1px solid #451a03; padding: 20px; background-color: #fffaf3;' },
  { id: 'p17', name: 'Modern Gallery', style: 'border: 1px solid #000; padding: 50px; background-color: #fff; box-shadow: 20px 20px 0px #e2e8f0;' },
  { id: 'p18', name: 'Botanical Garden', style: 'border: 2px solid #166534; padding: 20px; background-image: url("https://www.transparenttextures.com/patterns/leaf.png");' },
  { id: 'p19', name: 'Cosmic Star', style: 'border: 1px solid #4c1d95; padding: 20px; background: radial-gradient(circle, #ffffff 0%, #f5f3ff 100%);' },
  { id: 'p20', name: 'Urban Concrete', style: 'border: 6px solid #334155; padding: 20px; background-color: #f1f5f9; border-style: inset;' },
];

export const BORDER_FRAME_INSTRUCTION = `### STYLIST FRAME PROTOCOL ###
Wrap content in a beautiful randomized frame. Choose ONE style from this list for each generation:
1. Double Border: border: 4px double #ea580c; padding: 15px; border-radius: 12px;
2. Modern Shadow: border: 1px solid #e2e8f0; padding: 20px; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
3. Royal Accent: border-left: 8px solid #1e3a8a; border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; padding: 15px; border-radius: 4px;
4. Minimalist Dot: border: 2px dotted #94a3b8; padding: 15px; border-radius: 8px;
5. Gradient Glow: border: 1px solid #f97316; padding: 15px; border-radius: 20px; box-shadow: 0 0 15px rgba(249, 115, 22, 0.2);`;

export const DEFAULT_STRICT_RULES: StrictRule[] = [
  {
    id: 'rule-precision-1',
    label: 'CORE: EXTREME PRECISION TRAP LOGIC',
    description: 'Forces secondary grammar nuances and extreme near-miss distractors.',
    promptInjection: 'STRICT CORE: Every item must test a primary rule and a secondary nuance. Apply EXTREME PROTOCOLS for ALL grammar. Distractors must look 90% correct but fail on situational logic or subtle grammar rules (e.g., "She is as good a student as my father is").',
    active: true,
    priority: 'High',
    category: 'Grammar'
  },
  {
    id: 'rule-logic-1',
    label: 'CORE: PATTERN DESTRUCTION',
    description: 'Rotate sentence structures to prevent predictable patterns.',
    promptInjection: 'STRICT CORE: Rotate sentence structures (Pos/Neg/Int). Max 2 identical structures in a row.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  {
    id: 'rule-no-ai-speak',
    label: 'SUPPORT: NO AI-SPEAK',
    description: 'Ban robotic phrases like "views print" or "understands text".',
    promptInjection: 'STRICT SUPPORT: Ban "AI-speak" like "He knows lines" or "He views print". Use natural child-level actions.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  {
    id: 'rule-no-markdown',
    label: 'SUPPORT: NO MARKDOWN',
    description: 'Ban asterisks. Use HTML tags only.',
    promptInjection: 'STRICT SUPPORT: HTML tags ONLY (<b>, <table>, <p>, <br>). No asterisks. DO NOT use <u> tags.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  {
    id: 'rule-contextual-clues-advanced',
    label: 'Contextual Clues - Advanced',
    description: 'Prioritize context-based clues for all questions, especially vocabulary and grammar.',
    promptInjection: 'Prioritize context-based clues for all questions, especially vocabulary and grammar.',
    active: true,
    priority: 'Average',
    category: 'Vocabulary'
  },
  {
    id: 'rule-ci-no-answers',
    label: 'STRICT: C/I NO ANSWERS',
    description: 'Ensure Correct/Incorrect exercises never show answers in student view.',
    promptInjection: 'STRICT: For all Correct/Incorrect (C/I) exercises, NEVER include the answer (C or I) in the student worksheet. Only provide the blank (____) and the sentence. The answers MUST only appear in the final Answer Key section.',
    active: true,
    priority: 'High',
    category: 'Grammar'
  },
  {
    id: 'rule-no-repetition',
    label: 'STRICT: NO REPETITION',
    description: 'Ensure vocabulary and grammar structures are not repeated across questions.',
    promptInjection: 'STRICT: Do not repeat the same vocabulary words or exact grammar structures across different questions in the same exercise. Every item must be unique.',
    active: true,
    priority: 'High',
    category: 'General'
  }
];

export const DEFAULT_MASTER_PROTOCOLS: StrictRule[] = [
  {
    id: 'mp-test-blueprint',
    label: 'TEST BLUEPRINT ARCHITECTURE',
    description: 'Forces planning before generation.',
    promptInjection: 'BLUEPRINT PHASE: Before generating, silently design a blueprint: 1. Skill Targets, 2. Difficulty Distribution (30% Easy, 40% Mid, 30% Hard), 3. Distractor Types, 4. Micro-themes. Apply expert human pedagogy intuitively through machine-level control.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  {
    id: 'mp-post-audit',
    label: 'POST-GENERATION QUALITY AUDIT',
    description: 'Simulates human proofreading and error checking.',
    promptInjection: 'POST-AUDIT: Verify no unintended answer patterns (e.g., C-C-C) or information leaks between items. Rewrite if violations are found.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  {
    id: 'mp-distractor-psychology',
    label: 'DISTRACTOR PSYCHOLOGY & PLAUSIBILITY',
    description: 'Enforces professional-grade, plausible trap design.',
    promptInjection: 'DISTRACTOR DESIGN: Keep all options plausible (90% correct). NO "stupid wrong" answers. Every question must include 1 NEAR-MISS distractor (almost correct but slightly wrong in meaning). Intuitively mix semantic confusion, contextual misfits, and real ESL learner errors.',
    active: true,
    priority: 'High',
    category: 'Grammar'
  },
  {
    id: 'mp-human-imperfection',
    label: 'HUMAN IMPERFECTION LAYER',
    description: 'Breaks AI perfection patterns to feel more human.',
    promptInjection: 'HUMAN IMPERFECTION: Do not make every item perfectly balanced. Humans sometimes repeat patterns and sometimes make easier distractors. Allow controlled imperfection (uneven sentence lengths, occasional "confidence-builder" items). Too intelligent = AI; Too structured = robotic. Be an expert human.',
    active: true,
    priority: 'Medium',
    category: 'General'
  },
  {
    id: 'mp-corpus-realism',
    label: 'CORPUS REALISM & MICRO-CONTEXT',
    description: 'Ensures natural textbook phrasing and scenarios.',
    promptInjection: 'CORPUS REALISM: Use high-frequency "textbook" English. Avoid overly descriptive "AI adjectives." Use 2-sentence micro-scenarios frequently to provide context.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  {
    id: 'mp-skill-tagging',
    label: 'COGNITIVE SKILL ROTATION',
    description: 'Forces varied thinking levels.',
    promptInjection: 'SKILL ROTATION: Rotate items intuitively between Recall, Application, Inference, and Error Detection.',
    active: true,
    priority: 'Medium',
    category: 'General'
  },
  {
    id: 'mp-reading-paraphrase', 
    label: 'READING: PARAPHRASE & INFERENCE LOGIC', 
    description: 'Controls paraphrasing, keyword matching, and inference.', 
    promptInjection: 'READING LOGIC: 1. ZERO-KEYWORD MATCHING: Never use the exact nouns/adjectives from the text in the correct answer. 2. PARAPHRASE: Shift word forms and use synonyms (simple structural swaps for low levels, complex shifts for high levels). 3. INFERENCE: Test what is implied, not just stated. 4. TFNG: "False" is opposite, "Not Given" is related but unconfirmed.', 
    active: true, 
    priority: 'High', 
    category: 'Reading' 
  },
  { 
    id: 'mp-human-test', 
    label: 'HUMAN-TEST ARCHITECTURE', 
    description: 'Enforces exam-writer logic: simple vocab, high thinking.', 
    promptInjection: 'HUMAN-TEST PROTOCOL: Keep vocabulary simple (A2-B1 max) but raise thinking depth. Use natural phrasing and light idioms. Follow standard exam structures (e.g., KET 3-part structure for lower levels). Scale difficulty by critical thinking, not just obscure vocabulary.', 
    active: true, 
    priority: 'High', 
    category: 'Reading' 
  },
  { 
    id: 'mp-vocab-pure', 
    label: 'PURE VOCABULARY CONTROL', 
    description: 'Enforces same part of speech for all options.', 
    promptInjection: 'In vocabulary sections, all answer choices must be the same part of speech and grammatical form. Students must rely on meaning only, not grammar clues.', 
    active: true, 
    priority: 'High', 
    category: 'Vocabulary' 
  },
  { 
    id: 'mp-answer-key', 
    label: 'ANSWER KEY & LAYOUT LOGIC', 
    description: 'Controls answer distribution and visual formatting.', 
    promptInjection: 'LAYOUT & KEYS: 1. Use the pre-assigned answer keys provided for each part. 2. Distribute items evenly in 2-column layouts. 3. MCQ Layout: Short options on one line (10 spaces between), long options on double lines. DO NOT use vertical lists.', 
    active: true, 
    priority: 'High', 
    category: 'General' 
  },
  {
    id: 'mp-grammar-exhaustion',
    label: 'GRAMMAR RULE EXHAUSTION',
    description: 'Forces the AI to test every single sub-rule of a grammar topic.',
    promptInjection: 'GRAMMAR EXHAUSTION: Intuitively identify and test all specific sub-rules for the target topic, including structural inversions and advanced nodes (Subjunctive, Causatives) where applicable.',
    active: true,
    priority: 'High',
    category: 'Grammar'
  },
  {
    id: 'mp-scenario-chaos',
    label: 'SCENARIO & SYNTACTIC VARIETY',
    description: 'Forces unique themes and varied sentence structures.',
    promptInjection: 'VARIETY: Use unique, vivid scenarios. Vary sentence starters and the position of key grammar signals (Floating Marker Principle) so students cannot scan mechanically.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  {
    id: 'mp-pragmatic-boundary',
    label: 'PRAGMATIC BOUNDARY & SITUATIONAL TESTING',
    description: 'Tests situations and meaning, not just grammatical forms.',
    promptInjection: 'SITUATIONAL TESTING: Test situations and meaning, not just forms. Example: "You have to visit the dentist if your tooth hurts." -> INCORRECT (opinion/advice requires "must"). Example 2: "You must try the cake!" -> CORRECT (personal insistence). Inject cross-topic errors naturally.',
    active: true,
    priority: 'High',
    category: 'Grammar'
  },
  {
    id: 'mp-correct-incorrect-logic',
    label: 'CORRECT/INCORRECT SITUATIONAL LOGIC',
    description: 'Enforces situational distractors for C/I items.',
    promptInjection: 'C/I LOGIC: NO MCQs. Use "1. _____" (5 underscores). Test situations! A sentence can be grammatically correct but SITUATIONALLY INCORRECT.',
    active: true,
    priority: 'High',
    category: 'Grammar'
  },
  {
    id: 'mp-pedagogical-notes',
    label: 'PEDAGOGICAL VOICE',
    description: 'Adds human-like "Teacher Tips" or reminders.',
    promptInjection: 'Randomly insert one small <b>Tip:</b> or <b>Remember!</b> box at the start of one section to simulate a teacher-made worksheet.',
    active: true,
    priority: 'Medium',
    category: 'General'
  },
  {
    id: 'mp-no-free-verb',
    label: 'EXTREME STRICT: NO FREE-MAIN VERB',
    description: 'Prevents giving away the main verb in the question stem.',
    promptInjection: 'NO FREE-MAIN VERB: Never place the main verb in the question stem if it reveals the structure. The main verb MUST be bundled into the answer options. Example: "You ____ the steps exactly." Options: A. has to follow B. have to follow C. must follow D. must to follow.',
    active: true,
    priority: 'High',
    category: 'Grammar'
  },
  {
    id: 'mp-level-calibration',
    label: 'EXTREME STRICT: LEVEL CALIBRATION',
    description: 'Strictly scales vocabulary, sentence length, and text complexity according to the target academic level.',
    promptInjection: 'LEVEL CALIBRATION IS MANDATORY. You must strictly adapt text length, vocabulary, and grammar to the selected level. KIDS/BEGINNER: Max 50-80 words. Super easy vocabulary (A1). Short, simple sentences (Subject-Verb-Object). NO words like "established", "vital", "irrigation". NO inferential or critical thinking questions. Keep it literal. ELEMENTARY/PRE-INT: 100-150 words. Basic compound sentences. INTERMEDIATE: 200-300 words. ADVANCED: 400+ words, complex academic vocabulary. If the level is Kids, the text MUST look like a children\'s book.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  {
    id: 'mp-no-inline-keys',
    label: 'NO INLINE ANSWER KEYS',
    description: 'Prevents answer keys from being printed next to the questions.',
    promptInjection: 'NO INLINE KEYS: NEVER show the answers (True), (False), or (A) inside the test questions or sentences. All answer keys MUST be placed completely separate at the very end of the entire test output, under a "Teacher Answer Key" section.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  {
    id: 'mp-contextual-clues',
    label: 'Contextual Clues Focus',
    description: 'Prioritize context-based clues for all questions, especially vocabulary and grammar.',
    promptInjection: 'Prioritize context-based clues for all questions, especially vocabulary and grammar.',
    active: true,
    priority: 'Medium',
    category: 'General'
  },
  {
    id: 'mp-custom-exercise-logic',
    label: 'CUSTOM EXERCISE ADAPTATION',
    description: 'Ensures AI can handle user-defined exercise types.',
    promptInjection: 'CUSTOM EXERCISES: If an exercise type is not standard (e.g., user-added), analyze its label and professional label to determine the best pedagogical format. Apply standard formatting (nested tables for MCQs, long blanks for writing) where appropriate.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  {
    id: 'mp-reading-uniqueness',
    label: 'READING: ABSOLUTE PASSAGE UNIQUENESS',
    description: 'Forces unique reading passages for every reading exercise.',
    promptInjection: 'STRICT UNIQUENESS: Every single reading exercise MUST have its own unique reading passage. NEVER reuse the same text for multiple exercises in the same test. Each passage must be distinct in content and theme.',
    active: true,
    priority: 'High',
    category: 'Reading'
  }
];

export const INITIAL_TEMPLATES: InstructionTemplate[] = [
  // --- GRAMMAR MASTERY (REORDERED & UPDATED) ---
  { id: 'g_mcq', category: 'GRAMMAR', label: 'MCQ', professionalLabel: 'Choose the BEST option A, B, C or D to complete the following sentences.', prompt: 'Part: Choose the BEST option A, B, C or D to complete the following sentences testing {{TOPIC}}. Apply [NO-FREE-VERB]. Format: Put options on a new line below the question. Use a nested 4-column table for options. MANDATORY: Put 7 non-breaking spaces (&nbsp;) before "A." in the first cell for indentation. Apply PRAGMATIC BOUNDARY logic.', columnCount: 2 },
  { id: 'g_correct_incorrect', category: 'GRAMMAR', label: 'Correct/Incorrect', professionalLabel: 'Write C (correct) or I (incorrect).', prompt: 'Part: Write C (correct) or I (incorrect) for {{TOPIC}}. Apply PRAGMATIC BOUNDARY logic. STRICT: DO NOT include the answer (C or I) in the student worksheet. The answer MUST ONLY appear in the Answer Key section at the end. NEVER put the actual answer next to the sentence in the student view.', columnCount: 2 },
  { id: 'g_circle', category: 'GRAMMAR', label: 'Circle', professionalLabel: 'Circle the correct answers.', prompt: 'Part: Circle the correct answers for {{TOPIC}}. STRICT: No MCQ options.', columnCount: 1 },
  { id: 'g_complete_sentences', category: 'GRAMMAR', label: 'Sentence Complete', professionalLabel: 'Complete the following sentences.', prompt: 'Part: Complete the following sentences for {{TOPIC}}. Note: Use {{BLANK}} and provide the base verb in parentheses at the end of the blank.', columnCount: 1 },
  { id: 'g_pair', category: 'GRAMMAR', label: 'Double MCQ', professionalLabel: 'Double-gap MCQ testing two different aspects of {{TOPIC}} in one sentence.', prompt: 'Part: Double-gap MCQ testing two different aspects of {{TOPIC}} in one sentence. Select the correct pair of words to complete each sentence. Format: Put options on a new line below the question. Use a nested 4-column table for options to ensure they are perfectly aligned (A, B, C, D). Apply PRAGMATIC BOUNDARY logic. Each option must contain two words separated by a slash (e.g., A. word1 / word2).', columnCount: 1 },
  { id: 'g_spelling', category: 'GRAMMAR', label: 'Spelling Rules', professionalLabel: 'Spelling Rules.', prompt: 'Part: Spelling Rules. Complete the table following the spelling rules for the given words related to {{TOPIC}}. This is based on the lessons. STRICT: No MCQ options.', columnCount: 0 },
  
  // --- ADDITIONAL GRAMMAR ---
  { id: 'g_write_correct_form', category: 'GRAMMAR', label: 'Write Correct Form', professionalLabel: 'Write the correct form of …..', prompt: 'Part: Write the correct form of ….. for {{TOPIC}}. Note: Use {{BLANK}}.', columnCount: 1 },
  { id: 'g_rewrite_sentences', category: 'GRAMMAR', label: 'Rewrite Sentences', professionalLabel: 'Rewrite the following sentences.', prompt: 'Part: Rewrite the following sentences about {{TOPIC}}. I need a line for students to write. Provide a long blank line ({{BLANK}}{{BLANK}}{{BLANK}}) for each item.', columnCount: 1 },
  { id: 'g_box', category: 'GRAMMAR', label: 'Word Box', professionalLabel: 'Complete the following sentences using the words/ phrases in the box. Check the correct forms of grammar.', prompt: 'Part: Complete the following sentences using the words/ phrases in the box. Check the correct forms of grammar for {{TOPIC}}. MANDATORY: Use a <div> with class="word-bank-box-alt" for the word bank.', columnCount: 1 },
  { id: 'g_cloze_paragraph', category: 'GRAMMAR', label: 'Cloze Passage', professionalLabel: 'Cloze Passage (Full Paragraph): Fill in the blanks.', prompt: 'Part: Cloze Passage (Full Paragraph): Fill in the blanks with appropriate words for {{TOPIC}}.', columnCount: 1 },

  // --- FULL TEST COMBINATIONS ---
  { id: 'g_full_mastery', category: 'GRAMMAR', label: 'FULL GRAMMAR TEST', professionalLabel: 'Complete the comprehensive grammar assessment covering all major structures.', prompt: 'PART: FULL GRAMMAR TEST. Generate a 4-part test. ITEM COUNT: Generate exactly {{COUNT}} items for EACH part. NUMBERING: Number every single item in each part starting from 1. PARTS: 1. Write C (correct) or I (incorrect), 2. MCQ, 3. Circle the correct answers, 4. Double-Gap MCQ. Apply NO-FREE-VERB mandate.', columnCount: 0 },
  { 
    id: 'g_full_test', 
    category: 'GRAMMAR', 
    label: 'Mixed Test', 
    professionalLabel: 'Complete the comprehensive grammar assessment.',
    prompt: `PART: Mixed Test. 
Section I: Individual Grammar
A. Circle the correct option (15 items)
B. Mark sentences C (Correct) or I (Incorrect) (20 items)
C. Write correct word form (13 underscores + base verb) (10 items)
D. Write negative + question forms (2 sentences)
E. MCQ: Choose the best options, A, B, C or D. (10 items)

Section II: Mixed Grammar
A. Find & correct 10 mistakes. (not topic-based)
B. Choose best option (mixed grammar) (10 items)
C. Double-gap MCQ (not topic-based) (10 items)
D. Teacher Mode (Error Correction: 1 paragraph with 5 errors, Students rewrite below) (not topic-based)`,
    columnCount: 0
  },
  { id: 'r_full_mastery', category: 'READING', label: 'Reading Comprehension', professionalLabel: 'Complete the comprehensive reading assessment to evaluate comprehension and inference skills.', prompt: 'PART: FULL READING TEST. Generate a 5-part test. ITEM COUNT: Generate exactly 10 items for EACH part. NUMBERING: Number every single item in each part starting from 1. COLUMN: 1 column layout. PARTS: 1. True/False, 2. MCQ, 3. Short Answers, 4. Inferential, 5. Critical Thinking. Use a DIFFERENT reading passage for each part. Length and level of thinking based on selected Academic Level. Apply Reading Logic Firewall.', columnCount: 1 },
  { id: 'g_copy', category: 'GRAMMAR', label: 'Copy', professionalLabel: 'Transcribe the following vocabulary exercises accurately into your notebook.', prompt: 'Part: Transcribe the following vocabulary exercises accurately into your notebook for {{TOPIC}}, but randomize the exercise numbers and order. STRICT: No MCQ options.', columnCount: 0 },
  { id: 'g_odd_one_out', category: 'GRAMMAR', label: 'Odd One', professionalLabel: 'Identify the grammatically incorrect sentence from the options provided.', prompt: 'Part: Identify the incorrect sentence from the options provided.', columnCount: 0 },
  { id: 'g_editing', category: 'GRAMMAR', label: 'Editing', professionalLabel: 'Identify and correct the grammatical errors in the following paragraph.', prompt: 'Part: Identify and correct the grammatical errors in the following paragraph. This is the mixed grammar test. The answer can be any types of grammar lessons. Correct all the mistakes.', columnCount: 0 },
  { id: 'g_reduce', category: 'GRAMMAR', label: 'Reduce', professionalLabel: 'Rewrite the following sentences by reducing them to fewer words while maintaining the original meaning.', prompt: 'Part: Rewrite the following sentences by reducing them to fewer words.', columnCount: 0 },
  { id: 'g_best_rewrite', category: 'GRAMMAR', label: 'Best Rewrite', professionalLabel: 'Choose the most appropriate rewrite for each of the following sentences.', prompt: 'Part: Choose the most appropriate rewrite for each sentence.', columnCount: 0 },
  { id: 'g_cloze_passage_short', category: 'GRAMMAR', label: 'Cloze', professionalLabel: 'Complete the cloze passage by filling in the blanks with appropriate grammatical forms.', prompt: 'Part: Complete the cloze passage by filling in the blanks with appropriate words for {{TOPIC}}. Generate a coherent paragraph with 5-10 numbered blanks (e.g., (1) ________). Provide a word bank in a box at the top if appropriate, or let students use their own knowledge.', columnCount: 0 },
  
  // READING
  { id: 'r_tf_stmt', category: 'READING', label: 'True/False', professionalLabel: 'Read the following statements and determine if they are True or False based on the text.', prompt: 'Part: Read the following statements and determine if they are True or False based on the text about {{TOPIC}}. DO NOT GENERATE A, B, C, D OPTIONS. DO NOT USE MULTIPLE CHOICE FORMAT. MANDATORY: Use a wide variety of subjects (e.g., "The dog", "Sarah", "The weather", "They"). DO NOT start every sentence with "I" or the same character\'s name.', columnCount: 1 },
  { id: 'r_mcq', category: 'READING', label: 'MCQ', professionalLabel: 'Choose the appropriate options A, B, C or D based on the detailed reading passage.', prompt: 'Part: Choose the appropriate options A, B, C or D based on the detailed reading passage about {{TOPIC}}. Follow with MCQ testing critical thinking. Format: Put options on a new line below the question. Use a nested 4-column table for options. MANDATORY: Put 7 non-breaking spaces (&nbsp;) before "A." in the first cell for indentation.', columnCount: 1 },
  { id: 'r_short_answer', category: 'READING', label: 'Short Answer', professionalLabel: 'Complete the summary using no more than two words or a number from the text.', prompt: 'Part: Complete the summary using no more than two words or a number from the text about {{TOPIC}}. Follow with fill-in-the-blank summary sentences requiring exact words from the text. STRICT: No MCQ options.', columnCount: 1 },
  { id: 'r_inferential', category: 'READING', label: 'Inferential', professionalLabel: 'Answer the following inferential questions based on the author\'s perspective.', prompt: 'Part: Answer the following inferential questions based on the author\'s perspective about {{TOPIC}}. Follow with discussion questions testing author attitude and implications. STRICT: No MCQ options.', columnCount: 1 },
  { id: 'r_critical_thinking', category: 'READING', label: 'Critical Thinking', professionalLabel: 'Apply critical thinking to answer the following questions based on the analytical reading of the text.', prompt: 'Part: Apply critical thinking to answer the following questions based on the text.', columnCount: 1 },
  { id: 'r_tfng', category: 'READING', label: 'T/F/NG Analysis', professionalLabel: 'Read the text and indicate whether the statements are True (T), False (F), or Not Given (NG).', prompt: 'Part: Read the text and indicate whether the statements are True (T), False (F), or Not Given (NG) about {{TOPIC}}. DO NOT GENERATE A, B, C, D OPTIONS. DO NOT USE MULTIPLE CHOICE FORMAT. MANDATORY: Use a wide variety of subjects (e.g., "The dog", "Sarah", "The weather", "They"). DO NOT start every sentence with "I" or the same character\'s name.', columnCount: 0 },
  { 
    id: 'r_mcq_expert', 
    category: 'READING', 
    label: 'Expert MCQ', 
    professionalLabel: 'Choose the correct option A, B, C or D based on an expert-level analysis of the text.', 
    prompt: 'Part: Choose the correct option A, B, C or D based on an expert-level analysis of the text. Apply [LEXICAL OVERLAP TRAP]. Distractors must include: 1. A "Partial Truth" (mentioned in the text but incomplete), 2. An "Opposite," and 3. A "Contextual Misfit." Apply Zero-Keyword Matching.', 
    columnCount: 0 
  },
  { 
    id: 'r_referential_qs', 
    category: 'READING', 
    label: 'Referential', 
    professionalLabel: 'Determine the referential resolution for the specified pronouns in the passage.', 
    prompt: 'Part: Determine the referential resolution for the specified pronouns in the passage. This tests structural understanding.', 
    columnCount: 0 
  },
  { 
    id: 'r_summary_cloze', 
    category: 'READING', 
    label: 'Summary', 
    professionalLabel: 'Complete the summary of the passage by filling in the blanks with words from the text.', 
    prompt: 'Part: Complete the summary of the passage by filling in the blanks with words from the text. Students must find the correct words from the text to fill the blanks. Use exact word-form from the text.', 
    columnCount: 0 
  },

  // VOCABULARY
  { id: 'v_study_table', category: 'VOCABULARY', label: 'Definition Table', professionalLabel: 'Study the following vocabulary words and their corresponding definitions.', prompt: 'Part: Study the following vocabulary words and their corresponding definitions. Use a 2-column HTML table. Column 1: Number + Word/Phrase. Column 2: Easy Definition. STRICT: No MCQ options.', columnCount: 1 },
  { id: 'v_sentence_study', category: 'VOCABULARY', label: 'Study Example', professionalLabel: 'Study the usage of the following vocabulary words in the provided sentences.', prompt: 'Part: Study the usage of the following vocabulary words in the provided sentences. STRICT: No MCQ options. These are example sentences for learning.', columnCount: 1 },
  { id: 'v_supply_terms', category: 'VOCABULARY', label: 'Supply Key Terms', professionalLabel: 'Read the definitions and provide the correct key terms.', prompt: 'Part: Read the definitions and provide the correct key terms. Use a 2-column HTML table. Column 1: Easy Definition. Column 2: Blank line for Key Term. Randomize order. STRICT: No MCQ options.', columnCount: 1 },
  { id: 'v_box', category: 'VOCABULARY', label: 'Vocabulary Box', professionalLabel: 'Complete the following sentences using the correct words or phrases from the box.', prompt: 'Part: Complete the following sentences using the correct words or phrases from the box. Use long blanks. MANDATORY: Use a <div> with class="word-bank-box-alt" for the word bank. STRICT: No MCQ options.', columnCount: 1 },
  { id: 'v_matching', category: 'VOCABULARY', label: 'Definition Match', professionalLabel: 'Match the following vocabulary words with their appropriate definitions.', prompt: 'Part: Match the following vocabulary words with their appropriate definitions. Use a standard list format. STRICT: No MCQ options.', columnCount: 1 },
  { id: 'v_copy', category: 'VOCABULARY', label: 'Copy', professionalLabel: 'Transcribe the following vocabulary exercises accurately into your notebook.', prompt: 'Part: Transcribe the following vocabulary exercises accurately into your notebook. STRICT: No MCQ options.', columnCount: 1 },
  { id: 'v_synonym_swap', category: 'VOCABULARY', label: 'Synonym Swap', professionalLabel: 'Rewrite each sentence by replacing the underlined word with an appropriate synonym.', prompt: 'Part: Rewrite each sentence by replacing the underlined word with an appropriate synonym. Use a long blank line. STRICT: No MCQ options.', columnCount: 1 },
  { id: 'v_mcq', category: 'VOCABULARY', label: 'MCQ', professionalLabel: 'Choose the appropriate options A, B, C or D to complete each sentence.', prompt: 'Part: Choose the appropriate options A, B, C or D to complete each sentence. Format: Put options on a new line below the question. Use a nested 4-column table for options. MANDATORY: Put 7 non-breaking spaces (&nbsp;) before "A." in the first cell for indentation. Apply Grammar Blackout.', columnCount: 1 },
  { id: 'v_tf', category: 'VOCABULARY', label: 'T/F', professionalLabel: 'Read the statements and indicate whether they are True (T) or False (F).', prompt: 'Part: Read the statements and indicate whether they are True (T) or False (F). FORMAT STRICTLY AS: "1. ______ [Statement]". DO NOT GENERATE A, B, C, D OPTIONS. DO NOT USE MULTIPLE CHOICE FORMAT. MANDATORY: Use a wide variety of subjects (e.g., "The dog", "Sarah", "The weather", "They"). DO NOT start every sentence with "I" or the same character\'s name.', columnCount: 1 },
  { id: 'v_speaking', category: 'VOCABULARY', label: 'Speaking', professionalLabel: 'Discuss the following questions with a partner to practice your speaking skills.', prompt: 'Part: Discuss the following questions with a partner to practice your speaking skills. Generate open-ended discussion questions related to {{TOPIC}}. STRICT: No MCQ options.', columnCount: 1 },
  { id: 'v_writing_definition_1', category: 'VOCABULARY', label: 'Writing Definition (Style 1)', professionalLabel: 'Provide the correct definition for each of the following vocabulary words.', prompt: 'Part: Provide the correct definition for each of the following vocabulary words. Use a 2-column HTML table. Column 1: Word. Column 2: Long blank line for Definition (________________________________). STRICT: No MCQ options.', columnCount: 1 },
  { id: 'v_writing_definition_2', category: 'VOCABULARY', label: 'Writing Definition (Style 2)', professionalLabel: 'Read the definitions and write the corresponding vocabulary word.', prompt: 'Part: Read the definitions and write the corresponding vocabulary word. Use a 2-column HTML table. Column 1: Definition. Column 2: Blank line for Word. Randomize order. STRICT: No MCQ options.', columnCount: 1 },
  { id: 'v_writing_definition_3', category: 'VOCABULARY', label: 'Writing Definition (Style 3)', professionalLabel: 'Match the word to its definition and write it down.', prompt: 'Part: Match the word to its definition and write it down. Use a 2-column HTML table. Column 1: A list of definitions. Column 2: A blank line for the word. Include a word bank at the top in a box.', columnCount: 1 },
  { id: 'v_supply_terms_v2', category: 'VOCABULARY', label: 'Supply Key Terms (Pro)', professionalLabel: 'Match the descriptions with the correct key terms from the provided list.', prompt: 'Part: Match the descriptions with the correct key terms. Use a professional 2-column HTML table with zebra striping (alternate row backgrounds). Column 1: Number + First Letter of Word (e.g. 1. B). Column 2: Detailed Description. Header Row: "WORD" and "DESCRIPTION".', columnCount: 1 },
  { id: 'v_matching_pro', category: 'VOCABULARY', label: 'Definition Match (Pro)', professionalLabel: 'Match the terms on the left with their definitions on the right.', prompt: 'Part: Match the terms on the left with their definitions on the right. Use a 3-column HTML table. Col 1: Terms (A, B, C...). Col 2: Blank for answer (____). Col 3: Definitions (1, 2, 3...). Randomize both sides.', columnCount: 1 },
  { id: 'v_matching_v3', category: 'VOCABULARY', label: 'Term Match (Style 3)', professionalLabel: 'Connect the terms with their meanings.', prompt: 'Part: Connect the terms with their meanings. Use a 2-column layout where the term is on the left and a blank line is on the right. (e.g. 1. Apple _________________).', columnCount: 1 },
  { id: 'v_matching_v4', category: 'VOCABULARY', label: 'Term Match (Style 4)', professionalLabel: 'Draw a line to match the word with its definition.', prompt: 'Part: Draw a line to match the word with its definition. Use a 2-column layout with wide spacing. Column 1: Words (1, 2, 3...). Column 2: Definitions (A, B, C...). Randomize both sides.', columnCount: 1 },
  { id: 'v_spelling_correction', category: 'VOCABULARY', label: 'Spelling Correction', professionalLabel: 'Identify the misspelled word in each sentence and write the correct version on the right.', prompt: 'Part: Identify the misspelled word in each sentence and write the correct version on the right. Use a 2-column layout. Col 1: Sentence with one intentional spelling error. Col 2: Blank line for correction.', columnCount: 1 },
  { id: 'v_cipher_code', category: 'VOCABULARY', label: 'Cipher Code', professionalLabel: 'Use the code key below to decode the vocabulary words.', prompt: 'Part: Use the code key below to decode the vocabulary words. First, generate a 2-row HTML table mapping A-Z to random letters. Then, list 10 encoded words with blanks next to them for decoding.', columnCount: 1 },
  { id: 'v_cloze_test', category: 'VOCABULARY', label: 'Cloze Test', professionalLabel: 'Fill in the blanks in the following paragraph using the words from the word bank.', prompt: 'Part: Fill in the blanks in the following paragraph using the words from the word bank. Generate a coherent paragraph related to {{TOPIC}} with 10 blanks. Provide a word bank in a box at the top.', columnCount: 1 },
  { id: 'v_synonyms', category: 'VOCABULARY', label: 'Synonyms', professionalLabel: 'Provide appropriate synonyms for the vocabulary words in the table.', prompt: 'Part: Provide appropriate synonyms for the vocabulary words in the table.\n5 columns: 1 vocabulary, 2 synonym 1, 3, syn 2, ....\nIf there are more than 5 synonyms, please have another row.', columnCount: 5 },
  { id: 'v_synonym_writing', category: 'VOCABULARY', label: 'Syn Writing', professionalLabel: 'Complete the synonym writing task as instructed.', prompt: 'Part: Complete the synonym writing task as instructed.\n5 columns: 1 vocabulary, 2 synonym 1, 3, syn 2, ....\nIf there are more than 5 synonyms, please have another row.', columnCount: 1 },
  { id: 'v_circle', category: 'VOCABULARY', label: 'Circle', professionalLabel: 'Circle the correct vocabulary word from the options provided in parentheses.', prompt: 'Part: Circle the correct word from the options provided.', columnCount: 0 },
  { id: 'g_error_identification', category: 'GRAMMAR', label: 'Error ID', professionalLabel: 'Identify the underlined part (A, B, C, or D) that contains an error.', prompt: 'Part: Identify the underlined part (A, B, C, or D) that contains an error in each sentence. Underline 4 parts of the sentence and label them A, B, C, D.', columnCount: 1 },
  { id: 'v_categorization', category: 'VOCABULARY', label: 'Categorize', professionalLabel: 'Categorize the following words into the correct groups.', prompt: 'Part: Categorize the following words into the correct groups. Use a 3-column table for categories. Provide a word bank at the top.', columnCount: 0 },
  { id: 'v_crossword_clues', category: 'VOCABULARY', label: 'Crossword Clues', professionalLabel: 'Solve the clues to find the vocabulary words.', prompt: 'Part: Solve the clues to find the vocabulary words. List clues for "Across" and "Down". No grid needed, just the clues and blanks.', columnCount: 2 },
  
  // GENERALS
  { id: 'gen_tf', category: 'GENERALS' as any, label: 'True/False (Mixed)', professionalLabel: 'Read the mixed content and determine if the statements are True or False.', prompt: 'Part: Read the mixed content (Grammar, Vocabulary, and Reading) and determine if the statements are True or False about {{TOPIC}}. Apply logic across all domains.', columnCount: 1 },
  { id: 'gen_correct_incorrect', category: 'GENERALS' as any, label: 'Correct/Incorrect (Mixed)', professionalLabel: 'Identify if the mixed usage is Correct or Incorrect.', prompt: 'Part: Identify if the mixed usage (Grammar, Vocabulary, and Reading) is Correct (C) or Incorrect (I) for {{TOPIC}}. STRICT: DO NOT include the answer (C or I) in the student worksheet. The answer MUST ONLY appear in the Answer Key section at the end. NEVER put the actual answer next to the sentence in the student view.', columnCount: 2 },
  { id: 'gen_full_mixed', category: 'GENERALS' as any, label: 'FULL GENERAL TEST', professionalLabel: 'Complete the comprehensive mixed assessment covering Grammar, Vocabulary, and Reading.', prompt: 'PART: FULL GENERAL TEST. Generate a 3-part test mixing Grammar, Vocabulary, and Reading. Part 1: True/False, Part 2: Correct/Incorrect, Part 3: MCQ. Apply situational logic across all domains.', columnCount: 0 },
  {
    id: 'gen_matching_classic',
    category: 'GENERALS' as any,
    label: 'Matching (Classic)',
    professionalLabel: 'Classic Matching Exercise',
    prompt: 'Generate a matching exercise with words on the left (1-8) and definitions on the right (A-H). Use a line (____) before each word for the answer.',
    columnCount: 2
  },
  {
    id: 'gen_matching_boxed',
    category: 'GENERALS' as any,
    label: 'Matching (Boxed)',
    professionalLabel: 'Boxed Matching Exercise',
    prompt: 'Generate a matching exercise where definitions are on the right and answers are written in boxes on the left. Include a word bank at the bottom.',
    columnCount: 2
  },
  {
    id: 'gen_matching_column',
    category: 'GENERALS' as any,
    label: 'Matching (Column A/B)',
    professionalLabel: 'Column A & B Comparison',
    prompt: 'Generate a matching exercise with Column A and Column B. Use boxes for answers on the far left. Separate columns with a vertical line.',
    columnCount: 2
  },
  {
    id: 'mixed_full_test',
    category: 'Mixed' as any,
    label: 'Mixed Subject Test',
    professionalLabel: 'Comprehensive Mixed Assessment (Grammar, Reading, Vocabulary)',
    prompt: `PART: Mixed Subject Test. 
Generate a multi-section test covering:
1. Grammar: MCQ and Correct/Incorrect.
2. Reading: A short passage followed by True/False, MCQ, and Inferential Reading.
3. Vocabulary: MCQ and True/False.
Apply all relevant Master Protocols for each section.`,
    columnCount: 1
  },
  { id: 'custom_subject_test',
    category: 'CUSTOM' as any,
    label: 'Custom Subject Test',
    professionalLabel: 'Subject-Specific Assessment',
    prompt: 'Generate a custom test for the subject {{TOPIC}}. Mix relevant exercise types (MCQ, T/F, Matching) specific to this subject.',
    columnCount: 1
  },
];
