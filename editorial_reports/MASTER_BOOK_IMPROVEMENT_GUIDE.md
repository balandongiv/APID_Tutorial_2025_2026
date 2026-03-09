# MASTER_BOOK_IMPROVEMENT_GUIDE

## Book-Level Findings

The manuscript introduces a project‑based module on hydroponics product development for third‑year Industrial Physics students. While the overall scope is appropriate, the draft exhibits several systemic issues that undermine its effectiveness.

- **Target audience fit:** Throughout the sessions the language and examples oscillate between general engineering design and business management. Many activities assume prior knowledge of design tools or business concepts that Industrial Physics students may not possess. Conversely, the physics underpinning of hydroponic systems (e.g., fluid dynamics, thermodynamics, materials) is rarely highlighted. A stronger, discipline‑specific orientation is needed to ensure that physics students see how their technical skills apply to product development. Linking every session’s examples and discussion to applied physics will improve relevance.
- **Recurring clarity problems:** Most chapters start abruptly with long enumerated tasks and lack introductions, learning objectives and signposting. Instructions are often ambiguous or grammatically incorrect. Cross‑references are broken or missing, and chapters frequently end without a summary or next steps. Without clear explanations of why activities matter, students may struggle to connect tasks to broader learning goals. Many sessions also repeat content or refer to non‑existent files, creating confusion.
- **Recurring terminology problems:** Key terms (innovation charter, direct vs. indirect products, market segment, functional decomposition, classification tree, screening matrix, concept test) are introduced without definition. Acronyms such as LLM, CLO, ITEL, PD and guidelines numbered 1–5 appear without explanation. Units and abbreviations vary across chapters. A unified glossary and consistent definitions are essential to reduce cognitive load.
- **Recurring pedagogy problems:** The curriculum lacks explicit learning objectives, examples tailored to the hydroponic context, evaluation criteria for exercises, and reflective prompts. Activities often consist of long lists of questions or tasks without scaffolding or templates. Students are not told how to judge the quality of their work or how it connects to subsequent sessions. The absence of worked examples and physics‑based illustrations reduces engagement. Research shows that clearly articulated objectives and aligned activities improve student performance and motivation【708506183906356†L160-L169】.
- **Recurring formatting / LaTeX problems:** Several LaTeX issues recur: duplicated `\input` commands (Session 0), missing or broken references, long tables that are difficult to read, inconsistent heading levels, and absent `\chapter` commands. Some tables are formatted with landscape orientation without clear need, and captions are missing or incomplete. Shared macros or packages are not defined centrally, leading to compile risks. There is no dedicated glossary or notation file, causing ad‑hoc definitions across chapters.
- **Global structural recommendations:** Each session should follow a consistent structure: a concise introduction explaining the session’s purpose and learning objectives; definitions of new terms; contextual examples linking to hydroponic product development and applied physics; clearly numbered activities with explanatory text; templates or sample tables/figures; evaluation criteria and guidance for collaboration; a concluding summary that links to the next session. Figures and tables should be consistently captioned and referenced. A global glossary, notation list and style guide should govern terminology, units and formatting across the book.

## Shared-File Conflict Resolution

| Shared file | Sessions touching it | Conflict / risk | Resolved decision |
|---|---|---|---|
| **main.tex** | Session 0, Session 5/6, Session 8 and many others | Duplicate `\input{1b_llm.tex}` lines in Session 0 cause repeated content; references to `appendix/revision_history` do not exist; chapters lack proper `\chapter` commands. | Remove duplicate `\input` commands and ensure each chapter is included once; comment out or create missing files (e.g., `revision_history.tex`); insert `\chapter{}` or `\chapter*{}` for every session; standardise the order of inputs to reflect the course flow. |
| **preamble.tex / macros** | All sessions | Lack of a central preamble leads to ad‑hoc package loading; missing definitions (e.g., for coloured boxes, long tables) cause compile errors. | Create a dedicated `preamble.tex` or expand `main.tex` to include all required packages (e.g., `longtable`, `tabularx`, `tikz`, `tcolorbox`, `minted` if code is added) and define common environments. Place macro definitions (e.g., for verbs in functional decomposition, call‑out boxes) in a shared `macros.tex`. |
| **glossary / notation files** | All sessions | No global glossary exists; terms are defined inconsistently. | Introduce a `glossary.tex` file where all key terms (innovation charter, need, metric, functional decomposition, concept classification tree, etc.) are defined once. Use the `glossaries` package for consistent referencing. |
| **bibliography / references** | Sessions 0 and 12 | External citations (e.g., learning objectives article, functional decomposition guide, concept testing article) are referenced in reports but not integrated into the manuscript. | Create a `references.bib` file and use `biblatex` or similar to cite sources properly. Ensure that citations appear in the text with consistent formatting. |
| **shared figure folders** | Sessions 5/6, 7, 8, 9, 10 | Some sessions refer to images (e.g., thermostat, bicycle suspension) unrelated to hydroponic design; other sessions need new templates or example diagrams. | Standardise image naming and storage in session‑specific folders (e.g., `session7/hydroponic_needs_example.pdf`); remove irrelevant figures or move them to an appendix; ensure that common templates (tables, matrices, classification trees) are stored centrally and referenced across sessions. |

## Global Style Decisions

To maintain consistency across the book, the following style decisions should be adopted:

- **Terminology canon:** Establish a glossary containing agreed definitions for key terms (e.g., *innovation charter*, *opportunity*, *market segment*, *mission statement*, *need*, *metric*, *functional decomposition*, *classification tree*, *screening matrix*, *concept test*). All sessions must refer to these terms exactly and define them upon first use. Avoid introducing new terms without adding them to the glossary.
- **Notation canon:** For functional decomposition, describe functions with verb–noun pairs in plain text (e.g., “pump nutrient solution”). For matrices: use plus/zero/minus (+/0/–) notation for screening matrices, 1–5 importance scales for needs, and 1–5 scoring scales for concept scoring. Use table headers to label metrics and guidelines clearly. For guidelines in specification tables, write out their names rather than referring only to numbers.
- **Unit style:** Use SI units consistently (e.g., L/min for flow rate, °C for temperature, mg/L for nutrient concentration). Currency should be expressed in Malaysian Ringgit (RM) with two decimal places where necessary. When dealing with scales (1–5), explain the meaning of each value in the caption or text.
- **Heading style:** Each session should begin with `\chapter{Session N: <Title>}` (or `\chapter*{}` if unnumbered), followed by `\section`, `\subsection`, and `\subsubsection` as needed. Avoid deep nesting beyond three levels. Use clear, descriptive titles rather than generic labels (e.g., “Purpose and Learning Objectives” instead of “Activity 1”).
- **Activity / callout naming:** Number activities sequentially within each session (e.g., “Activity 1: Market Analysis”), ensuring numbers reset for each chapter. Use consistent labels for optional exercises (e.g., “Optional Reflection”). Call‑out boxes should be styled uniformly (e.g., grey shading) and labelled clearly (e.g., “Example Charter”).
- **Figure / table caption style:** All figures and tables must have a caption in the format “Figure X.Y: Description” or “Table X.Y: Description”, where X is the session number and Y is the sequence within the session. Captions should be descriptive and refer to the figure/table in the text. Use `\label{}` for cross‑references. Long tables should use `longtable` or `tabularx` with appropriate width specifications.
- **Voice / tone:** Use a professional, instructional voice addressed directly to students (“Your team should…”, “You will…”). Avoid colloquialisms and personal commentary. Maintain present tense and active voice. When encouraging creativity, frame instructions positively rather than admonishing (“Use your imagination to generate opportunities” instead of “Do not use LLMs”).

## Chapter-by-Chapter Improvement Plan

### Session 0 – Front Matter / Preface / Orientation
**Top fixes:** Introduce explicit learning objectives; remove duplicate `\input` commands; rewrite the preface to align with the hydroponics and Industrial Physics context; define all acronyms and terms; reorganise content into clear sections (course purpose, project theme, weekly activities, assessments); add signposting and a summary; correct grammar.

**P0 items:** Define APID, LLM, CLO, ITEL and other acronyms; remove duplicated `\input{1b_llm.tex}`; rewrite or replace the Python‑oriented preface; add a `\chapter{Introduction}` heading; fix missing files (`revision_history.tex`); correct incomplete sentences and grammar errors; ensure each list has context.

**P1 items:** Add learning objectives; break dense lists into subsections; provide a timeline table of deliverables and due dates; integrate golden rules into a succinct call‑out; connect the hydroponics theme to applied physics; streamline the voice to a consistent second‑person instruction.

**P2 items:** Develop a glossary; include diagrams (e.g., Gantt chart) summarising the semester; add external resource links; create a FAQ section.

**Implementation notes:** Coordinate with editors of later sessions to ensure that definitions introduced here remain consistent. Remove or create missing files referenced in `main.tex`. Place golden rules in an appendix or call‑out to declutter the main narrative.

### Session 1 – 
**Top fixes:** Introduce a session overview with learning objectives; define direct vs. indirect products; organise example products into a table linked to physics principles; add guidance on evaluating and selecting products and companies; provide a concluding summary connecting to the next session.

**P0 items:** Add explicit learning objectives; define key terms; standardise list formatting; insert a conclusion summarising tasks and prepping for Session 2; correct grammar and punctuation.

**P1 items:** Reorder content logically (overview → definitions → examples → tasks); use a table to present example products with associated physics concepts; provide criteria for evaluating opportunities; align the voice to second person.

**P2 items:** Include a template or worksheet for documenting product and company choices; suggest optional research activities; add reflection prompts linking opportunity identification to applied physics and industrial design.

**Implementation notes:** Cross‑check definitions of direct/indirect products with later chapters; design the example table using `tabularx` for readability; ensure tasks connect to the hydroponic context; emphasise that the chosen product will be developed throughout the course.

### Session 2 – 
**Top fixes:** Introduce learning objectives and define product development processes, flows and organisational structures; reorganise the chapter into clear subsections; simplify the process type table or split it into manageable pieces; provide examples relevant to hydroponics; connect process choices to physics considerations and project constraints.

**P0 items:** Add learning objectives; define key terms (generic process, spiral, complex, departmental vs. organisational structures); fix grammar; clarify distinctions between concepts; remove duplication of phrases.

**P1 items:** Reorganise content into logical sections (process types, process flows, departmental structures, organisational structures); simplify the table and provide context; insert transitions between tasks; connect process selection to hydroponic product types and physics (e.g., technology‑push for sensor‑intensive systems).

**P2 items:** Include flowcharts of the stage‑gate process; provide decision matrices or templates for selecting processes; add case studies from agriculture or industrial design.

**Implementation notes:** Test compile the redesigned tables using `tabularx` or `longtable`; coordinate with the glossary editor to define all terms; ensure that process types used here match those referenced later (e.g., in functional decomposition and concept selection).

### Session 3 – 
**Top fixes:** Define “innovation charter”; add learning objectives; clarify instructions for drafting charters; provide examples (including one hydroponic example); guide the selection and synthesis process; connect the charter to applied physics.

**P0 items:** Define the term and its purpose; add learning objectives; fix grammatical errors; give step‑by‑step guidance on drafting and selecting charters; correct ambiguous phrases.

**P1 items:** Include contextual explanations of why broad charters encourage exploration; provide criteria for evaluating charters; add signposting and summarise the session’s output; relate charters to physics (e.g., energy efficiency, nutrient monitoring).

**P2 items:** Offer a fill‑in‑the‑blank charter template; include a peer review step; suggest optional readings on innovation charters.

**Implementation notes:** Ensure the chosen charter feeds into opportunity identification in Session 4; coordinate with the glossary to define “scope” and other relevant terms; use call‑out boxes to present example charters; avoid duplicating phrases like “innovation chart.”

### Session 4 – 
**Top fixes:** Add learning objectives explaining the purpose of opportunity identification; fix cross‑reference labels to the innovation charter; define key terms; provide evaluation criteria for screening opportunities; specify a standard template for recording and ranking opportunities; include context linking to hydroponics and physics.

**P0 items:** Add a clear introduction and learning objectives; correct broken `\ref{}` labels; define “opportunity,” “charter,” and “value creation”; describe evaluation criteria (feasibility, alignment, impact, resources); specify deliverable formats; correct grammar.

**P1 items:** Break long enumerations into labelled subsections (Purpose, Procedure, Expected Output); include relevant examples and demonstrate how physical principles inspire opportunities; emphasise connection to the innovation charter and hydroponic context; insert a concluding summary.

**P2 items:** Suggest optional readings on brainstorming methods; provide a case study illustrating opportunity screening; offer decision‑making techniques (e.g., multi‑vote) for prioritising opportunities.

**Implementation notes:** Check that labels used for the innovation charter section in Session 3 match those referenced here; provide a table template using `tabularx`; define evaluation criteria in the glossary; ensure the prioritised list feeds into product planning in Sessions 5/6.

### Session 5 – 
**Top fixes:** Provide learning objectives and definitions for market segmentation, technology integration, manufacturing/service analysis, assumptions/constraints and mission statement; supply templates for market analysis and mission statements; clarify evaluation criteria and expectations; remove unprofessional remarks; connect the planning activities to the hydroponic and physics context.

**P0 items:** Introduce objectives; define all terms; add a `\chapter{}` heading; delete filler text (e.g., “Im here…”); remove ambiguous numbering; provide explicit criteria for market analysis and mission statements; correct grammar and tense errors.

**P1 items:** Organise questions under meaningful subheadings; emphasise the link between market analysis and physics constraints; include example market segments and mission statements; describe collaborative methods for refining analyses; move optional remarks to a clearly labelled subsection.

**P2 items:** Offer a bibliography of resources on product planning; suggest market research tools; include a mission statement checklist; give tips on professional tone.

**Implementation notes:** Coordinate with Session 6 (which shares the same file) to clearly demarcate the sessions; test compile new tables; place templates in separate files if they will be reused; ensure consistency of terms across the book.

### Session 6 – 
**Top fixes:** Distinguish Session 6 content within the combined Session 5/6 file; define refinement objectives; provide a process for merging individual market analyses and mission statements into a group plan; supply consolidation templates; link the refined mission statement to subsequent sessions.

**P0 items:** Insert a clear heading for Session 6; state learning objectives focusing on synthesis and refinement; describe steps for consolidating analyses; provide a final mission statement template; connect outputs to Session 7 and Session 8.

**P1 items:** Describe collaboration techniques for merging analyses; encourage verification of data and assumptions; emphasise physics context in market and technology choices; include reflection prompts on the evolution from Session 5 to Session 6.

**P2 items:** Suggest peer review of mission statements; provide example mission statements; recommend collaborative tools for tracking revisions.

**Implementation notes:** Work closely with the Session 5 editor to avoid duplication and ensure a smooth transition; test compile new sections and templates; make sure refined documents are referenced in later chapters.

### Session 7 – 
**Top fixes:** Add learning objectives and context for customer needs identification; define terms such as customer statement, needs statement, lead user, extreme user and variable type; provide guidance on survey design and translation of statements into needs; supply standard table templates; incorporate examples relevant to hydroponics; emphasise how needs feed into specifications.

**P0 items:** Introduce an objective‑driven overview; define all key terms; standardise table formats; clarify deliverables (survey questions, needs table, lead/extreme user analysis, reflective response); explain the importance of lead/extreme users; correct grammar and phrasing.

**P1 items:** Organise activities into subsections (Survey Question Generation, Needs Translation, Lead/Extreme User Consideration, Reflection); give guidelines for crafting unbiased survey questions; replace generic examples with hydroponic examples; explain variable types; encourage reflection on biases; add a concluding summary.

**P2 items:** Provide links to resources on survey design; introduce techniques such as empathy mapping; suggest software tools for organising needs; include a reflective exercise on innovation triggered by unmet needs.

**Implementation notes:** Ensure the needs identified here align with the market analysis and mission statement from Sessions 5/6; design the tables with `p{}` columns for readability; update glossary entries for all new terms; test compile the long table of needs.

### Session 8 – 
**Top fixes:** Introduce learning objectives and the role of specifications; define metric guidelines explicitly; replace irrelevant examples with hydroponic ones; provide templates for needs‑importance tables, metric definition tables, need‑metric matrices and benchmarking tables; clarify how to evaluate metrics and encourage critical evaluation of LLM outputs; add a conclusion.

**P0 items:** Add an introduction; define guidelines for metrics (reflect customer needs, support design freedom, practical measurement, subjective measures, market comparison); replace the bicycle suspension example with a hydroponic example; explain how to derive and justify metrics; provide templates and remove duplicate headings; correct grammar and numbering.

**P1 items:** Organise content into sections (Assigning Importance, Defining Metrics, Evaluating Guidelines, Need‑Metric Matrix, Benchmarking and Visualisation); contextualise guidelines with examples; encourage engineering reasoning; explain competitive benchmarking; describe graphing options; add a concluding summary linking to concept generation.

**P2 items:** Include optional practice problems and further reading; suggest multi‑criteria decision tools; encourage reflection on trade‑offs between competing metrics.

**Implementation notes:** Coordinate with Session 7 to ensure needs feed directly into the needs‑importance table; test compile new tables and matrices; define all new terms in the glossary; caution students about relying solely on LLMs for metrics and graphs.

### Session 9 – 
**Top fixes:** Introduce learning objectives explaining functional decomposition and solution search; define decomposition approaches and instruct on verb–noun pair notation; provide templates for decomposition diagrams, requirement–function matrices and solution search tables; include hydroponic examples; explain how to evaluate solutions; add a summary connecting outputs to concept generation.

**P0 items:** Add an introduction; define functional decomposition, user‑action sequence decomposition and customer‑needs decomposition; emphasise using generic verb–noun pairs and avoiding implementation details【979038334412772†L77-L121】; provide templates; explain requirement–function matrices; clarify solution search criteria; correct grammar.

**P1 items:** Structure the chapter into clear sections; replace or contextualise the seed planter example with a hydroponic example; describe diagramming conventions; encourage credible research and citation; explain how the requirement–function matrix ensures coverage of needs; include a conclusion.

**P2 items:** Offer optional exercises on decomposing familiar devices; suggest diagramming tools; discuss how decomposition reveals innovation opportunities; encourage peer feedback on diagrams.

**Implementation notes:** Ensure that customer requirements from Session 7 feed into the requirement–function matrix; design diagrams using `tikz` or provide external figures; label all tables and diagrams; update glossary entries for decomposition approaches.

### Session 10 – 
**Top fixes:** Add learning objectives and an introduction explaining concept generation; define concept classification trees and combination tables; provide hydroponic examples; show how to create and use these tools; offer guidance on generating and evaluating rough concepts; connect concept generation to functional decomposition and specifications; include a conclusion.

**P0 items:** Provide definitions of classification trees and combination tables; add learning objectives; include hydroponic examples; specify deliverable formats (trees, tables, concept descriptions); clarify how many concepts to generate; correct ambiguous instructions.

**P1 items:** Organise content into sections (Purpose, Classification Trees, Combination Tables, Concept Generation, In‑Class Consolidation); incorporate physics and hydroponic context; provide evaluation criteria; encourage creativity and diversity of solutions; clarify the scope and number of concepts; add a conclusion.

**P2 items:** Suggest optional exercises using diagramming software; recommend readings on creative design methods; encourage peer feedback; discuss how classification trees reveal gaps in solution space.

**Implementation notes:** Coordinate with Session 9 to ensure that sub‑problems and solutions align; design templates using `tabular` or `tikz`; standardise numbering of concepts; update glossary entries for new terms.

### Session 11 – 
**Top fixes:** Provide learning objectives explaining concept selection; define selection criteria, screening matrices and scoring matrices; supply templates and examples; link selection criteria to needs and specifications; describe how to construct and interpret screening matrices; offer guidance on when to perform scoring; include a conclusion.

**P0 items:** Add an introduction and learning objectives; define key terms; provide a screening matrix template; explain how to derive selection criteria from customer needs and enterprise considerations; clarify the +/0/– scoring method; describe the concept scoring process and when to use it; correct grammar.

**P1 items:** Organise content into sections (Purpose, Generating Criteria, Building the Screening Matrix, In‑Class Consolidation, Concept Scoring); provide hydroponic examples; offer collaboration techniques; suggest evaluation criteria; summarise the selection process; link to next steps.

**P2 items:** Provide optional examples of scoring matrices; offer exercises on simpler products; suggest readings on multi‑criteria decision analysis; encourage reflection on biases.

**Implementation notes:** Coordinate with Session 10 to ensure concept names and numbering match; design templates using `tabular`; update glossary entries for selection tools; test compile tables for readability.

### Session 12 – 
**Top fixes:** Introduce learning objectives and define concept testing; explain the purpose of concept testing and how it reduces risk【203286957744365†L147-L170】; define survey terms and methods; provide guidance on selecting populations and calculating sample sizes; supply templates for concept test plans and surveys; provide examples tailored to hydroponics; clarify communication methods; add a conclusion.

**P0 items:** Add an introduction defining concept testing and stating objectives; provide definitions of survey population, sample size, survey format, communication method; explain how to formulate research questions; include guidelines for selecting populations and calculating sample sizes; supply templates; clarify grammar and phrasing; connect the test to selected concepts.

**P1 items:** Structure content into sections (Purpose, Purpose Definition, Survey Population, Survey Format, Survey Questions, Communication Method); link to previous sessions (selected concepts); provide hydroponic examples; include sample questions; explain pros and cons of different survey formats; encourage iterative refinement; summarise the session.

**P2 items:** Suggest external resources on survey design; propose using online survey tools; encourage ethical considerations; include reflection on how feedback may impact designs.

**Implementation notes:** Ensure that the concept test plan references the concepts selected in Session 11; design templates using `tabular`; update glossary for new terms; caution students about bias and sample representativeness; test compile templates; include a section on ethical conduct if possible.

### Session 13 – 
**Top fixes:** Confirm whether a final session exists; if so, create a Session 13 chapter that outlines final presentations and reflections; define learning objectives (synthesising the design process, presenting the final concept or prototype, reflecting on learning); describe deliverables (final report, presentation, prototype); provide evaluation criteria; specify logistics and timelines.

**P0 items:** If there is a final session, draft a `\chapter{Final Presentation and Reflection}` that explains the session’s purpose; define deliverables and their structure; provide a rubric for evaluation; specify presentation length and format; give submission deadlines; remove placeholders in `main.tex` referencing missing chapters.

**P1 items:** Encourage student reflection on lessons learned and application of physics principles; relate skills developed to professional practice; offer presentation tips; include examples of effective final reports and presentations; suggest inviting external reviewers.

**P2 items:** Suggest optional poster or demo sessions; recommend including industry guests; encourage students to consider future work or commercialisation; include an ethics reminder.

**Implementation notes:** Verify course requirements with instructors; coordinate with other editors to compile final deliverable templates; ensure cross‑references to earlier chapters remain valid; if a final session is not included, remove references to Session 13 in the table of contents and assignment schedules.

## Suggested Revision Sequence
1. Shared/global fixes
2. P0 fixes across all sessions
3. P1 pedagogy improvements
4. P2 refinements

## Final QA Before Release
- [ ] terminology unified
- [ ] notation unified
- [ ] figure/table styles unified
- [ ] callout styles unified
- [ ] all references valid
- [ ] XeLaTeX compile passes from main.tex
- [ ] no broken includes
- [ ] no duplicate labels
- [ ] summary guidance exists for every session
