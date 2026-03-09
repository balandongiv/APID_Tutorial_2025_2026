# Revision Report – Session 10 / Concept Generation

## Chapter Overview
| Item | Description |
|---|---|
| **Chapter file(s)** | `latex/session10/session10.tex` |
| **Related/shared files** | Shared preamble and macros, global glossary, and any figures or tables referenced from previous sessions (e.g., functions from Session 9) |
| **Current chapter purpose** | This chapter introduces concept generation tools—concept classification trees and concept combination tables—and guides students through creating rough product concepts. Pre‑class activities ask students to familiarise themselves with these tools and develop classification trees and combination tables for critical sub‑problems. Students then generate rough product concepts by combining solutions. In class, teams consolidate their combination tables and select the top concepts for further development. |
| **Intended student takeaway** | Students should learn structured methods for generating diverse concepts, understand how to classify and combine sub‑solutions, and practise creating and evaluating rough product concepts that align with customer needs and specifications. |
| **Audience‑specific risks** | The chapter lacks learning objectives, definitions, and examples. Instructions are brief and may not be sufficient for students unfamiliar with classification trees and combination tables. There is no guidance on how to evaluate or describe concepts, nor any connection to the hydroponic project or physics principles. Without context, students may produce superficial concepts. |

## Relevant Editorial Feedback (Mapped)
- **Missing learning objectives and context**: The session starts with activities but does not explain the purpose of concept generation or how it builds on functional decomposition. Students need clear objectives【708506183906356†L160-L169】.
- **Undefined tools**: “Concept classification trees” and “concept combination tables” are referenced without explanation. Students unfamiliar with design methodologies may not know how to create or use them.
- **Lack of examples**: There are no examples illustrating how to build a classification tree or combination table, nor how to combine sub‑solutions into concepts. Without models, students may struggle to apply the methods.
- **No connection to hydroponics or physics**: The chapter provides generic instructions without linking to the hydroponic product. Examples should be tailored to the course project to reinforce relevance.
- **Sparse guidance on evaluating concepts**: Students are told to generate and select concepts but are not given criteria for evaluation. There is no emphasis on aligning concepts with needs, specifications, or physics considerations.
- **Ambiguous scope**: The instruction to develop classification trees “for each critical sub‑problem identified in Activity 1” presumes students recall these sub‑problems but doesn’t specify how to determine them.
- **Missing conclusion**: The chapter lacks a summary or clear transition to concept selection in Session 11.

## Action Plan (Prioritized)

### P0 – Must fix
- **Add learning objectives and introduction**: Begin with a section explaining the goal of concept generation (to explore a broad solution space before converging) and list objectives such as (1) creating concept classification trees, (2) developing concept combination tables, and (3) generating rough product concepts.
- **Define key tools**: Introduce concept classification trees (hierarchical diagrams that categorise solution approaches) and concept combination tables (matrices that systematically combine sub‑solutions). Provide concise definitions and situate them within the broader design methodology.
- **Provide relevant examples**: Include a hydroponic example of a classification tree (e.g., categories for nutrient delivery methods, structural designs, monitoring technologies) and a corresponding combination table. This will help students see how to organise sub‑solutions.
- **Clarify linkage to previous session**: Explain that the critical sub‑problems and solutions identified in Session 9 form the building blocks for the classification tree and combination table. Remind students to refer to their functional decomposition and solution lists.
- **Specify deliverable formats**: Provide templates for classification trees (e.g., using lists or diagrams) and combination tables (e.g., a table with sub‑problems as columns and solutions as rows). Instruct students on how to document the resulting rough concepts (short descriptions with key features, technology, functionality, and uniqueness).

### P1 – Should improve
- **Enhance structure and readability**: Divide the chapter into clear sections such as “Purpose and Learning Objectives,” “Concept Classification Trees,” “Concept Combination Tables,” “Generating Rough Product Concepts,” and “In‑class Consolidation.” Provide explanatory text before listing procedures.
- **Incorporate physics and hydroponic context**: Frame examples around the hydroponic system, such as categorising nutrient delivery mechanisms (gravity‑fed, pump‑driven), structural materials (PVC, stainless steel), or monitoring technologies (electrical conductivity sensors, pH sensors). Highlight relevant physics principles (fluid flow, materials properties) to engage Industrial Physics students.
- **Offer guidance on evaluating concepts**: Suggest criteria such as feasibility, degree of innovation, alignment with customer needs, compliance with specifications, and incorporation of physics principles. Provide a simple scoring rubric.
- **Encourage creativity and diversity**: Remind students to explore a wide range of possibilities (including unconventional ideas) and to combine sub‑solutions in novel ways. Mention creativity techniques like morphological charts (which are concept combination tables) or brainstorming variations.
- **Clarify the scope of concept generation**: Explain that students should generate at least five distinct concepts and that each concept should integrate sub‑solutions covering all major functions.
- **Add a conclusion**: Summarise the outputs (classification trees, combination tables, concept descriptions) and explain that these will be evaluated and screened in Session 11.

### P2 – Nice to have
- Provide optional exercises on drawing classification trees using software (e.g., draw.io, PowerPoint) or LaTeX (`tikz`).
- Suggest reading on creative design methods, such as morphological analysis and TRIZ, to broaden students’ concept generation skills.
- Encourage peer feedback or group critique sessions to improve concept diversity and quality.
- Include a brief discussion on how classification trees can reveal gaps in solution space that warrant further research.

## Concrete Rewrite Instructions

### Chapter text
- **Insert an introductory section** that explains the role of concept generation in the product development process, lists learning objectives, and connects the activities to previous sessions (functional decomposition and specifications).
- **Define concept classification trees**: Explain that they organise solution principles hierarchically, with categories at the upper levels and specific solutions at the leaves. Provide a hydroponic example: a tree categorising nutrient delivery systems (e.g., passive drip, active pump, aeroponic misting) and support structures (e.g., vertical towers, horizontal pipes).
- **Define concept combination tables**: Explain that they systematically explore combinations of solutions across sub‑problems. Provide a template table with sub‑problems (from the functional decomposition) as columns and a list of alternative solutions for each. Show how to create new concepts by selecting one solution from each column and combining them.
- **Rewrite Activity 1** to instruct students to develop classification trees and combination tables for each critical sub‑problem identified in Session 9. Include guidance on categorising solutions by functionality, cost, feasibility, and user experience. Provide an example classification tree and combination table for a hydroponic nutrient delivery problem.
- **Rewrite Activity 2** to guide students in generating rough product concepts. Instruct them to (1) select combinations of sub‑solutions from the table, (2) describe each concept’s key features (technology, functionality, unique selling points), and (3) provide at least five concepts. Encourage creativity and ensure that each concept addresses all major functions.
- **Rewrite in‑class activities** to have teams compare and merge classification trees and combination tables, then jointly select the top five concepts based on defined criteria. Provide a simple scoring rubric with criteria such as feasibility, novelty, alignment with needs and specifications, and physics integration.
- **Add a concluding paragraph** summarising tasks and pointing ahead to concept selection and scoring in Session 11.

### Activities / exercises
- **Classification tree template**: Provide a simple LaTeX or diagram template with three or four levels of hierarchy. Suggest using `\begin{tikzpicture}` or nested lists.
- **Combination table template**: Provide a table with sub‑problems as column headers and rows listing alternative solutions. Include instructions for generating combinations.
- **Concept description template**: Provide a form or table with fields for concept name, selected sub‑solutions, brief description, key technologies, unique features, and potential advantages/disadvantages.
- **Scoring rubric**: Offer criteria and a scale (e.g., 1–5) for evaluating concepts. Encourage teams to discuss and agree on scores during the in‑class session.

### Figures / tables / captions
- **Include a hydroponic classification tree figure**: Create an example classification tree for a hydroponic system and label it clearly (e.g., “Example Classification Tree for Hydroponic Nutrient Delivery System”). Use a descriptive caption.
- **Include a combination table example**: Provide a simple table with sub‑problem categories and candidate solutions. Include a caption explaining how combinations generate different concepts.
- **Standardise formatting**: Use consistent column widths, clear headings, and wrap long text with `p{}` columns. Label all figures and tables with `\label{}` and refer to them in the text.

### Terminology / notation / units
- Define “concept classification tree,” “concept combination table,” and “rough product concept.” Ensure consistent use of terms.
- Use consistent numbering and naming for activities and concepts. Avoid abbreviations without explanation.
- When describing technologies, use appropriate technical terms and units where relevant (e.g., flow rate in L/min for pumps).

### LaTeX / implementation notes
- Use `\chapter{Concept Generation}` at the beginning. Add `\section` and `\subsection` headings for each major part.
- If including diagrams, either create them in LaTeX (e.g., `tikz`) or include external images. Ensure diagrams compile correctly and are sized appropriately.
- Provide templates in separate input files or inline in the chapter. Test compile after inserting tables and diagrams to ensure layout correctness.
- Remove placeholder or irrelevant comments.

## Readability & Pedagogy Checklist
✓ learning objective is clear
✓ terms are defined before use
✓ examples match student level and physics context
✓ transitions are explicit
✓ tasks are scannable and structured
✓ chapter closes with summary/review
✓ figures/tables help more than they distract

## Notes for the Implementation Agent
- Coordinate with the Session 9 editor to ensure that the sub‑problems and solutions identified there are used consistently here. Maintain numbering and terminology across chapters.
- When creating example classification trees and tables, ensure that they reflect the hydroponic product and highlight relevant physics principles (e.g., fluid dynamics, materials properties).
- Update or create glossary entries for “classification tree,” “combination table,” “concept,” and any new terms introduced. Check that cross‑references are correct.
Improvement note: Improved clarity, corrected grammar, refined structure, resolved consistency issues, and applied the editor’s checklist recommendations. Added explicit concept-generation objectives, defined the core tools, inserted hydroponic examples and templates, and clarified the shortlist criteria for Session 11.
