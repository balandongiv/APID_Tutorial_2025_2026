# Revision Report – Session 5 & 6 / Product Planning

## Chapter Overview
| Item | Description |
|---|---|
| **Chapter file(s)** | `latex/session5_and_6/session5_and_6.tex` |
| **Related/shared files** | Shared LaTeX preamble (`latex/latex/preamble.tex`), `main.tex`, macro definitions, and common figures such as benchmarking tables and charts |
| **Current chapter purpose** | This two‑session chapter guides students through early product planning. Pre‑class activities ask students to perform market analysis (segments, technology integration, manufacturing and service considerations, target market and price, assumptions/constraints) and draft an initial mission statement. In class, teams refine their analyses and mission statements, culminating in a unified plan. An optional “Remarks” section encourages deeper justification and citation of sources. |
| **Intended student takeaway** | Students should learn to systematically analyse markets and constraints for a chosen product, create a clear mission statement capturing the product’s purpose and value proposition, and collaboratively refine these planning elements. |
| **Audience‑specific risks** | Without explicit learning objectives and context, Industrial Physics students may not see the relevance of market analysis or mission statements to physics‑based design. The current draft lacks definitions, examples, and scaffolding. Many instructions are densely enumerated without explanation, making cognitive load high. There is no link to hydroponic or physics concepts, and the optional remarks section could confuse students without guidelines on research quality. |

## Relevant Editorial Feedback (Mapped)
- **No learning objectives or overview**: The chapter begins with instructions but does not explain why market analysis and mission statements matter in product development. Explicit learning objectives should be added to orient students【708506183906356†L160-L169】.
- **Undefined key terms**: Terms such as “market segment,” “technology integration,” “mission statement,” and “stakeholders” are used without definition. Provide concise definitions or point to a glossary.
- **Weak connection to Applied Physics**: Activities are generic and do not relate examples to hydroponic systems or industrial physics. Illustrate how physics principles influence product planning (e.g., physical limitations affecting manufacturing, thermal considerations shaping technology integration).
- **Overly long enumerations**: The list of questions under market analysis is dense and lacks substructure. Breaking the list into subheadings and adding explanatory context will improve readability.
- **Lack of templates and examples**: Students are told to “document your initial thoughts” and “draft a preliminary mission statement” but are not given templates. Provide sample tables or forms for market analysis and mission statements to ensure consistency and reduce ambiguity.
- **Unclear evaluation criteria**: There is no guidance on how detailed or evidence‑based the analyses and mission statements should be, or how they will be assessed. Introduce criteria such as relevance to the innovation charter, alignment with customer needs, feasibility, and justification with credible sources.
- **Optional remarks section is ambiguous**: The “Remark (Optional)” heading appears within the main text and reads like an aside. It needs clearer placement, consistent formatting, and explicit instructions on citation format and research quality.
- **Grammatical and stylistic issues**: Phrases such as “be prepared to share and discuss your analysis with your team members” should use present tense (“prepare to share…”). The final line “Im here just to make sure the Session running number is tally…” is unprofessional and should be removed.
- **Missing conclusion**: The chapter lacks a summary that ties together market analysis and mission statement creation, and does not preview how the outputs feed into subsequent sessions.

## Action Plan (Prioritized)

### P0 – Must fix
- **Add learning objectives and context**: Begin with a short introduction explaining that product planning translates market and technical insights into a clear mission statement guiding development. List specific objectives, e.g., “Identify key market segments and constraints,” “Draft a mission statement that aligns with customer needs and business goals.”
- **Define terms and include a glossary**: Provide brief definitions for market segmentation, technology integration, manufacturing/service objectives, assumptions/constraints, mission statement, stakeholders, and target market. Link to a global glossary for consistency.
- **Provide templates and examples**: Include a market analysis table (columns for segment, characteristics, size, needs, competition, technology opportunities, constraints) and a mission statement template (e.g., one‑paragraph format or table fields). Example entries should relate to the hydroponic product chosen in Session 1.
- **Clarify evaluation criteria**: Explain that each analysis should be evidence‑based and aligned with the innovation charter and customer needs. Encourage citing credible sources for market data or trends. Provide a rubric or set of questions to evaluate the mission statement (e.g., Does it describe the product? highlight benefits? state primary market? reflect constraints?).
- **Remove extraneous text**: Delete “Im here just to make sure the Session running number is tally as what I disclose in ITEL” and reposition the optional remarks section as a clearly labelled subheading.

### P1 – Should improve
- **Improve structure and readability**: Organise the chapter into subsections with explanatory text before lists. For example, before listing questions for market analysis, explain why each element matters (e.g., technology integration influences feasibility and cost).
- **Incorporate physics context**: Provide at least one example where physical principles influence product planning. For instance, discuss how heat transfer constraints or fluid dynamics impact hydroponic system design and should be considered in manufacturing and service objectives.
- **Add illustrative examples**: Provide example market segments (e.g., urban household gardeners vs. commercial greenhouse operators), example technologies (e.g., IoT sensors), and sample mission statements to guide students.
- **Emphasise collaborative skills**: In the in‑class activities, suggest techniques for reaching consensus (e.g., nominal group technique, weighted voting) and emphasise the value of integrating diverse perspectives.
- **Explicitly describe optional remarks**: Move the optional remarks into a dedicated subsection “Optional Remarks for Extra Credit” with bullet points guiding students to provide deeper analysis, cite sources, and explore industry trends. Specify citation format (e.g., APA) and emphasise the importance of credible sources.
- **Add a conclusion**: Summarise key tasks and explain that the final mission statement will inform subsequent sessions on customer needs and specifications.

### P2 – Nice to have
- Provide a short bibliography of recommended readings on product planning and mission statements to encourage further study.
- Suggest that students use simple market research tools (e.g., Statista, academic databases) to obtain data for the analysis.
- Include a checklist for the mission statement to ensure clarity (e.g., clear product description, value proposition, target market, business goals, constraints, stakeholders).
- Offer guidance on professional tone and conciseness when drafting mission statements.

## Concrete Rewrite Instructions

### Chapter text
- **Insert an introductory paragraph** explaining the role of product planning in the product development process and listing learning objectives. For example: “Product planning translates insights from the innovation charter and opportunity identification into a structured plan guiding development. In this session you will (1) analyse market segments, technology opportunities, manufacturing and service considerations, target customers and price points, assumptions and constraints; and (2) draft a mission statement that articulates your product’s purpose, benefits, and goals.”
- **Define key terms** either inline or via a glossary. For example, “A **market segment** is a group of potential customers with common characteristics or needs” and “A **mission statement** summarises the product’s purpose, key benefits, target markets, and business goals.”
- **Restructure the market analysis activity**: Group related questions under subheadings (e.g., “Market Segments,” “Technology Integration,” “Manufacturing and Service,” “Target Market and Price,” “Assumptions, Constraints, and Stakeholders”). For each, briefly explain its importance and then list the questions. Encourage students to support their answers with data or references.
- **Provide a market analysis table template** in LaTeX, with columns for segment name, description, needs, technology opportunities, manufacturing/service considerations, price range, assumptions, constraints, and stakeholders. Include a short example row relevant to a hydroponic product.
- **Rewrite the mission statement activity**: Explain what a mission statement should contain and give a sample structure. Encourage students to write a concise paragraph that includes product description, benefit proposition, business goals, primary and secondary markets, key assumptions, constraints, and stakeholders. Suggest writing multiple drafts and refining them through peer feedback.
- **Revise the remarks section**: Create a `\subsection*{Optional Remarks (for extra credit)}` with clear guidance. Explain that students can earn extra credit by providing deeper justifications, citing credible sources, analysing risks, competitors, trends, and stakeholder priorities. Outline the citation format (e.g., APA) and emphasise critical thinking.
- **Remove filler text** (“Im here just…”) and ensure all language is in the third person and free from colloquialisms.
- **Add a conclusion** summarising the tasks, reminding students to compile their market analysis and mission statement, and previewing how these outputs will inform customer needs identification in Session 7.

### Activities / exercises
- **Clarify deliverables and deadlines**: For each pre‑class activity, specify what students need to submit (e.g., completed market analysis table and mission statement) and when. For in‑class activities, instruct teams to share individual analyses, merge insights into a collaborative version, and produce a unified mission statement.
- **Introduce evaluation criteria**: Provide a rubric or checklist for assessing the mission statement and market analysis (e.g., completeness, alignment with charter, feasibility, clarity, supporting evidence). This helps students focus their efforts.
- **Encourage evidence‑based analysis**: Recommend students consult credible sources (market reports, academic papers) for data on market size, technological trends, or competitor analysis. Provide guidelines on citing sources.
- **Suggest collaborative techniques**: Outline methods such as silent brainstorming, dot voting, or affinity mapping to help teams reach consensus during in‑class sessions.

### Figures / tables / captions
- **Add a market analysis table figure** with an informative caption (e.g., “Example Market Analysis Template for Hydroponic Product Planning”). Use the `tabular` or `longtable` environment to accommodate multiple rows.
- **Provide a mission statement template** as a table or box with labelled fields. Ensure captions explain how to use the template.
- Ensure figure/table numbering is consistent and references are correctly labelled using `\label{}` and `\ref{}`.

### Terminology / notation / units
- Standardise terms such as “target price” (use consistent units, e.g., Malaysian Ringgit) and “primary/secondary market.”
- Use consistent capitalisation (e.g., “Mission Statement” vs. “mission statement”).
- If citing market data, include units and currency symbols appropriately and consistently throughout the book.

### LaTeX / implementation notes
- Use `\chapter{Product Planning}` only once and ensure the chapter appears correctly in the table of contents.
- Add `\section`, `\subsection`, and `\subsubsection` headings with descriptive titles for each activity.
- Ensure optional sections are clearly labelled (e.g., `\subsection*{Optional Remarks (Extra Credit)}`) and excluded from the table of contents if desired.
- When adding new tables or templates, test compile with `longtable` or `tabularx` to avoid page overflow.
- Remove commented placeholders and unprofessional remarks.

## Readability & Pedagogy Checklist
- [ ] learning objective is clear
- [ ] terms are defined before use
- [ ] examples match student level and physics context
- [ ] transitions are explicit
- [ ] tasks are scannable and structured
- [ ] chapter closes with summary/review
- [ ] figures/tables help more than they distract

## Notes for the Implementation Agent
- Coordinate with the Session 1 and Session 2 editors to ensure that the chosen hydroponic product and process definitions are referenced consistently in the market analysis and mission statement.
- After adding templates, verify that column widths suit typical answers and adjust with `p{}` column specifications as needed.
- Create or update glossary entries for terms introduced in this chapter and ensure they are used consistently throughout the book.
- Check cross‑references to ensure that subsequent sessions (e.g., Session 7 on customer needs) correctly refer to the mission statement and market analysis outputs.