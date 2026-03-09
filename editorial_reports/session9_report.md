# Revision Report – Session 9 / Functional Decomposition and Solutions

## Chapter Overview
| Item | Description |
|---|---|
| **Chapter file(s)** | `latex/session9/session9.tex`, `rfm_table_landscape.tex` |
| **Related/shared files** | Shared preamble and macros, global glossary, and any template diagrams referenced across sessions |
| **Current chapter purpose** | This chapter introduces students to functional decomposition and solution searches. Pre‑class activities instruct students to analyse a product’s primary functions, select and document an appropriate decomposition approach, and search for solutions to critical sub‑problems. In class, students collaborate to choose a decomposition approach, construct a requirement–function matrix, and compile solutions. |
| **Intended student takeaway** | Students should learn to break down a complex product into functional elements, document the rationale for choosing a decomposition method, represent functions using diagrams or trees, and gather potential solutions for each sub‑problem using external research. |
| **Audience‑specific risks** | The current chapter lacks learning objectives, definitions, and context linking functional decomposition to product development. Instructions are verbose and unstructured. Examples (e.g., planting seed flowchart and seed planter system) are not tailored to the hydroponic project and may confuse readers. There is no guidance on evaluating or selecting solutions. The requirement–function matrix is referenced but not explained. |

## Relevant Editorial Feedback (Mapped)
- **Missing learning objectives and context**: The chapter does not explain why functional decomposition matters or how it connects to earlier needs and specification activities. Adding learning objectives and context will help students appreciate the purpose of this session【708506183906356†L160-L169】.
- **Undefined decomposition approaches**: Functional decomposition, decomposition by sequence of user actions, and decomposition by key customer needs are mentioned without definition. Students need descriptions and examples to choose appropriately.
- **Unclear instructions and grammar**: Sentences like “Make sure to use VERB when expressing the functional element of the individual operation and operation” are confusing. Clearer wording is needed. The steps should emphasise using generic verb–noun pairs (e.g., “move water,” “extract coffee”) and avoiding implementation details【979038334412772†L77-L121】.
- **Examples not aligned with course project**: The seed planter examples are unrelated to hydroponic product development. Replace them with a hydroponic example or general abstract example. If external examples are kept, explain their relevance.
- **Lack of templates for documentation**: Students are told to document their chosen decomposition approach and create diagrams or flowcharts but are not provided with templates or guidelines for the diagrams. Similarly, the requirement–function matrix is referenced via an input file but not explained.
- **No guidance on solution search**: The search activity instructs students to find five solutions for each sub‑problem but does not explain how to assess solution quality or document sources. Students need criteria and examples.
- **Missing conclusion**: There is no summary section tying decomposition and solution search back to the overall development process or indicating how outputs feed into concept generation (Session 10).

## Action Plan (Prioritized)

### P0 – Must fix
- **Add learning objectives and introduction**: Begin by explaining that functional decomposition helps designers understand how a product works by breaking it into smaller, manageable functions. Objectives should include (1) selecting an appropriate decomposition method, (2) documenting the product’s functional hierarchy using diagrams and verb–noun pairs, and (3) gathering multiple solution ideas for critical functions.
- **Define decomposition approaches**: Provide concise definitions for functional decomposition, user‑action sequence decomposition, and customer‑needs decomposition. Explain when each might be appropriate and refer to the global glossary.
- **Emphasise proper functional description**: Instruct students to describe functions generically using verb–noun pairs and to avoid premature implementation details. Cite guidelines that functional decomposition breaks a complex task into smaller tasks using generic verbs and that mixing implementation details should be avoided【979038334412772†L77-L121】.
- **Provide templates and examples**: Offer a template for documenting the chosen decomposition method and a blank function tree diagram. Provide a hydroponic example (e.g., breaking down nutrient delivery in a hydroponic system) to illustrate how to use verb–noun pairs.
- **Explain the requirement–function matrix**: Introduce the matrix conceptually (a table mapping customer requirements to product functions) and provide a template. Explain how to populate it using the needs identified in Session 7 and the functions derived here.
- **Clarify solution search criteria**: Instruct students to evaluate solutions based on feasibility, alignment with functional requirements, technical maturity, and resource requirements. Provide a table template for recording solutions with source references and brief descriptions.

### P1 – Should improve
- **Improve organisation and readability**: Structure the chapter into sections: “Purpose and Learning Objectives,” “Selecting a Decomposition Approach,” “Documenting Functional Decomposition,” “Requirement–Function Matrix,” and “Searching for Solutions.” Provide short explanatory text before enumerated steps.
- **Replace or contextualise seed planter example**: Either replace the planting flowchart and seed planter tree with a hydroponic example (e.g., nutrient supply system) or clearly label them as generic examples. Explain how they illustrate the decomposition process.
- **Provide diagramming guidelines**: Suggest tools (e.g., block diagrams, hierarchical trees) for representing functions. Describe conventions such as top‑level function at the root, sub‑functions as branches, and naming with verb–noun pairs.
- **Encourage use of credible sources**: Emphasise that solution search should draw from academic papers, patents, industrial reports, and credible web sources. Provide guidelines on citing sources and summarising key information (author, year, source, description).
- **Explain requirement–function matrix usage**: Describe how the matrix helps ensure functions cover all customer requirements and identify gaps. Encourage students to iterate on their decomposition if requirements are not met.
- **Add a conclusion**: Summarise the session’s outcomes and explain how the decomposed functions and solution options will feed into concept generation in Session 10.

### P2 – Nice to have
- Provide optional exercises asking students to decompose a familiar device (e.g., an electric kettle) to practice using verb–noun pairs.
- Suggest using software (e.g., Lucidchart, draw.io) for creating clear diagrams.
- Include a brief discussion on how functional decomposition can reveal opportunities for innovation by identifying overlooked sub‑functions.
- Encourage peer feedback on decomposition diagrams to improve clarity and completeness.

## Concrete Rewrite Instructions

### Chapter text
- **Add an introductory section** that defines functional decomposition and explains its role in turning customer needs and specifications into design concepts. State learning objectives explicitly.
- **Define the decomposition approaches** in a new subsection. For example:
  - *Functional Decomposition*: Breaks the product into its basic functions, described by verb–noun pairs (e.g., “move water,” “control temperature”).
  - *Decomposition by Sequence of User Actions*: Describes the product in terms of the sequence of user interactions.
  - *Decomposition by Key Customer Needs*: Groups functions according to major customer needs.
  Provide guidance on selecting an approach based on project context.
- **Rewrite Activity 1 (Decompose the Problem)** to include clear steps: (1) identify the product’s main function, (2) list major sub‑functions using verb–noun pairs, (3) choose a decomposition approach and justify the choice, (4) create a function diagram, and (5) write brief descriptions of how each function operates. Emphasise that descriptions should remain technology‑agnostic【979038334412772†L77-L121】.
- **Insert a hydroponic example** illustrating functional decomposition for a nutrient delivery system (e.g., “store nutrient solution,” “pump solution,” “monitor nutrient levels,” “deliver oxygen”). Use this example to demonstrate verb–noun notation.
- **Rewrite Activity 2 (Search and Find Solutions)** to provide criteria for solution quality and a template for documenting solutions. Ask students to list at least five potential solutions for each critical sub‑function, including source references (author, year, publication) and a brief description. Encourage students to assess feasibility and alignment with customer needs.
- **Introduce the requirement–function matrix** in a new subsection. Explain its purpose and provide a template with rows for requirements (from Session 7) and columns for functions. Instruct students to mark whether each function satisfies each requirement and to identify any unmet requirements.
- **Reframe the in‑class activities** to guide teams through consolidating individual decompositions, agreeing on a single approach, and populating the requirement–function matrix together. Provide strategies for reaching consensus.
- **Remove irrelevant or confusing content**: Delete or move the seed planter flowchart and tree to an appendix or replace with a hydroponic example. Clarify grammar and remove phrases like “Use appropriate diagrams, such as function diagrams or flowcharts, to visually represent the decomposition process and the relationships between different components or subfunctions” by providing a more specific description.
- **Add a closing section** summarising tasks and describing how the outputs (function diagrams, requirement–function matrix, solution list) will support concept generation and combination in Session 10.

### Activities / exercises
- **Functional decomposition template**: Provide a diagram outline (e.g., using `tikz` or a simple nested list) with space for the top‑level function and multiple sub‑functions. Include placeholders for verb–noun pairs and brief descriptions.
- **Requirement–function matrix template**: Offer a table with requirements listed vertically and functions horizontally. Provide instructions on marking intersections.
- **Solution search table**: Provide a table template with columns: Sub‑function, Candidate Solution, Source (author/year), Description, Feasibility/Notes. Encourage students to include at least five entries per critical sub‑function.
- **Consensus activity**: Suggest using sticky notes or digital tools to group similar functions and vote on decomposition approaches.

### Figures / tables / captions
- **Update or replace example diagrams**: If using an example in the chapter, ensure it is relevant to the hydroponic product. Provide descriptive captions (e.g., “Functional Decomposition of a Hydroponic Nutrient Delivery System”). Use consistent styling and verb–noun notation.
- **Label tables and diagrams** clearly and reference them in the text. Use the `\label{}` and `\ref{}` commands appropriately to avoid broken references.

### Terminology / notation / units
- Define and consistently use verb–noun pairs for functions. Provide examples.
- Standardise the naming of decomposition approaches and ensure they match the glossary.
- Use consistent formatting for lists and enumerations to aid readability.

### LaTeX / implementation notes
- Use `\chapter{Functional Decomposition and Solutions}` at the start to align with the book’s structure.
- Structure sections and subsections logically. For diagrams, consider using `tikz` or include externally generated figures. Ensure diagrams compile correctly and scale appropriately.
- Remove unused commented code and ensure that `\input{session9/rfm_table_landscape}` references an existing, updated table file. If the table is moved or replaced, update the `\input{}` accordingly.

## Readability & Pedagogy Checklist
- [ ] learning objective is clear
- [ ] terms are defined before use
- [ ] examples match student level and physics context
- [ ] transitions are explicit
- [ ] tasks are scannable and structured
- [ ] chapter closes with summary/review
- [ ] figures/tables help more than they distract

## Notes for the Implementation Agent
- Coordinate with the Session 7 editor to ensure that customer requirements used in the requirement–function matrix align with needs previously identified. Maintain consistent numbering and terminology across sessions.
- Ensure that any new diagrams are created using a consistent style and compiled correctly in LaTeX. If using external tools, save figures in a compatible format (e.g., PDF) and include them in the `session9` folder.
- Update or create glossary entries for decomposition approaches, verb–noun pairs, and requirement–function matrix. Verify cross‑references to avoid broken links.