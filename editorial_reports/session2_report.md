# Revision Report – Session 2 / Stage Gate and Development Processes

## Chapter Overview
| Item | Description |
|---|---|
| **Chapter file(s)** | `session2/session2.tex` |
| **Related/shared files** | Relies on global macros and classes in `main.tex`. The long table of process types uses the `longtable` environment, which is defined in the preamble. |
| **Current chapter purpose** | To introduce students to the stage‑gate approach for product development, guide them in choosing a suitable product development process, process flow, departmental structure and organizational structure for their project, and provide a reference table summarising different process types with features and examples. |
| **Intended student takeaway** | Students should understand the differences between generic, technology‑push, platform, process‑intensive and other product development processes; be able to select the most appropriate process and process flow for their chosen product; appreciate the interdisciplinary nature of product development teams; and choose an organisational structure that fits their project goals. |
| **Audience‑specific risks** | The chapter presents multiple complex tasks without clear learning objectives. The instructions are repetitive and occasionally grammatically incorrect. Key terms such as “development process flow,” “departmental structure” and “organizational structure” are not defined or contrasted. The long table of process types is cramped and may not be scannable. Third‑year Industrial Physics students may struggle to connect the process categories to their hydroponics projects without contextual examples. |

## Relevant Editorial Feedback (Mapped)
- **Missing learning objectives** – As in the previous session, the chapter lacks an overview explaining why selecting a process and structure matters and what students will achieve.
- **Dense instructions and repetition** – Activities are described in long paragraphs with redundant phrases (“consider the characteristics and requirements of your product…” appears multiple times). This makes it hard to discern the essential tasks.
- **Unclear terminology** – Terms such as “generic process,” “spiral product development process” and “complex system development process” are listed without definition or examples. Similarly, “departmental structure” and “organisational structure” are used interchangeably.
- **Lack of connection to physics and hydroponics** – The process categories are listed in the abstract; there is no discussion of how certain processes might suit products in hydroponic agriculture or how applied physics principles influence the choice (e.g., technology‑push may involve novel sensor technologies).
- **Overly wide table** – The `longtable` listing process types spans four columns, requiring landscape orientation. Some descriptions are verbose; the table could be simplified or broken into separate tables for readability.
- **Grammar and punctuation issues** – Examples include “The objective of this activity is to help you in understanding different types…” (use “help you understand”), missing articles and inconsistent capitalization.
- **Poor signposting** – Subsections jump from selecting process types to organisational structure without transitions or explanation of how the decisions interact.

## Action Plan (Prioritized)

### P0 – Must fix
- **Introduce learning objectives** at the start. For example: “By the end of this session you should be able to: (1) identify different product development processes and select an appropriate one for your project; (2) choose a development process flow that matches your product’s complexity; and (3) propose a departmental and organisational structure that supports your development strategy.”
- **Define key terms** – Provide brief definitions of product development processes (market‑pull, technology‑push, platform, etc.), development process flows (generic, spiral, complex) and organisational structures (functional, project, lightweight, heavyweight). Cite examples relevant to hydroponics.
- **Clarify distinctions** between “departmental structure” (the combination of marketing, engineering, quality assurance etc.) and “organisational structure” (functional vs. project vs. matrix). Use headings to separate these concepts.
- **Fix grammar and punctuation** across the chapter; rewrite sentences to be clear and concise.

### P1 – Should improve
- **Reorganise content into logical subsections** – For example: Introduction and learning objectives; Overview of development processes; Selecting a process for your product; Understanding development process flows; Choosing a departmental structure; Choosing an organisational structure; In‑class discussion and decision making.
- **Provide context and examples** – For each process type, briefly explain when it is appropriate and give an example relevant to hydroponic products (e.g., “Process‑intensive products are common in nutrient solution manufacturing where the process constraints dictate product characteristics”). For each organisational form, discuss pros and cons.
- **Simplify the process type table** – Break the long table into two smaller tables if necessary, or use bullet lists rather than a 4‑column table. Provide only essential features and examples to improve readability. Label the table clearly and reference it in the text.
- **Add transitions** – At the end of each subsection, summarise what students should have decided and how it feeds into the next step (e.g., “After selecting your process type, you will now examine the flow and organisational structure that best support it”).
- **Use consistent voice and list formatting** – Start each instruction with a verb and maintain second‑person perspective.
- **Connect to applied physics** – Discuss how certain processes might be more suitable for products involving sensors (technology‑push) or bespoke components (customised products) and relate departmental needs to physics (e.g., instrumentation engineering). This helps students see relevance.

### P2 – Nice to have
- **Include a flowchart** depicting the stage‑gate process and where the session’s decisions fit within the overall product development timeline.
- **Provide sample selection criteria** or a decision matrix for choosing a process, process flow and organisation. Encourage groups to justify their choices in a structured way.
- **Offer case studies** of real products that followed different development processes, particularly in agriculture or industrial design.

## Concrete Rewrite Instructions

### Chapter text
- **Begin with a “Session Overview”** describing why selecting a product development process is critical and how it influences downstream design decisions.
- **Define each process type** in a concise paragraph. For example: “Generic or market‑pull products start with a market opportunity and select technologies to meet customer needs. Examples include sporting goods and furniture.” Use similar format for other types. Consider moving the table to an appendix if it is too detailed.
- **Introduce development process flows** (generic, spiral, complex) with definitions and explain differences. Provide one example for each flow.
- **Explain departmental and organisational structures separately**. Use subheadings and define functional, project, lightweight and heavyweight structures. Provide guidance on when to choose each, considering team size and project complexity.
- **Rewrite activity instructions** to be concise and action‑oriented. For example: “Identify which development process aligns with your product’s characteristics by comparing your project to the descriptions above. Document your choice and justify it in one paragraph.”
- **Add a “Summary and Next Steps” section** at the end instructing students to consolidate their decisions (process type, flow, departmental and organisational structures) and prepare to discuss them with their group.

### Activities / exercises
- **Combine repetitive activities** – For example, activities 1 and 2 both involve selecting processes. Consolidate into a single task with clear steps: research process types, match to your product, and document your decision.
- **Add guidance on research** – Suggest sources (textbook exhibits referenced in the original text) or additional reading to help students differentiate processes.
- **Encourage justification** – Require students to provide reasons for their choices based on project goals, market demands and technical considerations.

### Figures / tables / captions
- **Redesign the process type table** – Limit each row to essential information: process type, one‑sentence description and one example. Use a caption (“Table X: Summary of product development process types”). Place the table near the discussion of process types.
- **Include a decision matrix template** as a figure or table to help students evaluate and compare options.

### Terminology / notation / units
- **Explain specialised terms** (e.g., “spiral process”) on first use. Include them in a glossary if available.
- **Use consistent notation** when referring to process types (capitalise names and use the same order throughout the chapter).

### LaTeX / implementation notes
- **Ensure the `longtable` environment** is necessary; if not, switch to `tabularx` for better fit. Avoid landscape orientation unless absolutely required.
- **Add labels and captions** to all tables for cross‑referencing. Use `\ref` in the text when discussing them.
- **Check line breaks** within table cells to prevent text from overflowing outside column borders.

## Readability & Pedagogy Checklist
- [ ] learning objective is clear
- [ ] terms are defined before use
- [ ] examples match student level
- [ ] transitions are explicit
- [ ] tasks are scannable
- [ ] chapter closes with summary/review
- [ ] figures/tables help more than they distract

## Notes for the Implementation Agent
- Coordinate with the session covering functional decomposition (Session 9) to ensure that process and organisational terminology are used consistently. Consider adding the selected process type and structure to a shared project tracking document.
- When editing the table, ensure that LaTeX column widths are appropriate and avoid splitting rows across pages unless using `longtable` properly. Test compile to check table layout.