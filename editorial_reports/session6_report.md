# Revision Report – Session 6 / Product Planning (Continuation)

## Chapter Overview
| Item | Description |
|---|---|
| **Chapter file(s)** | `session5_and_6/session5_and_6.tex` – Sessions 5 and 6 are combined in a single LaTeX source. |
| **Related/shared files** | Shared preamble (`main.tex` or `preamble.tex`), global macros, and any tables or charts used across sessions. |
| **Current chapter purpose** | Session 6 continues the product planning work begun in Session 5. It focuses on refining market analyses, aligning mission statements with customer needs, and finalising a unified plan for the selected hydroponic product. |
| **Intended student takeaway** | Students should build on their earlier research to refine market segmentation, technology integration, manufacturing/service considerations, assumptions, constraints and stakeholder analysis, and to finalise a coherent mission statement that guides subsequent design activities. |
| **Audience‑specific risks** | Without clear differentiation from Session 5, the combined chapter may confuse students about which activities belong to which session. The absence of a standalone Session 6 file obscures the continuation of product planning tasks and could result in missing guidance on refining analyses and mission statements. |

## Relevant Editorial Feedback (Mapped)
- **Combined sessions create ambiguity** – Because Sessions 5 and 6 share a single file, there is no clear hand‑off or delineation between initial planning (market analysis and draft mission statements) and refinement/feedback. Students need to know how the work should evolve in Session 6.
- **Missing learning objectives for refinement** – The chapter continues instructing students to discuss analyses but doesn’t state explicit objectives for the refinement session (e.g., synthesising individual analyses into a group document, validating assumptions, and finalising the mission statement).
- **Lack of guidance on merging work** – There is minimal direction on how to merge individual market analyses and mission statements into a coherent group plan. Students may not know how to reconcile differing opinions or quality levels.
- **No link to subsequent sessions** – The chapter does not clearly connect the refined plan to upcoming activities on customer needs (Session 7) and specifications (Session 8). Students need to understand that the mission statement and market analyses form the foundation for future work.
- **Repeated issues from Session 5** – All issues identified in the Session 5 report (missing definitions, absence of templates, lack of physics context, vague criteria, etc.) persist here. Since the file covers both sessions, the fixes recommended for Session 5 apply equally to Session 6.

## Action Plan (Prioritized)

### P0 – Must fix
- **Create a clear session division** – Insert headings or comments in the LaTeX file indicating where Session 5 ends and Session 6 begins. This clarifies the scope of each session.
- **State learning objectives for Session 6** – At the beginning of the Session 6 section, articulate that students will (1) synthesise individual market analyses and mission statements, (2) validate assumptions and data, and (3) finalise a mission statement that will guide needs identification and specifications.
- **Provide a process for merging analyses** – Describe specific steps for combining individual research: compare market segment findings, reconcile differences, integrate technology opportunities, refine assumptions/constraints, and agree on stakeholder priorities.
- **Add templates for consolidation** – Offer a group market analysis table with columns for consensus entries and notes on rationale. Provide a template for the final mission statement, encouraging concise, well‑structured prose.
- **Ensure continuity to later sessions** – Include a concluding paragraph that explains how the finalised mission statement and refined market analysis feed into Session 7 (customer needs) and Session 8 (product specifications).

### P1 – Should improve
- **Clarify roles and collaboration** – Suggest methods such as rotating facilitators or using decision matrices to resolve disagreements. Encourage equitable participation and documentation of sources.
- **Encourage verification of data** – Instruct students to cross‑check market data, technology trends, and constraints against credible sources. Recommend citing data where possible.
- **Emphasise physics context** – Remind students to relate market and technology choices to the physical principles underlying their product (e.g., fluid mechanics, material selection), reinforcing relevance to Industrial Physics students.
- **Provide a reflection prompt** – Ask students to reflect on how their understanding of the market and constraints evolved from Session 5 to Session 6 and what implications this has for the mission statement.

### P2 – Nice to have
- Offer optional peer review – Encourage groups to exchange mission statements with another team for feedback before finalising.
- Provide example mission statements – Show good and poor examples to illustrate clarity and alignment with customer needs and specifications.
- Suggest tools – Recommend collaborative tools (e.g., shared spreadsheets, Trello boards) to manage group analyses and track revisions.

## Concrete Rewrite Instructions

### Chapter text
- **Insert a clear subsection heading** (e.g., `\subsection{Refining Market Analysis and Mission Statements (Session 6)}`) at the point where Session 6 content begins. Preface this with a brief overview of the session’s objectives.
- **Rewrite the refinement instructions** to include steps: (1) review individual analyses and identify commonalities/differences; (2) discuss and agree on market segments, technology integrations, manufacturing/service considerations, pricing, assumptions and constraints; (3) finalise a mission statement that synthesises key insights; and (4) document sources and rationale.
- **Add a group market analysis table** example in the LaTeX source, similar to the template recommended in the Session 5 report but with additional columns for notes and consensus decisions.
- **Provide a mission statement template**: instruct teams to write one concise paragraph addressing product purpose, benefits, target markets, technology approach, and constraints. Encourage teams to use bullet points or a table to draft and then convert to prose.
- **Conclude the section** with a paragraph linking the outputs to subsequent chapters: explain that the refined mission statement sets the scope for customer needs in Session 7 and metrics in Session 8.

### Activities / exercises
- **Group synthesis exercise**: Describe an activity where team members take turns presenting their market analyses, with others asking questions and identifying points of agreement and disagreement. Provide guidelines on reaching consensus (e.g., prioritising criteria based on the innovation charter).
- **Mission statement refinement**: Include a short activity where teams evaluate their draft mission statements against a checklist (e.g., clarity, alignment with needs, feasibility, physics relevance) and refine accordingly.

### Figures / tables / captions
- **Market analysis consolidation table**: Suggest including a figure or table template with columns for Market Segment, Needs, Technology Opportunities, Manufacturing/Service Considerations, Pricing, Assumptions/Constraints, Stakeholders, Consensus Decision, Rationale. Use a clear caption (e.g., “Group Market Analysis Consolidation Template”).
- **Mission statement template**: Provide an example box or table with fields for product description, value proposition, target market, key technologies, and constraints. Include a caption explaining how to use the template.

### Terminology / notation / units
- Maintain consistent definitions and usage for terms like “market segment,” “assumptions,” “constraints,” and “stakeholders.” Ensure these align with the glossary and other sessions.
- Use consistent currency and units when discussing pricing and cost (e.g., Malaysian Ringgit). Define any new terms added during refinement.

### LaTeX / implementation notes
- Add the new subsection and templates within the existing `session5_and_6.tex` file. Ensure proper section numbering and placement in the table of contents.
- Use `tabularx` or `longtable` environments for the consolidation table to manage width. Test compile for page breaks.
- Label all new tables and reference them appropriately in the text.

## Readability & Pedagogy Checklist
- [ ] learning objective is clear
- [ ] terms are defined before use
- [ ] examples match student level and physics context
- [ ] transitions are explicit
- [ ] tasks are scannable and structured
- [ ] chapter closes with summary/review
- [ ] figures/tables help more than they distract

## Notes for the Implementation Agent
- Because Sessions 5 and 6 share a single file, coordinate closely with the Session 5 editor. Ensure that combined edits do not create redundancy. Mark where Session 6 begins to avoid confusion.
- When adding new templates or subsections, maintain consistency with the style used in other chapters (headings, table formatting, caption style). Test compile to ensure all new elements compile correctly.
- Ensure that refined mission statements and market analyses are accessible to later sessions by storing them in a shared folder or referencing them clearly. 