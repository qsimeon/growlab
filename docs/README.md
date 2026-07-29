# docs/ — accumulated reasoning

Explanations, design rationale and insights as they were worked out, kept so the
eventual write-up doesn't have to reconstruct them from memory.

| file | what it holds |
|---|---|
| `methodology.md` | Why FLOPs and not wall-clock. Why "fix the LR" was wrong. What to control vs tune. |
| `architecture.md` | How the pieces fit: grow/ → Autolab → data.json → Maritime. What each platform does and does not do. |
| `provenance.md` | Where the idea came from, why one-shot expansion died, why growth escapes it. |
| `audit.md` | The adversarial review: every claim tested, what survived, what died, verified numbers. |
| `open-questions.md` | What we still don't know, ranked by how much it would change the story. |

**Convention:** when a substantial explanation gets written in conversation, it
lands here rather than evaporating. Each file is standalone — no cross-file
context needed to read one.
