# Command entry ownership

- Add commands only in the declared domain directories; no Python launchers in scripts/ root.
- Name commands verb_noun.py. Add each entry to catalog.json with purpose, implementation and lifecycle status.
- Keep model/data algorithms, public contract checks and shared publication code in their owning src package. Import it normally; do not import or exec another script by path.
- Existing substantial validation/build commands may own command orchestration. New reusable logic belongs in src.
- legacy/reports is retained historical tooling, not the default UI or a place to implement new features.
- Write new run outputs under exp/<run-id>. Never embed credentials, private test details, checkpoint copies or generated data in scripts.
- Preserve the user's stopped navtrain student-scoring workflow; do not reintroduce a bulk scoring launcher.
