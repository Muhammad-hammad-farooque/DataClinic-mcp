"""Server instructions, sent once at initialize.

These cost nothing after the first turn and are the cheapest defence against
tool misuse. They state behaviour the tool schemas cannot express -- above all
that ``load_dataset`` already profiles, which removes the most common
redundant round trip in the workflow.

Kept byte-stable: any session state here would invalidate the client's prompt
cache on every turn (spec section 10.3).

See spec section 7.6.
"""

INSTRUCTIONS = """\
This server performs exploratory data analysis on files and databases.

Choosing a tool:
- load_dataset already returns a profile and top findings. Do not call
  profile immediately after it.
- For database tables, call profile directly -- it computes in SQL without
  transferring rows. Only load_dataset when you need row-level operations.
- find_issues returns a recommended fix with each problem. Do not ask which
  fix to apply; apply it or explain why not.
- clean_data and transform_data take a list of operations. Batch them into
  one call rather than calling repeatedly.

Reading results:
- Statistics cover the full dataset unless a "sampled" field is present.
- "resources" lists URIs holding full detail; read one only if the digest is
  insufficient.
- "truncated" states what was omitted and how to retrieve it.

Writing:
- Sources are never modified in place. export writes elsewhere.
- Mutations are reversible with history(action="undo").
"""
