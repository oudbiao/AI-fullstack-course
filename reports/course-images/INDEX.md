# Course Image QA Report Index

This index tracks course-image QA reports that are useful for future maintenance. Keep this file short: one row per accepted report or major image batch. Candidate images should remain in `tmp/` until QA passes; official Markdown references should stay WebP-only.

| Report | Scope | Status |
|---|---|---|
| [ch06 vertical refine](ch06-vertical-refine/qa-report.md) | Chapter 6 vertical teaching image refinement | Accepted and promoted |
| [ch07 vertical refine](ch07-vertical-refine/qa-report.md) | Chapter 7 vertical teaching image refinement | Accepted and promoted |
| [ch08 vertical refine](ch08-vertical-refine/qa-report.md) | Chapter 8 vertical teaching image refinement | Accepted and promoted |
| [ch09 vertical refine](ch09-vertical-refine/qa-report.md) | Chapter 9 vertical teaching image refinement | Accepted and promoted |
| [intro career refine](intro-career-refine/qa-report.md) | Intro career-transition route image | Accepted and promoted |
| [ch02/ch03 content-image sync](content-image-sync-20260521-final/qa-report.md) | Python AI API role and database design image sync | Accepted and promoted |
| [ch03 Python DB sync](ch03-python-db-content-sync-20260521/qa-report.md) | Python database workflow support-ticket image sync | Accepted and promoted |
| [ch07 embedding sync](ch07-embedding-sync-20260521/qa-report.md) | Embedding result-map refresh | Accepted and promoted |
| [ch08 template document sync](ch08-template-doc-sync-20260521/qa-report.md) | SOP template document generation images | Accepted and promoted |
| [ch08 SOP assistant sync](ch08-sop-assistant-sync-20260521/qa-report.md) | SOP document assistant workflow and production-line images | Accepted and promoted |
| [ch08 RAG foundation SOP sync](ch08-rag-foundation-sop-sync-20260521/qa-report.md) | SOP-oriented RAG foundation images | Accepted and promoted |
| [ch08 SOP app/engineering sync](ch08-sop-app-eng-sync-20260521/qa-report.md) | SOP app-dev and engineering images | Accepted and promoted |
| [ch09 agent SOP/status sync](ch09-agent-sop-status-sync-20260521/qa-report.md) | Agent SOP/status image and content sync | Accepted and promoted |
| [ch13 final image QA](ch13-open-source-llm-final/qa-report.md) | Chapter 13 open-source LLM image set | Accepted and promoted |
| [ch13 route refine](ch13-overview-route-refine/qa-report.md) | Chapter 13 overview route refinement | Accepted and promoted |
| [ch13 compute routes refine](ch13-compute-routes-refine/qa-report.md) | Chapter 13 local CPU, free Colab, rented GPU route image | Accepted and promoted |
| [ch13 image dedup final](ch13-image-dedup-final/qa-report.md) | Chapter 13 runtime-decision and study-checklist image deduplication | Accepted and promoted |

## Maintenance checklist

Before promoting a new image batch, confirm: dimensions fit the lesson, text is readable, locale language is correct, no fake metrics/secrets/brands are invented, the concept matches nearby code/output, Markdown references are WebP-only, and `python3 scripts/validate_course_image_refs.py` passes.
