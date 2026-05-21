# ch08 Template Document Generation Image Sync QA

Date: 2026-05-21

## Scope

Regenerated and promoted the six images used by the ch08 template document generation page after the example changed from classroom discount courseware to a support-operations refund escalation SOP:

- `ch08-template-schema-to-render-map.webp`
- `ch08-template-schema-to-render-map-en.webp`
- `ch08-template-schema-to-render-map-ja.webp`
- `ch08-template-payload-render-result-map.webp`
- `ch08-template-payload-render-result-map-en.webp`
- `ch08-template-payload-render-result-map-ja.webp`

## Checks

- All accepted candidate PNGs were generated at `1024x1792`.
- All promoted WebP files report `1024x1792`.
- Visual contact-sheet review passed for layout, language fit, and lack of stretched or compressed composition.
- OCR/targeted scan found no old `Discount`, `Upper elementary`, `Classroom`, `courseware`, `teaching_goal`, `concept_block`, `example_block`, `exercise_block`, `Knowledge Review`, or `Example Explanation` residues.
- The first Japanese schema candidate was rejected for English explanatory labels and regenerated with Japanese step labels before promotion.
- Final images match the current page terms: `document_goal`, `policy_block`, `case_block`, `checklist_block`, `Refund Escalation SOP`, and the localized zh-Hans/ja equivalents.

## Commands

```bash
python3 scripts/generate_course_images.py --only ch08-template-schema-to-render-map-en.png ch08-template-schema-to-render-map.png ch08-template-schema-to-render-map-ja.png ch08-template-payload-render-result-map-en.png ch08-template-payload-render-result-map.png ch08-template-payload-render-result-map-ja.png --output-dir tmp/ch08-template-doc-sync-20260521 --report-dir reports/course-images/ch08-template-doc-sync-20260521 --parallel-per-key --continue-on-error --overwrite
python3 scripts/generate_course_images.py --only ch08-template-schema-to-render-map-ja.png --output-dir tmp/ch08-template-doc-sync-20260521 --report-dir reports/course-images/ch08-template-doc-sync-20260521 --continue-on-error --overwrite
cwebp -q 90 tmp/ch08-template-doc-sync-20260521/<stem>.png -o static/img/course/<stem>.webp
```
