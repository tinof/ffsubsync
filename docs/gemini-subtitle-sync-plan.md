# Gemini-assisted subtitle sync plan

## Goal

Use Gemini as an optional semantic validation and cleanup layer when syncing an
external subtitle against an embedded subtitle in another language. The model
should help identify comparable dialogue anchors, not replace the deterministic
timing algorithm.

## Why this helps

Timing-only subtitle alignment assumes that subtitle visibility patterns are
similar enough across languages. That often works, but it can be confused by:

- different line splitting between languages
- SDH or closed-caption-only cues
- forced-only or sparse subtitle tracks
- speaker labels and sound effects
- subtitle streams from a different cut

Gemini can compare nearby English and Finnish subtitle groups and decide whether
they represent the same dialogue, even when line numbers do not match exactly.
For example, English line 3 may map to Finnish line 2, or English lines 3-5 may
map to Finnish lines 2-3.

## Proposed pipeline

1. Parse the external subtitle and embedded reference subtitle.
2. Normalize both tracks:
   - remove speaker labels where safe
   - remove SDH cues, music notes, and sound-effect captions
   - merge very short adjacent captions into utterance groups
   - preserve original line IDs and timing metadata
3. Run existing timing-based alignment to get a rough offset or scale.
4. Build local comparison windows using that rough alignment.
5. Ask Gemini to map subtitle groups across languages and return structured JSON.
6. Keep only high-confidence semantic matches as timing anchors.
7. Fit offset, scale, or piecewise timing transform from those anchors.
8. Validate anchor consistency.
9. Fall back to normal audio or subtitle timing sync if the semantic gate fails.

## Gemini role

Gemini should return structured anchor mappings, not rewritten subtitles.

Example response shape:

```json
{
  "usable_for_sync": true,
  "matches": [
    {
      "reference_ids": [3, 4, 5],
      "target_ids": [2, 3],
      "confidence": 0.94,
      "relationship": "same_dialogue",
      "notes": "same exchange, split differently"
    }
  ],
  "rejected_reference_ids": [8],
  "warnings": ["reference contains SDH cues"]
}
```

Use Gemini's structured JSON output support with a schema so the pipeline does
not parse free-form prose. Official reference:
https://ai.google.dev/gemini-api/docs/structured-output

## Anchor validation

Accept Gemini anchors only when all of these hold:

- enough matches are returned across the episode, not only one scene
- each accepted match has high confidence
- timing residuals fit one coherent offset or scale, or a small number of
  piecewise segments
- matched anchors are monotonic: later source lines map to later target lines
- the final scale is plausible unless there is strong evidence otherwise
- SDH-heavy, forced-only, commentary, or sparse tracks are rejected

## Suggested CLI behavior

Keep this optional at first:

```bash
ssync --semantic-reference gemini video.mkv
```

Automatic use can come later when:

- an embedded subtitle reference exists
- normal embedded-subtitle timing alignment looks suspicious
- `GEMINI_API_KEY` is set
- the subtitle file is long enough to justify the API call

## Privacy and cost

Send small comparison windows, not full subtitle files, when possible. Redact or
skip if the user disables network calls. Cache Gemini results by subtitle file
hash, model name, prompt version, and window IDs to avoid repeated costs.

## Implementation sketch

- Add a `semantic_sync.py` module for grouping, prompt creation, schema parsing,
  and anchor validation.
- Add tests with fake model responses first.
- Add a provider interface so Gemini is one implementation, not hard-wired
  throughout sync logic.
- Add a CLI flag and clear logs explaining whether semantic sync was used,
  rejected, or skipped.
