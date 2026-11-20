# Synthetic review generation prompt

Used to create `synthetic_reviews.jsonl`. It targets the slices where the lab 1 model was weakest.
Each generator received this prompt with its own list of categories.

---

Write realistic, varied customer reviews of local businesses in the style of Yelp. They will be
used as extra *training* data for a sentiment classifier with labels `negative`, `neutral` and
`positive`, following the guideline in `golden_labels/GUIDELINE.md`.

For each category below, write **50 reviews**, with labels spread as the category says.

Requirements:
- Vary the business type: restaurants, cafes, bars, hotels, hair salons, dentists, car repair,
  gyms, shops, delivery services. Invent business names, or use none. Never use real ones.
- Vary length (8 to 90 words), tone, vocabulary and reviewer voice: casual, formal, typos,
  slang, ALL CAPS, emoji, run-on sentences. No two reviews may share an opening phrase.
- The label must be what a careful human applying the guideline would choose. Don't make the
  difficulty artificial: these should read like real reviews people write.
- Don't copy or paraphrase any review you've seen in the lab's data.

Output one JSON object per line: `{"text": "...", "label": "...", "category": "..."}`.

## Categories
| category | what it is | labels |
|---|---|---|
| `sarcasm` | literal words are positive, meaning is negative (or the reverse, rarely) | ~45 negative, ~5 positive |
| `mixed_lean` | clear praise *and* clear complaints, but an overall lean | ~25 negative, ~25 positive |
| `negation` | sentiment carried by negated phrases: "not bad at all", "wasn't impressed", "can't complain", "never disappoints" | mixed, ~1/3 each |
| `lukewarm` | genuinely middling verdicts: "it was fine", "average", "nothing special" | ~40 neutral, ~10 slight lean either way |
| `comparison` | verdict expressed by comparison with another place or a past visit | mixed, ~1/3 each |
| `short` | 2 to 8 words, often informal: "meh.", "10/10 would return", "never again lol" | mixed, ~1/3 each |
