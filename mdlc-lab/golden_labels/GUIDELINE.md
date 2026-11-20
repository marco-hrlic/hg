# Sentiment labelling guideline, v1

You are labelling the **overall sentiment of the reviewer towards the business they are reviewing**,
as expressed in the text. Choose exactly one label per review.

| Label | Use when |
|---|---|
| `negative` | The reviewer is unhappy overall. Complaints dominate, or the reviewer says they won't return or wouldn't recommend it, even if they mention some positives. |
| `positive` | The reviewer is happy overall. Praise dominates, or they would return or recommend it, even if they mention minor complaints. |
| `neutral` | The verdict is genuinely lukewarm ("it was OK", "average", "nothing special"), genuinely balanced with no overall lean, or the text contains no evaluation at all (purely factual or informational). |
| `ambiguous` | Two careful readers applying this guideline could reasonably disagree, or the text is unintelligible, not a review, or not in English. Use it sparingly. These items are excluded from the golden set. |

## Rules for hard cases

1. **Overall verdict wins.** Weigh what the reviewer concludes over individual details. The final
   statement, a stated intention to return or not, and an explicit recommendation carry the most weight.
2. **Mixed reviews.** "Food great, service awful": decide which side the reviewer leans to overall,
   using rule 1. Only if there is no lean at all, use `neutral`.
3. **Lukewarm is neutral.** "Decent", "fine", "OK", "not bad but not great", "average" with no clear
   lean → `neutral`. "Not bad at all", "pleasantly surprised" → `positive`.
4. **Sarcasm.** Label the intended meaning, not the literal words.
5. **Negation.** Read it carefully: "not disappointed" is positive; "not great" is mildly negative
   or neutral depending on context.
6. **Comparisons.** Judge the business being reviewed, not the one it's compared with.
7. **Off-target complaints.** Complaints about parking, the weather, other customers or Yelp itself
   count only if the reviewer holds them against the business.
8. **Don't guess the star rating.** A 3-star-sounding review isn't automatically neutral. Label
   what the text says.
9. **Changed experiences.** "Used to love it, now it's gone downhill": label the current verdict.

## Confidence

For each item also record `high` or `low` confidence. `low` means you applied the rules but a
colleague might reasonably choose differently.
