# ManipuLens Annotation Codebook — v0.1

This codebook defines the manipulation-technique taxonomy used for all annotation
(human, LLM, and weak supervision). Every downstream label depends on these
definitions. Changes require a version bump and re-validation of agreement.

**Unit of annotation:** a single headline (or social-media post title), without
the article body. Each dimension is scored independently on a 0–2 ordinal scale:

- **0 — absent**: technique not present.
- **1 — mild**: technique present but restrained.
- **2 — strong**: technique is the headline's dominant device.

A headline can score on multiple dimensions simultaneously. Score what the text
*does*, not whether you agree with it or believe it.

---

## D1. Curiosity gap / forward reference

Withholding the key information the headline itself promises, forcing a click.
Markers: unresolved deictics ("this", "these", "here's what"), "you won't
believe", "the reason will shock you", cliffhanger constructions.

| Score | Example |
|---|---|
| 0 | "Senate passes $1.2T infrastructure bill 69-30" |
| 1 | "Here's what the new tax rules mean for homeowners" |
| 2 | "You Won't Believe What This Senator Said Next" |
| 2 | "This one trick doctors don't want you to know" |
| 1 | "The surprising reason your electric bill went up" |

**Edge case:** legitimate explainers ("What to know about X") score 1 only if
the headline teases a specific withheld fact; general topic framing scores 0.

## D2. Outrage bait

Framing designed to provoke anger or indignation, often at a person or group.
Markers: "slams", "destroys", "blasts", "humiliates", loaded epithets,
us-vs-them framing.

| Score | Example |
|---|---|
| 0 | "Governor vetoes education funding bill" |
| 1 | "Governor slams critics of education veto" |
| 2 | "Governor DESTROYS whiny critics in epic takedown" |
| 2 | "They want to take away everything you've worked for" |
| 0 | "Court rules against city in zoning dispute" |

**Edge case:** quoting a subject's own inflammatory words scores 1, not 2,
if attributed with quotation marks.

## D3. Fear-mongering

Amplifying threat, danger, or catastrophe beyond what the content supports.
Markers: "terrifying", "deadly", "warning", "crisis", "could kill", worst-case
hypotheticals presented as imminent.

| Score | Example |
|---|---|
| 0 | "CDC reports seasonal flu cases up 12% over last year" |
| 1 | "Health officials warn flu season could strain hospitals" |
| 2 | "Deadly flu outbreak: is your family at risk?" |
| 2 | "The silent killer hiding in your kitchen" |
| 1 | "Experts sound alarm over rising sea levels" |

**Edge case:** genuinely urgent public-safety notices (evacuation orders,
recalls) score 0 even with strong language, if the threat is literal.

## D4. False certainty / overclaiming

Presenting contested, preliminary, or speculative information as settled fact.
Markers: "proves", "confirms", "the truth about", definitive claims from single
studies, absence of hedging where hedging is warranted.

| Score | Example |
|---|---|
| 0 | "Study suggests link between diet and sleep quality" |
| 1 | "New study shows diet controls your sleep" |
| 2 | "Scientists prove this food ruins your sleep" |
| 2 | "The truth about vaccines they aren't telling you" |
| 0 | "Researchers find correlation between exercise and mood" |

## D5. Emotional framing

Loading the headline with affect-heavy language (positive or negative) beyond
informational need. Markers: "heartbreaking", "devastating", "incredible",
"stunning", superlatives, melodrama.

| Score | Example |
|---|---|
| 0 | "Local library reopens after renovation" |
| 1 | "Beloved local library finally reopens after long renovation" |
| 2 | "Heartbreaking scenes as tearful librarians reopen beloved landmark" |
| 2 | "Absolutely stunning: the most incredible comeback in history" |
| 1 | "Community celebrates library's triumphant return" |

## D6. Sensational formatting

Typographic and structural attention devices. Markers: ALL-CAPS words,
multiple exclamation/question marks, listicle bait ("27 things..."),
"(number) + you/your" templates, emoji spam.

| Score | Example |
|---|---|
| 0 | "City council approves budget for 2026" |
| 1 | "7 changes coming to the city budget" |
| 2 | "27 INSANE budget facts!! #9 will blow your mind" |
| 2 | "BREAKING!!! You NEED to see this" |
| 1 | "Quiz: how well do you know your city budget?" |

---

## Annotation protocol

1. Read the headline once at natural speed; score first impressions, then verify
   against markers.
2. Score each dimension independently — do not let one dimension anchor another.
3. When torn between two scores, take the lower one (conservative default).
4. Do not research the story. Annotate the text as a reader would encounter it.
5. Flag headlines that are not news-like (product listings, sports scores) as
   `out_of_scope` rather than forcing scores.

## Reliability policy

- Gold set is double-annotated; Krippendorff's α (ordinal) computed per dimension
  (`manipulens.labeling.agreement`).
- **Pre-committed rule:** any dimension with α < 0.6 on the gold set is merged
  into its nearest neighbor or dropped before model training. Pruning is
  reported in the dataset card, not hidden.

## Changelog

- **v0.1** — initial six-dimension taxonomy with 0–2 ordinal scales.
