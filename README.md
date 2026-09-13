# TeacherAboveMllmBelow : kinematics-grading

A reusable framework for **reverse-generation stress testing of MLLM-based
grading assistants**: instead of asking "is this answer correct?", it asks
*"at what point during a gradual correction from a wrong answer to the right
one does the grader start accepting it:and does that point make sense?"*

Given a wrong answer and a correct answer to the same question, the
framework synthesizes a smooth sequence of in-between images, has a
vision-capable grading assistant score every step against a rubric, and
analyzes exactly where and why it starts giving passing scores. This
repository ships a full, tested, runnable implementation of that pipeline
for a set of kinematics-graph reading exercises, plus everything needed to
adapt it to grade your own material or stress-test a different grading
assistant.

## Gap

Grading assistants built on multimodal LLMs are usually evaluated on
*correct* answers: does the assistant give full marks to a good answer and
low marks to a bad one? This framework asks the complementary, harder
question: **when a submission is still wrong, but visibly closer to
correct than before, how early does the assistant start treating it as
acceptable:and is that consistent with what its own grading rubric
says?** A grader that gives a passing score to a half-finished answer, or
whose scoring reasoning contradicts its own stated rubric, has a real
weakness that ordinary "grade this correct answer" testing will never
surface.

Two independent grading policies are included, so you can separate two very
different explanations for a high score: *"the grader was too lenient given
the context it was told,"* versus *"the submission is genuinely a good
answer on its own merits."* See **Two grading policies**, below.

## How it works

1. **Corpus generation.** For each exercise, take another exercise's
   *correct* answer image as a guaranteed-wrong starting point, and
   generate a short sequence of in-between images that gradually morph it
   toward the *real* correct answer for the exercise in question (curve
   extraction + parameterized correction strategies, tuned per-instance
   with Optuna so the morph is monotonic and visually smooth). Each
   sequence is 7 images: `00` (the wrong starting point), `01`-`05` (five
   graded correction steps, at nominal 20/40/60/80/95% progress), and `06`
   (the real correct answer, used as a reference but never graded).
2. **Grading.** Every one of the `01`-`05` images is sent to a
   vision-capable grading assistant along with the exercise's rubric and
   instruction text, and graded independently, at scale, with automatic
   retries and full resumability.
3. **Analysis.** A score-ratio threshold (default 0.60) defines
   "accepted." For every correction sequence, find the first step that
   crosses it, how visually far that step still is from the real answer
   (via a frozen ResNet-50 embedding distance), whether scores rise
   monotonically or dip along the way, and how much of an accepted score
   rests on partial credit for rubric rules the grader itself marked
   unsatisfied.
4. **Surrogate + explainability (optional).** A small differentiable model
   is trained to imitate the grading assistant's score from pixels alone
   (the real grading API has no gradients), enabling pixel-level
   attribution (Integrated Gradients+SmoothGrad, Occlusion, Guided
   Grad-CAM) to visualize *what in the image* the imitated grading
   behavior is keying on.

## Two grading policies

The same corpus can be graded twice, under two different sets of
instructions given to the assistant, to separate "the grading was lenient"
from "the image is genuinely good":

- **`step_calibrated`** (the default): the assistant is told the image is
  one step of an ongoing correction, roughly how far along it should be,
  and is shown the real correct-answer image for comparison. It is
  explicitly told that early steps may earn partial credit just for moving
  in the right direction, and that only the final step should be graded
  strictly.
- **`standalone_submission`**: the assistant is given no information
  suggesting the image is a draft or a step of anything:it grades the
  image exactly as if it were a complete, final answer, with no reference
  image to compare against. Partial credit is still allowed, but must be
  justified strictly by how close the image's measured properties are to
  the rubric's own numeric thresholds, never by assumed effort or
  progress.

Run the same corpus under both (into separate output directories) and
compare the resulting acceptance patterns:a large gap tells you how much
of your grading assistant's behavior is really about the submissions, and
how much is an artifact of how the grading prompt frames the task. See
`grading/prompts.py` for the exact two prompts, and `analysis/acceptance.py`
for the metrics used to compare them (`per_step_pass_rate`,
`first_accepted_step_distribution`, `partial_credit_on_unsatisfied_rules`,
`trajectory_score_monotonicity`).

## Requirements

- Python 3.10+
- A vision-capable chat-completions endpoint to grade against: Azure OpenAI
  (what this was built and tested against), plain OpenAI, or any
  API-compatible provider. If your endpoint is already a full
  `.../chat/completions` URL, set `ENDPOINT` to it directly and the client
  will call it as-is; otherwise set `ENDPOINT` to your Azure resource's base
  URL and `DEPLOYMENT` to the deployment name, and the Azure-style path is
  built for you (see `config.py`'s `AzureConfig.chat_url`).
- A GPU is optional but strongly recommended for the surrogate/XAI stages
  (CPU works, just much slower); grading and analysis are CPU-only.

## Data: what one exercise needs

Each exercise lives in `data/exercice_<N>/` and needs exactly three files:

```
data/exercice_2/
  correct_2.png       the true, fully correct answer image for this exercise
  instruction_2.txt    plain-text problem statement (what a correct answer looks like)
  rubric_2.json        the grading rubric (see below)
```

`rubric_<N>.json` defines the scoring criteria as a small list of
independent rules:

```json
{
  "max_score": 10,
  "rules": [
    {"id": "forward", "points": 3, "quantity": "dfdx", "relation": "gt", "expected": 0.1},
    {"id": "parking",  "points": 4, "quantity": "dfdx", "relation": "eq", "expected": 0},
    {"id": "reverse",  "points": 3, "quantity": "dfdx", "relation": "lt", "expected": -0.1}
  ]
}
```

- `id`: a short, unique name for the rule.
- `points`: how many of `max_score`'s points this rule is worth.
- `quantity`: the property of the graph being checked -- `"f"` (the
  plotted function's value), `"dfdx"` (its first derivative / slope), or
  `"d2fdx2"` (its second derivative / curvature). These are read and
  applied by the grading assistant itself from the image, not computed by
  this codebase -- the rubric only needs to describe the criterion in
  terms the assistant can visually evaluate.
- `relation` / `expected`: the threshold the quantity must satisfy --
  `gt`/`ge`/`lt`/`le`/`eq` against `expected`.

`config/default.yaml`'s `generation.source_map` controls which exercises
are used as each other's "guaranteed wrong" starting points; the 4 sample
exercises here are all cross-mapped to each other, but this is just
configuration -- add more exercises by adding more `data/exercice_<N>/`
folders and extending `source_map` and `exercises` in the config.

**Adapting this to a different domain:** the grading, analysis, and
explainability machinery are domain-agnostic given any corpus of
"wrong -> correct" image sequences with a rubric per exercise. Only the
*generation* stage (`generation/curve_extraction.py`,
`generation/correction.py`) is specific to line-graph images; grading a
different kind of visual submission means supplying your own corpus of
correction sequences (same `00`-`06` file-naming convention) and skipping
the `generate` step.

## Quickstart

```bash
git clone https://github.com/zammelRocks/TeacherAboveMllmBelow.git
cd TeacherAboveMllmBelow
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -e .[dev,reports]
cp .env.example .env   # fill in API_KEY / ENDPOINT / DEPLOYMENT
pytest                 # 94 tests, no network or GPU required for all but a handful
```

## Running the experiment

Every command below is resumable: interrupting and re-running picks up
where it left off rather than redoing finished work.

```bash
# 1. Generate the correction-sequence corpus (no API calls; local, CPU-only).
python -m kinematics_grading generate --out output/generated --optuna-dir output/optuna_logs

# 2. Grade every step against your configured assistant (real API calls).
python -m kinematics_grading grade --generated output/generated --out output/grading_results

# ...if a run gets interrupted (rate limits, network blips):
python -m kinematics_grading grade --generated output/generated --out output/grading_results --retry-until-clean

# 3. Compute acceptance-point, partial-credit, and monotonicity statistics.
python -m kinematics_grading analyze --grading-results output/grading_results --out output/analysis

# 4. (Optional) Train a differentiable surrogate of the grading assistant.
python -m kinematics_grading train-surrogate --grading-results output/grading_results --out output/surrogate

# 5. (Optional) Attribution panels for individual graded images.
python -m kinematics_grading xai --grading-results output/grading_results \
  --model output/surrogate/proxy_model.pt --out output/xai

# 6. (Optional) Combined per-trajectory panels (00->06, 4 attribution methods).
python -m kinematics_grading trajectory-panels --grading-results output/grading_results \
  --model output/surrogate/proxy_model.pt --out output/xai_trajectory_panels
```

To run the second grading policy on the same corpus for comparison (see
**Two grading policies** above), grade again into a *different* output
directory:

```bash
python -m kinematics_grading grade --generated output/generated \
  --out output/grading_results_standalone --partial-credit-policy standalone_submission
```

`notebooks/statistical_analysis.ipynb` and
`notebooks/standalone_submission_analysis.ipynb` are ready-to-run analysis
notebooks (`pip install -e .[reports]`) that reuse the same tested
`analysis/acceptance.py` functions as the CLI, produce the charts described
below, and (the second one) compare the two policies directly if you have
graded both.

## Interpreting the results

- **Acceptance rate**: % of correction sequences that ever cross the 0.60
  score threshold.
- **First accepted step**: for one sequence, the earliest step whose score
  crosses the threshold -- the central "how early does the grader give in"
  number.
- **Per-step pass rate**: independent of "first" -- at each step 1-5, what
  fraction of images score >= 0.60 on their own. The *shape* of this curve
  (gradual climb vs. a sudden jump) shows where the assistant's judgment
  shifts from strict to lenient.
- **Confusing / robust-to-noise**: an image scored *below* threshold despite
  looking visually very close to the correct answer is "confusing" (the
  grader may be too strict); scored *above* threshold despite looking
  visually far from it is "robust-to-noise" (the grader may be too
  lenient).
- **Partial credit from unsatisfied rules**: of an accepted image's
  awarded points, how much came from rubric rules the grader itself marked
  as *not* satisfied. A high share here means the acceptance is arithmetic
  (many small consolation credits adding up past the threshold), not a
  reflection of the rubric actually being met.
- **Monotonicity / regressions**: within one sequence, does the score ever
  *drop* at a later step? Under lenient, context-aware grading this rarely
  happens by construction; under standalone grading, regressions reveal
  real quality dips in the generation process itself, independent of the
  grading assistant.

All of these are computed by tested, reusable functions in
`analysis/acceptance.py` and `analysis/classification.py` -- see their
docstrings for exact definitions, and `tests/test_acceptance.py` for
worked examples of each.

## Project layout

```
config/default.yaml         single source of truth for thresholds/counts
data/                        exercise instructions, rubrics, correct-answer images
src/kinematics_grading/
  config.py                  Settings/AzureConfig loaders
  domain.py                  TrajectoryKey, StepInfo -- shared path/step parsing
  generation/                 curve extraction, correction strategies, Optuna tuning, rendering, pipeline
  grading/                    prompts (both policies, with caching), async vision client, normalize, JSONL store, pipeline
  analysis/                   embeddings (shared cache), acceptance-point / partial-credit / monotonicity statistics, confusing/robust-to-noise
  surrogate/                  differentiable proxy model (configurable staged fine-tuning), train/evaluate
  xai/                        saliency (IG+SmoothGrad), Occlusion, Guided Grad-CAM, combined per-trajectory panels
  cli.py                      `generate` / `grade` / `analyze` / `train-surrogate` / `xai` / `trajectory-panels`
tests/                        pytest suite, no network/GPU required for all but a handful of integration tests
scripts/run_smoke_test.py     small, real end-to-end run against your configured endpoint
notebooks/                    ready-to-run analysis notebooks (pip install -e .[reports])
```

## Design notes

A few non-obvious implementation choices, in case you're extending this:

- **Prompt caching.** Each grading prompt is split into a stable prefix
  (rubric, instruction, output format -- identical across every step of one
  exercise) and a short per-image suffix. This is cached client-side and is
  also what lets provider-side automatic prompt-prefix caching (Azure
  OpenAI / OpenAI) actually engage across a whole exercise's worth of
  requests.
- **ResNet stages are addressed by name, not list index**
  (`self.encoder.layer4`, not `body[7]`). An earlier, index-based version
  of this kind of code silently fine-tuned the wrong block because a
  flattened `nn.Sequential` reindexes torchvision's named children --
  addressing by name eliminates that entire bug class.
- **Grad-CAM and LIME were tried first for the combined attribution
  panels and both underperformed** on thin line-graph images: Grad-CAM's
  7x7 feature map is too coarse to localize a 1-2px-wide curve, and LIME's
  superpixel segments frequently land on background/border content rather
  than the curve. Guided Grad-CAM and small-patch Occlusion (via Captum)
  localize far better at comparable cost; see `xai/attributions.py`.
- **Embedding cache keys use `Path.as_posix()`**, not
  `str(Path(...).resolve())` -- on Windows, `np.savez`/`np.load` round-trip
  array names through a zip archive that silently normalizes backslashes to
  forward slashes, so a naive resolved-path key permanently misses on
  reload and silently re-embeds (and re-saves) everything every time.

## Testing

```bash
pytest                                        # full suite
pytest --ignore=tests/test_cli_trajectory_panels.py   # skip the slower GPU-touching tests
```

Tests are hermetic: they generate tiny real corpora and run the actual CLI
commands rather than mocking the pipeline, so CLI-wiring bugs (a missing
argument, an import error) are caught the same way a unit test would catch
a logic bug.
