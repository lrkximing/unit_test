# RACA Usage

## Dataset

- `data/ut_targets.json`: target-function dataset, currently 166 functions from 74 drivers.

## Environment

Set the kernel and Buildroot paths before running RACA. The kernel path should
point to the Linux source tree that Buildroot builds.

```bash
export RACA_LINUX_KERNEL_PATH=/path/to/buildroot/output/build/linux-custom
export RACA_BUILDROOT_DIR=/path/to/buildroot
export LLM_API_KEY=your_api_key
export LLM_MODEL=deepseek-v3.2
```

Set these only when using a custom API endpoint, result directory, or local
model.

```bash
export LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
export RACA_RESULT_ROOT=output_all
export RACA_LOCAL_MODEL_PATH=/path/to/local/model
```

## Run RACA

List the selected targets without modifying the kernel tree.

```bash
python main.py --dry-run-targets
```

Run RACA on one target function. This is useful for checking whether the
environment is configured correctly.

```bash
python main.py \
  --linux-kernel-path "$RACA_LINUX_KERNEL_PATH" \
  --buildroot-dir "$RACA_BUILDROOT_DIR" \
  --driver drivers/iio/humidity/am2315.c \
  --function am2315_crc \
  --result-root output_single
```

Run RACA on the full dataset defined in `data/ut_targets.json`.

```bash
python main.py \
  --linux-kernel-path "$RACA_LINUX_KERNEL_PATH" \
  --buildroot-dir "$RACA_BUILDROOT_DIR" \
  --result-root output_all
```

Remove generated KUnit files and config changes for a selected target.

```bash
python main.py \
  --linux-kernel-path "$RACA_LINUX_KERNEL_PATH" \
  --buildroot-dir "$RACA_BUILDROOT_DIR" \
  --driver drivers/iio/humidity/am2315.c \
  --function am2315_crc \
  --cleanup-only
```

## Metrics

Compute per-driver and per-category metrics from a completed `output_all`
directory.

```bash
python evaluate.py
python evaluate_types.py
```

Compute the iteration where each target reaches its best checkpoint.

```bash
python tools/compute_best_stop_iterations.py --root output_all
```

## Mutation Testing

Generate a shared mutant catalog from the target-function dataset.

```bash
python -m mutation.generate_mutant_catalog \
  --targets data/ut_targets.json \
  --linux-dir "$RACA_LINUX_KERNEL_PATH" \
  --replacement-policy representative \
  --output mutant_catalog.json \
  --csv mutant_catalog.csv
```

Run mutation testing for targets that have `mutation_ready.json` in the RACA
result directory.

```bash
python -m mutation.run_batch_from_manifests \
  --root output_all \
  --method raca \
  --output-root mutation_eval_raca \
  --replacement-policy representative \
  --mutation-scope target-with-passed \
  --buildroot-dir "$RACA_BUILDROOT_DIR" \
  --timeout 900 \
  --function-timeout 900 \
  --skip-existing
```

Aggregate mutation results into JSON and CSV summaries.

```bash
python -m mutation.aggregate_mutation_results \
  --result RACA=mutation_eval_raca \
  --catalog mutant_catalog.json \
  --output-json mutation_summary.json \
  --output-prefix mutation_scores
```
