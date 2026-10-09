# Repository scope

This private repository contains GAME7 movement measurement/scoring source,
legacy rubric definitions, documentation and tests, imported on 9 October 2026.
Participant datasets (`data/`), derived norms, session records, review outputs,
site-specific ROI files, recordings and local credentials are not uploaded.

The complete current local assessment source is also bundled in
https://github.com/hawhc/game7-assessment-bundle with the admin website, camera
backend, human-selfcalib and landmark dependencies.

The newer backend's live scoring policy is the admin project's
`data/k1-live-rubrics.json`, consumed by `dataset_rubric.py`. It uses this project's
measurement/scoring functions. Files under this repository's `rubrics/`, including
`*.proposed.json`, are legacy/development definitions and are not automatically
approved for the current live assessment policy.

Some offline commands in README.md rely on local datasets and hardcoded Mac paths.
Adapt those paths and supply the required data separately on another computer.
Uploading this repository does not deploy any website changes.
