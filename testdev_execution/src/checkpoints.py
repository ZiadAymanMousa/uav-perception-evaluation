# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Separate exact historical weight identity from new completed training."""
from pathlib import Path
from common import require, digest

def checkpoint_context(task, weights, mode, *, seed=None, baseline_run=None,
                       training_data=None, training_config=None, training_binding=None):
    weights = Path(weights)
    require(weights.is_file() and weights.suffix.lower() == '.pt', 'Explicit existing local .pt checkpoint required; no downloads')
    actual = digest(weights)
    extra = (baseline_run, training_data, training_config, training_binding)
    if mode == 'frozen-replay':
        require(seed in task['checkpoints'], 'Frozen replay requires seed0, seed1 or seed2')
        require(not any(p is not None for p in extra), 'Frozen identity replay does not take prospective training context')
        expected = task['checkpoints'][seed]
        require(actual == expected['weights_sha256'], 'Checkpoint differs from selected frozen seed')
        return {'mode': mode, 'seed': seed, 'weights_sha256': actual,
                'selected_development_epoch': expected['selected_development_epoch'],
                'training_claim_scope': 'Matches recorded historical checkpoint bytes; no new training or new baseline completion'}
    require(mode == 'prospective-completed-baseline', 'Unknown checkpoint mode')
    require(seed is None and all(p is not None for p in extra), 'Prospective mode requires separate actual completed-baseline data/config/binding/run context; no historical seed label')
    from execution_inputs import data_context, completed_checkpoint
    context = data_context(training_data, training_config, training_binding)
    record = completed_checkpoint(baseline_run, weights, context=context)
    # The legacy historical-smoke certificate remains smoke-only in its original adapter.
    return {'mode': mode, 'weights_sha256': actual,
            'baseline_record_sha256': digest(Path(baseline_run)/'run_record.json'),
            'baseline_progress_sha256': digest(Path(baseline_run)/'progress.json'),
            'baseline_config_sha256': record['config_sha256'], 'training_data_context': context,
            'training_claim_scope': 'Actual prospective completed baseline; cannot recover or be labelled historical seed results'}
