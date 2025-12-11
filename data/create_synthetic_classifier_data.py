#!/usr/bin/env python3
"""Create a tiny synthetic dataset for classifier smoke tests.
"""
import numpy as np
from pathlib import Path

def main():
    out_dir = Path('data/synthetic_test')
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = out_dir / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    # 12-lead signals, length 500
    L = 500
    C = 12
    train_n = 200
    val_n = 40
    test_n = 30
    num_labels = 10

    def create_split(n):
        X = rng.normal(0, 1, size=(n, C, L)).astype('float32')
        Y = (rng.random((n, num_labels)) > 0.8).astype('uint8')
        return X, Y

    X_train, Y_train = create_split(train_n)
    X_val, Y_val = create_split(val_n)
    X_test, Y_test = create_split(test_n)

    np.save(out_dir / 'train_input.npy', X_train)
    np.save(out_dir / 'val_input.npy', X_val)
    np.save(out_dir / 'test_input.npy', X_test)
    np.save(out_dir / 'train_target.npy', X_train)
    np.save(out_dir / 'val_target.npy', X_val)
    np.save(out_dir / 'test_target.npy', X_test)

    np.save(labels_dir / 'train_labels.npy', Y_train)
    np.save(labels_dir / 'val_labels.npy', Y_val)
    np.save(labels_dir / 'test_labels.npy', Y_test)
    import json
    labels = [f'label_{i}' for i in range(num_labels)]
    (labels_dir / 'labels.json').write_text(json.dumps(labels))
    print('Synthetic test data created in', out_dir)

if __name__ == '__main__':
    main()
