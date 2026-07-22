"""Verify that the local database, model, tokenizer, and Flask routes work."""

import json
import sys
from importlib.metadata import version
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def main() -> None:
    """Run a no-server smoke test and print evidence that a launch is safe."""
    from app.run import app

    client = app.test_client()
    health = client.get('/health')
    prediction = client.get('/go?query=We+need+clean+water+and+food')
    if health.status_code != 200:
        raise RuntimeError(f'Health route failed with HTTP {health.status_code}.')
    if prediction.status_code != 200:
        raise RuntimeError(f'Prediction route failed with HTTP {prediction.status_code}.')

    result = {
        'runtime': 'ready',
        'python': sys.version.split()[0],
        'scikit_learn': version('scikit-learn'),
        'database': str(PROJECT_DIR / 'data' / 'DisasterResponse.db'),
        'model': str(PROJECT_DIR / 'models' / 'classifier.pkl'),
        'health': health.get_json(),
        'prediction_smoke_test': 'passed',
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
