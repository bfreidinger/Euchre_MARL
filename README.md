# CMOR544

## Euchre Web App

Run the interactive Euchre website with:

```bash
python euchre/web/app.py
```

Then open `http://127.0.0.1:8000`.

To run it under `gunicorn` instead:

```bash
gunicorn --bind 127.0.0.1:8000 euchre.web.app:application
```

For a hosted deployment, bind to `0.0.0.0:$PORT`:

```bash
gunicorn --bind 0.0.0.0:${PORT:-8000} euchre.web.app:application
```

The web app lets you:

- play as the human player,
- choose your partner policy,
- choose the left and right opponent policies,
- mix rule-based, random, and saved QMIX checkpoints.
