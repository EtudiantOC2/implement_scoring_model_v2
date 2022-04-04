
gunicorn app:app --timeout 10 --bind 0.0.0.0:${PORT:-5000}