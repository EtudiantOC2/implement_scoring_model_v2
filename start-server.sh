if [ -z $API_URL ]; then
  # Pas d'API_URL => lancer l'API
  gunicorn wsgi:app --bind 0.0.0.0:${PORT:-8000}
else
  # Avec API_URL => lancer le dashboard
  mkdir -p ~/.streamlit/

  cat > ~/.streamlit/config.toml <<EOF
[server]
headless = true
port = ${PORT:-8042}
enableCORS = true
EOF

  cat > ~/.streamlit/credentials.toml <<EOF
[general]
email = "paulinegleiron@gmail.com"
EOF
  streamlit run dashboard.py
fi
