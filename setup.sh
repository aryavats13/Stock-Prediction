#!/bin/bash

# Make directory for Streamlit config
mkdir -p ~/.streamlit

# Create Streamlit config file
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml

# Create credentials file if needed
echo "[general]
email = \"\"
" > ~/.streamlit/credentials.toml
