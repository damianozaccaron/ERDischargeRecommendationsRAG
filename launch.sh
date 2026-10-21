#!/bin/bash -i
# Opens the site in the browser, for a desktop shortcut: starts it first, unless it is already running (a second copy
# would load the models again, past the memory of a 6 GB GPU). -i reads ~/.bashrc, where conda puts its python3 on the
# PATH, as in a terminal. Closing the terminal window stops the site.
cd "$(dirname "$0")"
if python3 -c "import socket; socket.create_connection(('localhost', 8501), 1)" 2>/dev/null; then
    setsid -f xdg-open http://localhost:8501
else
    python3 -m streamlit run code/app.py --server.port 8501
fi
