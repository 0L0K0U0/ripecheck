#!/usr/bin/env bash
# Double-click launcher for macOS (and a one-line launcher for Linux).
# If double-click does nothing on a Mac, run once:  chmod +x start_mac_linux.command
cd "$(dirname "$0")" || exit 1
ENV_NAME=ripecheck

fail() {
  echo
  echo "Something went wrong. Copy the text in this window and send it to whoever is helping you."
  read -r -p "Press Enter to close. "
  exit 1
}

# ---- 1. Find conda ----
if ! command -v conda >/dev/null 2>&1; then
  for d in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/opt/anaconda3" "$HOME/opt/miniconda3" "/opt/anaconda3" "/opt/miniconda3" "/usr/local/anaconda3" "/usr/local/miniconda3"; do
    if [ -x "$d/bin/conda" ]; then export PATH="$d/bin:$PATH"; break; fi
  done
fi
command -v conda >/dev/null 2>&1 || { echo "Could not find Anaconda or Miniconda."; fail; }
echo "Using conda at $(command -v conda)"
eval "$(conda shell.bash hook)" || fail

# ---- 2. Create the environment on first run ----
if ! conda activate "$ENV_NAME" >/dev/null 2>&1; then
  echo "First run: creating the $ENV_NAME environment. This takes a few minutes."
  conda create -n "$ENV_NAME" python=3.11 -y || fail
  conda activate "$ENV_NAME" || fail
fi

# ---- 3. Install packages if any are missing ----
if ! python -c "import flask, cv2, numpy" >/dev/null 2>&1; then
  echo "Installing packages. This only happens once."
  python -m pip install -r requirements.txt || fail
fi

# ---- 4. Run. The app opens your browser by itself. ----
echo
echo "Starting RipeCheck at http://127.0.0.1:5057"
echo "To stop it, press Ctrl+C in this window."
echo
python app.py
echo
echo "RipeCheck has stopped."
read -r -p "Press Enter to close. "
