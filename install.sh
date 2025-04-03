#!/bin/bash
python3 -m pip install --user opencv-python numpy scipy
chmod +x GoColonyCounter_V1.02.py
ln -s "$(pwd)/GoColonyCounter_V1.02.py" /usr/local/bin/GoColonyCounter
echo "✅ Installed! Now you can run GoColonyCounter from anywhere."
