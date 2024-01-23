#!/bin/bash

./convert.sh
echo
echo "Navigate to: http://localhost:8000/index.html"
echo
python -m http.server
