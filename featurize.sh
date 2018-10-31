#!/bin/sh
curl -X POST http://localhost:32817/featurize -H "Content-type: application/octet-stream" --data-binary @$@
