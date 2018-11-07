#!/bin/sh
curl -X POST http://hiro_wifi:32841/featurize -H "Content-type: application/octet-stream" --data-binary @$@
