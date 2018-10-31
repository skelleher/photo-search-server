#!/bin/sh
curl -X POST http://localhost:32816/query -H "Content-type: application/octet-stream" --data-binary @$@
