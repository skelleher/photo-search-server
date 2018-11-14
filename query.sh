#!/bin/sh
curl -X POST http://hiro_wifi:1980/v1/search -H "Content-type: application/octet-stream" --data-binary @$@
