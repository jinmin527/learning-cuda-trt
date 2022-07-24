#!/bin/bash

echo Post workspace/rq.jpg
wget --post-file=workspace/rq.jpg --read-timeout=1200 "http://127.0.0.1:9090/api/detect" -O-