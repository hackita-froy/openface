#!/bin/bash
docker run -p 9000:9000 -t -i -v /home/michael/code/openface:/openface -v /home/michael/code/nli_faces:/nli_faces openface "/openface/go.sh"
