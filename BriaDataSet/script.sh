#!/bin/bash

counter=1
for file_jpeg in *.jpg; do
    if [ -f "$file_jpeg" ]; then
        base_name="${file_jpeg%.*}"
        file_json="${base_name}.json"
        
        nuovo_nome_jpeg="Sign${counter}.jpg"
        nuovo_nome_json="Sign${counter}.json"

        mv "$file_jpeg" "$nuovo_nome_jpeg"
        mv "$file_json" "$nuovo_nome_json"

        ((counter++))
    fi
done
