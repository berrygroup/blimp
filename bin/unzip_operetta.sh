#!/bin/bash

echo Replacing spaces with underscores
for zip_file in "$1"/*.zip
do
  new="${zip_file// /_}"
  if [ "$new" != "$zip_file" ]
  then
    if [ -e "$new" ]
    then
      echo not renaming \""$zip_file"\" because \""$new"\" already exists
    else
      echo moving "$zip_file" to "$new"
    mv "$zip_file" "$new"
  fi
fi
done

for zip_file in "$1"/*.zip
do
  echo unzipping using
  echo 7za x $zip_file -o"$1"
  7za x $zip_file -o"$1"
done
