#!/usr/bin/env bash

curl --header 'Host: drivendata.s3.amazonaws.com' --header 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:41.0) Gecko/20100101 Firefox/41.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3' --header 'DNT: 1' --header 'Content-Type: application/x-www-form-urlencoded' 'https://drivendata.s3.amazonaws.com/data/8/public/bee_images.zip' -O -J -L
apt-get install -y unzip
unzip bee_images.zip
rm -f bee_images.zip