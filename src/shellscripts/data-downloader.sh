#!/bin/bash
wget -O 'FairfaxCounty.zip' 'https://gwu.box.com/shared/static/wvgmx7n7tbrhuk140t8tm9zkb2z5top9.zip'
unzip FairfaxCounty.zip 
rm FairfaxCounty.zip

wget -O 'preprocessed-data.zip' 'https://gwu.box.com/shared/static/4rilgmk1oyz237a6c888tw1mmm24b1g8.zip'
unzip preprocessed-data.zip -d preprocessed-data -x '__MACOSX/*'
rm preprocessed-data.zip