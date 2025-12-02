#!/bin/bash

DATA_DIR='../data'

mkdir -p "$DATA_DIR"

echo "Downloading Fairfax County.zip..."
wget -q -O 'FairfaxCounty.zip' 'https://gwu.box.com/shared/static/wvgmx7n7tbrhuk140t8tm9zkb2z5top9.zip'

echo "Unzipping Fairfax County.zip..."
unzip -q FairfaxCounty.zip

rm -rf "$DATA_DIR/FairfaxCounty"
mv FairfaxCounty "$DATA_DIR/"
rm FairfaxCounty.zip

echo "Downloading preprocessed-data.zip..."
wget -q -O 'preprocessed-data.zip' 'https://gwu.box.com/shared/static/r2wfegg5ut4lwe76s3wwe0qa831r9zbe.zip'

echo "Unzipping preprocessed-data.zip..."
unzip -q preprocessed-data.zip -d preprocessed-data -x '__MACOSX/*'

rm -rf "$DATA_DIR/preprocessed-data"
mv preprocessed-data "$DATA_DIR/"
rm preprocessed-data.zip

echo "Success! All data downloaded to $DATA_DIR!"