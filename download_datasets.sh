#!/usr/bin/env bash
set -e
echo "This script attempts to download public datasets for the project."
echo "You may need to edit URLs or accept licenses manually. Run from project root."

# Example Enron mail archive (requires internet)
echo "Attempting to download Enron subset (may not work from all environments)..."
mkdir -p data/raw
# Common Enron dataset mirrors or prepared CSVs may be hosted; replace these URLs if needed.
# The following are placeholders and may need to be changed to valid dataset URLs.
ENRON_URL="https://www.cs.cmu.edu/~./enron/enron_mail_20110402.tgz"
PHISHING_URL="https://raw.githubusercontent.com/khushnirs/Kaggle-Phishing-Data/master/phishing_site_urls.csv"

echo "Downloading Enron archive (this may take a while) ..."
if command -v wget >/dev/null 2>&1; then
    wget -c "$ENRON_URL" -P data/raw || echo "wget failed; please download $ENRON_URL manually and place in data/raw/"
else
    echo "wget not available; please download $ENRON_URL manually and place in data/raw/"
fi

echo "Downloading example phishing URL list ..."
if command -v wget >/dev/null 2>&1; then
    wget -c "$PHISHING_URL" -O data/raw/phishing_site_urls.csv || echo "phishing download failed; please fetch manually"
else
    echo "wget not available; please download $PHISHING_URL manually and place in data/raw/"
fi

echo "Done. You will need to run preprocessing to convert raw Enron mailboxes to a single CSV 'data/emails.csv'."
