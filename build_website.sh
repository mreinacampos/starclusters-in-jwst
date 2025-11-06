#bin/bash/sh

# build the html files
(cd web && bundle exec jekyll build)
rm -r ./docs
mv ./web/_site/ ./docs
git add docs
git add web
git add assets
git commit -m "Updating the website - html and raw files"
git push origin main