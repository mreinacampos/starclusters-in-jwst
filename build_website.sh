#bin/bash/sh

# build the html files
(cd web && bundle exec jekyll build)
#cp -r _site ../
rm -r ./docs
mv ./web/_site/ ./docs
git add docs
git commit -m "Updating the website"
git push origin main