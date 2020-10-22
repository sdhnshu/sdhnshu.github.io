#! /bin/sh
cd docs
rm -r about categories css img js tags experiments inspirations showcase
rm *.html *.xml
cd ../src
hugo --minify -d ../docs