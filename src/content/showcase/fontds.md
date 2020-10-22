+++
date = "2020-07-07"
title = "Font dataset"
showonlyimage = false
draft = false
image = "https://raw.githubusercontent.com/sdhnshu/fontds/master/opensans.jpg"
weight = 1
+++

Generate an image dataset from your favourite fonts.
<!--more-->

![img](https://raw.githubusercontent.com/sdhnshu/fontds/master/opensans.jpg)

### What is it?
I wanted an image dataset (1024x1024px) of fonts of a particular type for analysis. So I wrote a Python script to scrape [FontSquirrel](https://www.fontsquirrel.com/) and download fonts. Another script writes [filler text](https://en.wikipedia.org/wiki/Hamburgevons) to an image using those fonts.

### The download script
It works with fontsquirrel and downloads only the fonts that are stored onsite. With minor tweaks, you can download fonts from Google fonts or any other site. Regardless of the site, respect the service and don't ping it too often.

```python
python download_fonts.py --url https://www.fontsquirrel.com/fonts/list/classification/sans%20serif --start 1 --end 2
```
The download script takes the webpage URL and page numbers as parameters.

### The drawing script
It can work with both `.ttf` and `.otf` files to generate images. The image size is 1024x1024px with each character getting 256x256px. Some adjustments are made to center the character. The script uses the `Pillow` Python package to draw fonts on the image and saves it as `.jpg` files.

```python
python draw_fonts.py --f fontfiles/OpenSans-Semibold.ttf
```

The script generates images from all the fonts in the `fontfiles` folder, but with an optional file parameter, you can generate an image for a single font.

#### Github: [https://github.com/sdhnshu/fontds](https://github.com/sdhnshu/fontds)