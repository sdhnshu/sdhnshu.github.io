+++
date = "2020-07-07"
title = "How to build your own font image dataset"
showonlyimage = false
draft = false
image = "/img/posts/opensans.jpg"
weight = 1
+++

Generate an image dataset from your favourite fonts.
<!--more-->

![img](/img/posts/opensans.jpg)

- Grab the code from [github.com/sdhnshu/fontds](https://github.com/sdhnshu/fontds)

### Intro
In this post, I'll show a way to create a dataset of `1024px x 1024px` images showcasing your favourite fonts. You may use such a dataset for any kind of analysis or computer vision model training. The whole process takes __less than 5 mins__ and can be done on any basic computer. You will be able to choose the fonts to add to your dataset and the filler text ([*Hamburgefonstiv*](https://en.wikipedia.org/wiki/Hamburgevons) in this case).

### Thank you fontsquirrel.com
Before we move on, I'd like to thank the people behind [FontSquirrel.com](https://www.fontsquirrel.com/) for the amazing resources on their website. I've downloaded many free fonts from their website in the past. And we'll be downloading our `.otf` or `.ttf` fontfiles from there today. In the scripts I've written, I've made sure that I'm not taxing their servers by pinging them too often. Doing so will restrict others from accessing their resources and can get your IP address blacklisted. We do not want that. That said, let's get into the process.

### The Process

#### Step 1: Download the fonts

Currently, my code supports downloading exclusively from [FontSquirrel](https://www.fontsquirrel.com/) servers. This doesn't include the fonts linked from fontsquirrel to external sources (Offsite). It is easy enough to adapt it to other websites given you have some basic knowledge of [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/). Feel free to add a [Pull Request to the repo](https://github.com/sdhnshu/fontds/pulls) when you do so.

To download the fonts, all you have to do is filter the ones you want using the pane on the right and copy the URL of the page. It should look something like `https://www.fontsquirrel.com/fonts/list/classification/sans%20serif`. Also, take a note of the page numbers you want to download to and from.

After installing the required packages from [requirements.txt](https://github.com/sdhnshu/fontds/blob/master/requirements.txt), run the script as follows:

- `--url`: url of the page you want to scrape
- `--start`: first page number to scrape
- `--end`: last page number to scrape

```cmd
python download_fonts.py --url https://www.fontsquirrel.com/fonts/list/classification/sans%20serif --start 1 --end 2
```

All fontfiles are downloaded as `.zip` in the `fontfiles` folder. You can unzip them all at once by running `open *zip` on Macintosh if you have [Unarchiver](https://theunarchiver.com/) or by running `unzip '*.zip'` on Linux. For Windows, Winrar does the job I guess.

#### Step 2: Draw the fonts
Once all the font files are unzipped and ready, we can ask [Pillow](https://python-pillow.org/) to draw them onto a 1024x1024 canvas.

Because I wanted a dataset of images to train a Generative Adversarial Network, which works well with 4x4 grids, I decided to give each alphabet `256px x 256px`. The challenge I faced was that the alphabets did not align in the center of the space I allocated. So to solve that, I added some [padding to the top](https://github.com/sdhnshu/fontds/blob/master/draw_fonts.py#L21).

To convert all the fonts in the `fontfiles` folder to .jpg, run:

```cmd
python draw_fonts.py
```

If you want to convert just one of them, use something like:

```cmd
python draw_fonts.py --f fontfiles/OpenSans-Semibold.ttf
```

This will create images by the same name as the font in the `imgs` folder.

You can also change the filler text by changing the [text on this line](https://github.com/sdhnshu/fontds/blob/master/draw_fonts.py#L8).

---

That's all folks! Feel free to use the code for your projects as it is under the MIT License. But make sure you limit the ping to any servers to ensure availability to others. Happy Coding!