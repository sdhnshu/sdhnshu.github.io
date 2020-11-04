+++
date = "2020-11-04"
title = "What part of your real-estate do people spend most of their time in?"
showonlyimage = false
draft = false
image = "/img/posts/heatmap-combined.jpg"
weight = -1
+++

Learn how to analyze your space usage from standard CCTV footage.
<!--more-->

![img](/img/posts/heatmap-combined.jpg)

### Introduction
Whether you own a small shop, a supermarket, or a co-working space, its important for a business owner to understand how their clients use their space.

- Did the recent change in arrangement make a difference?
- What part of my store do people spend most of their time in?
- Do I need to increase or decrease the amount of space in a certain area?

These are the kind of questions you will be able to answer by the end of this post.

#### The Technique

Data collected from CCTV cameras are generally used retroactively. But it can be used for calculating footfall, analyzing traffic patterns, and identifying hot areas.

The technique I'll be specifying in this post is specifically about identifying hot areas but can be modified to be used for other use-cases as well.

In simple words it looks like follows:

> *Accumulating the amount of change in the area by comparing each frame with a frame without people.*

Let's break it down:

So the idea is to find the difference between a `frame from the CCTV footage` and an `image without people` in it. To choose an image without people, you can consider an image from that morning, when you opened the gates for your customers. And as the day progresses, each frame from your day will be compared to the one in the morning and the change will be recorded. This difference would represent any people visiting your store and also the changes in store arrangement if any.

To give you more clarity on the process, let us look into some code.

### Converting Video to Images

The first thing you would want to do when you've acquired the CCTV footage *(multiple GBs/day in size)* is to convert them into discrete images.

We'll be using [OpenCV](https://opencv.org/) for reading the video file and converting each frame into a black and white image. Doing so makes the comparison easier. We are willing to lose that information as it doesn't add any value anyway. I'm also adding a Gaussian blur to ensure smooth changes.
```python
import cv2

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale
    img = cv2.blur(img, (11, 11))  # Add gaussian blur
    return img
```

I'm skipping 5 frames between each save as they are pretty much the same. You can try experimenting with 10-15.

```python
if __name__ == "__main__":
    vid = cv2.VideoCapture('cctv-footage.avi')
    skip = 5

    i = 0
    while True:
        i += 1
        grabbed, frame = vid.read()
        if not grabbed:  # Stop if last frame is done
            break
        if i % skip:  # Skip frames
            continue
        frame = preprocess(frame)
        cv2.imwrite('images/{}.png'.format(i), frame)  # Save to images folder
    vid.release()
```

Once done, you'll have slightly blurred, black and white images representing discrete timestamps in your day. We are ready to do the comparison

### Comparing each frame with a static one

Before we start the comparison, we need to make sure that the image we'll be comparing it to (from the morning, without people), goes through the same `preprocess` function mentioned above. I've saved it as `no_people.jpg`. Let's read the image and save it to a variable called `no_people`.

```python
import cv2
no_people = cv2.imread('no_people.png', 0)
```

We need to initialize a 2D matrix called `total_usage` of the same size as our images (`1080 x 1920` in our case).  We'll be accumulating all the pixel differences in this matrix.

```python
import numpy as np
total_usage = np.zeros((1080, 1920))
```

Next, we loop through all the images in the `images` folder and accumulate the difference in the `total_usage` variable.

```python
import seaborn

img_locs = os.listdir('images')
img_locs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for img_loc in img_locs:
    frame = cv2.imread('images/{}'.format(img_loc), 0)
    total_usage += cv2.absdiff(no_people, frame)

map = seaborn.heatmap(total_usage).get_figure().savefig('heatmap.png', dpi=400)
```

In the end, `total_usage` is converted to a heatmap using [Seaborn](https://seaborn.pydata.org/) and saved as `heatmap.png`.

----

And there you have it. Easy image processing to find out some very interesting things about your space.