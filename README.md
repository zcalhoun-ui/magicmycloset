# Welcome to magicmycloset! 
This project combines algorithms, web design, and fashion. Using my actual clothing and outfits, I built a sparse tensor of top, bottom, and accessories/layering combinations to recommend new outfits. 

I envision this project being visible as a mildly interactive website.

### August 18th, 2026 Update
Currently, I have completed the following steps:

1. Initial data collection: I built two spreadsheets, one with each item and some metadata, the second with 270 existing outfits.
2. I filtered and cleaned these data sets and converted the second into a tensor.
3. I decomposed the tensor using tucker-d decomposition
4. I ran the tensor through SVD and LRA to return the most recommended future outfits.
5. I left-joined the tensor to the recommendations data frame to show the names of each clothing item. 

After this initial step, I realized a crucial error in the data frame I presented to this algorithm. See, SVD and LRA works by identifying the most crucial connections to compress to a smaller version of your matrix combinations. With a chosen k number of ranks, it then rebuilds the full matrix with the compressed version. Instead of 1s and 0s, the matrix cells are filled with decimals between 1 and 0, with a larger number representing a greater percent liklihood that there could be a connection between the items. By filtering out previous connections and sorting by descending decimal, you identify the most likely or "recommended" new connecitons. Pretty clever, right? This algoirthm has been used by tons of tech companies; its the basis for google's page rank, which shaped the internet as we know it today.  
However, my data set actually contains three types of outfits, rated in a fourth column. My next algorithm will introduce rating— some outfits get a 1, some a -1, and some a 0. That means many of the outfits I tested my algorithm on *were connections I didn't want to happen.* This strategy only looked at 3 dimensions, or columns from my original sheet, so it had no idea of this distinction. I basically poisoned it. 

That probably explains why some of its top outfits sound really cute, like  [blue cardigan, Flare jeans, fruit shirt] (in fact, I wore this shirt and pants in an instagram post), while others [orange corderoy, baggy jorts, grey v-neck sweater]... not so much.

I need to fix my python writing so that it is more function based and easy to read. Once I do that, I will re-run all of this on a filtered to rating == 1 or 0 version of the first data set.


### Updated August 19th, 2026
Yesterday, I ran the SVD algorithm with the -1s and 0s filtered out, and it did much better.

Next, I decided to build a small machine learning algorithm. Most of the prep code is exactly the same, except that I used PARAFAC decomposition instead of tucker-d. 

I did a lot of research to try and understand this better. Essentially, my model looks at my existing outfits and their ratings (either a 1 or a -1). It generates a normal curve of possible values for the ratings and runs each of them through a loss function, where numbers closer to the ideal 1 or -1 are scored higher. It then generates its predicted score for each of my combinations. A second function collects the top predicted scores and ranks them. 

The problem is, it doesn't really work. I mean, it does, but its recommendations are objectively horrible. It recommended mittens as an accessory for every single outfit? Or green sweatpants as the bottom in every single outfit? From my understanding, whats happening here is that my data set is too evenly balanced and too small. There's a roughly equal number of outfits with each item, and there really aren't very many outfits. So, for each seed, it just picks the first thing it sees and gets stuck on that. I know this because if I change the seed, then it just picks a new item. I have tried increasing the lambda penalty and decreasing the calculation speed. These changes made marginal improvements, but if I reset the seed than it started all over again. 

I could just hard-code the output to only include completely new combinations, but that feels like a cop-out. I'd rather improve on my actual algorithm, so I suppose that's next. 

Next: build out infrastructure some. 

And: learn web programming I guess? I must admit that doesn't really interest me that much. 

Okay: current files

1. first.py: ml algo
2. creating-an-encirment.md: this markdown file
3. identifier.csv: lists each clothing item, its identifier in the matrix, and some fun information to come like what seasons it is appropriate for
4. puretensor.py: simple svd here
5. matrix_numerical.csv: redo of matrix without any characters in numbers
6. matrix.csv: first matrix. i unfortunately built this with unique identifiers containing letters, so had to be converted for matrix math
### Updated Sep 3 
Was so nice to work on this again! Have been active on frontend and art mainly. Have not been active enough at all because I unfortunately have a life. Except today lolol this is like 6 hours worth of work. 
I added the filtering.py file. This is the same lra algorithm, have decided to focus on this, but restructured better in functions with a main(). I also mocked up taking in a season input from the user. It runs the algorithm using only the items marked to be appropriate for that season, so its noticing different connections in every season. This does make the overall thing a little more separated, might be nice to have a general option too? Unfortunately, the current code is very built around the seasonal version. Could possibly keep the filtering and puretensor files and go between them. Anyways, happy with this work! Ignore krita files, I forgot to add them to the .gitignore </3
