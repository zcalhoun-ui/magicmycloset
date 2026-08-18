## Welcome to magicmycloset! 
This project combines algorithms, web design, and fashion. Using my actual clothing and outfits, I built a sparse tensor of top, bottom, and accessories/layering combinations to recommend new outfits. 

I envision this project being visible as a mildly interactive website.

August 18th, 2026 Update
Currently, I have completed the following steps:

1. Initial data collection: I built two spreadsheets, one with each item and some metadata, the second with 270 existing outfits.
2. I filtered and cleaned these data sets and converted the second into a tensor.
3. I ran the tensor through SVD and LRA to return the most recommended future outfits.
4. I left-joined the tensor to the recommendations data frame to show the names of each clothing item. 

After this initial step, I realized a crucial error in the data frame I presented to this algorithm. See, SVD and LRA works by identifying the most crucial connections to compress to a smaller version of your matrix combinations. With a chosen k number of ranks, it then rebuilds the full matrix with the compressed version. Instead of 1s and 0s, the matrix cells are filled with decimals between 1 and 0, with a larger number representing a greater percent liklihood that there could be a connection between the items. By filtering out previous connections and sorting by descending decimal, you identify the most likely or "recommended" new connecitons. Pretty clever, right? This algoirthm has been used by tons of tech companies; its the basis for google's page rank, which shaped the internet as we know it today.  
However, my data set actually contains three types of outfits, rated in a fourth column. My next algorithm will introduce rating— some outfits get a 1, some a -1, and some a 0. That means many of the outfits I tested my algorithm on *were connections I didn't want to happen.* This strategy only looked at 3 dimensions, or columns from my original sheet, so it had no idea of this distinction. I basically poisoned it. 

That probably explains why some of its top outfits sound really cute, like  [blue cardigan, Flare jeans, fruit shirt] (in fact, I wore this shirt and pants in an instagram post), while others [orange corderoy, baggy jorts, grey v-neck sweater]... not so much.

I need to fix my python writing so that it is more function based and easy to read. Once I do that, I will re-run all of this on a filtered to rating == 1 or 0 version of the first data set.

Next, run the tensor factorization algorithm!

Next: build out infrastructure some. 

And: learn web programming I guess? I must admit that doesn't really interest me that much. 

Okay: current files

1. first.py: will build tensor factorization here
2. creating-an-encirment.md: this markdown file
3. identifier.csv: lists each clothing item, its identifier in the matrix, and some fun information to come like what seasons it is appropriate for
4. puretensor.py: simple svd here
5. matrix_numerical.csv: redo of matrix without any characters in numbers
6. matrix.csv: first matrix. i unfortunately built this with unique identifiers containing letters, so had to be converted for matrix math
   
