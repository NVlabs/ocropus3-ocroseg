
# ocroseg

This is a deep learning model for page layout analysis / segmentation.

There are many different ways in which you can train and run it, but by default, it will simply return the text lines in a page image.

# Segmentation

Segmentation is carried out using the `ocroseg.Segmenter` class. This needs a model that you can download or train yourself.


```bash
%%bash
model=lowskew-000000259-011440.pt
test -f $model || wget --quiet -nd https://storage.googleapis.com/tmb-models/$model
```


```python
%pylab inline
rc("image", cmap="gray", interpolation="bicubic")
figsize(10, 10)
```

    Populating the interactive namespace from numpy and matplotlib


The `Segmenter` object handles page segmentation using a DL model.


```python
import ocroseg
seg = ocroseg.Segmenter("lowskew-000000259-011440.pt")
seg.model
```




    Sequential(
      (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True)
      (2): ReLU()
      (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
      (4): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
      (6): ReLU()
      (7): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
      (8): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
      (10): ReLU()
      (11): LSTM2(
        (hlstm): RowwiseLSTM(
          (lstm): LSTM(64, 32, bidirectional=1)
        )
        (vlstm): RowwiseLSTM(
          (lstm): LSTM(64, 32, bidirectional=1)
        )
      )
      (12): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
      (13): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
      (14): ReLU()
      (15): LSTM2(
        (hlstm): RowwiseLSTM(
          (lstm): LSTM(32, 32, bidirectional=1)
        )
        (vlstm): RowwiseLSTM(
          (lstm): LSTM(64, 32, bidirectional=1)
        )
      )
      (16): Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
      (17): Sigmoid()
    )



Let's segment a page with this.


```python
image = 1.0 - imread("testdata/W1P0.png")[:2000]
print image.shape
imshow(image)
```

    (2000, 2592)





    <matplotlib.image.AxesImage at 0x7f6078b09690>




![png](README_files/README_7_2.png)


The `extract_textlines` method returns a list of text line images, bounding boxes, etc.


```python
lines = seg.extract_textlines(image)
imshow(lines[0]['image'])
```




    <matplotlib.image.AxesImage at 0x7f60781c05d0>




![png](README_files/README_9_1.png)


The segmenter accomplishes this by predicting seeds for each text line. With a bit of mathematical morphology, these seeds are then extended into a text line segmentation.


```python
imshow(seg.lines)
```




    <matplotlib.image.AxesImage at 0x7f60781a5510>




![png](README_files/README_11_1.png)


# Training

The text line segmenter is trained using pairs of page images and line images stored in `tar` files.


```bash
%%bash
tar -ztvf testdata/framedlines.tgz | sed 6q
```

    -rw-rw-r-- tmb/tmb      110404 2017-03-19 16:47 A001BIN.framed.png
    -rw-rw-r-- tmb/tmb       10985 2017-03-16 16:15 A001BIN.lines.png
    -rw-rw-r-- tmb/tmb       74671 2017-03-19 16:47 A002BIN.framed.png
    -rw-rw-r-- tmb/tmb        8528 2017-03-16 16:15 A002BIN.lines.png
    -rw-rw-r-- tmb/tmb      147716 2017-03-19 16:47 A003BIN.framed.png
    -rw-rw-r-- tmb/tmb       12023 2017-03-16 16:15 A003BIN.lines.png


    tar: write error



```python
from dlinputs import tarrecords
sample = tarrecords.tariterator(open("testdata/framedlines.tgz")).next()
subplot(121); imshow(sample["framed.png"])
subplot(122); imshow(sample["lines.png"])
```




    <matplotlib.image.AxesImage at 0x7f60e3d9bc10>




![png](README_files/README_14_1.png)


There are also some tools for data augmentation.

Generally, you can train these kinds of segmenters on any kind of image data, though they work best on properly binarized, rotation and skew-normalized page images. Note that by conventions, pages are white on black. You need to make sure that the model you load matches the kinds of pages you are trying to segment.


The actual models used are pretty complex and require LSTMs to function well, but for demonstration purposes, let's define and use a tiny layout analysis model. Look in `bigmodel.py` for a realistic model.


```python
%%writefile tinymodel.py
def make_model():
    r = 3
    model = nn.Sequential(
        nn.Conv2d(1, 8, r, padding=r//2),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(8, 1, r, padding=r//2),
        nn.Sigmoid()
    )
    return model
```

    Writing tinymodel.py



```bash
%%bash
./ocroseg-train -d testdata/framedlines.tgz --maxtrain 10 -M tinymodel.py --display 0
```

    raw sample:
    __key__ 'A001BIN'
    __source__ 'testdata/framedlines.tgz'
    lines.png float32 (3300, 2592)
    png float32 (3300, 2592)
    
    preprocessed sample:
    __key__ <type 'list'> ['A002BIN']
    __source__ <type 'list'> ['testdata/framedlines.tgz']
    input float32 (1, 3300, 2592, 1)
    mask float32 (1, 3300, 2592, 1)
    output float32 (1, 3300, 2592, 1)
    
    ntrain 0
    model:
    Sequential(
      (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
      (3): Conv2d(8, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): Sigmoid()
    )
    
    0 0 ['A006BIN'] 0.24655306 ['A006BIN'] 0.31490618 0.55315816 lr 0.03
    1 1 ['A007BIN'] 0.24404158 ['A007BIN'] 0.30752876 0.54983306 lr 0.03
    2 2 ['A004BIN'] 0.24024434 ['A004BIN'] 0.31007746 0.54046077 lr 0.03
    3 3 ['A008BIN'] 0.23756175 ['A008BIN'] 0.30573484 0.5392694 lr 0.03
    4 4 ['A00LBIN'] 0.22300518 ['A00LBIN'] 0.28594157 0.52989864 lr 0.03
    5 5 ['A00MBIN'] 0.22032338 ['A00MBIN'] 0.28086954 0.52204597 lr 0.03
    6 6 ['A00DBIN'] 0.22794804 ['A00DBIN'] 0.27466372 0.512208 lr 0.03
    7 7 ['A009BIN'] 0.22404794 ['A009BIN'] 0.27621177 0.51116604 lr 0.03
    8 8 ['A001BIN'] 0.22008553 ['A001BIN'] 0.27836022 0.5008192 lr 0.03
    9 9 ['A00IBIN'] 0.21842314 ['A00IBIN'] 0.26755702 0.4992323 lr 0.03

