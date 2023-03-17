# TuringQ DNN

TuringQ DNN operator inventory.
The operations in the library is intended to replace the operations in [NEUROPHOX](https://github.com/solgaardlab/neurophox/blob/61fb6f78441176ff8e82a41fcd3e7778b0809e99/neurophox/torch/generic.py#L436-L447). 

Currently supported operations:

```python
tqdnn.cc_mul
```

More operations is on the way ...

## Installation

1. Download the source code into a local folder, e.g. `my_folder`;
2. `cd my_folder/QZone-Dnn & pip install -e .`
3. run `python -c 'import tqdnn'`, if no error shown, the installation is successful.

## Usage

Replace the operations in the file [neurophox/neurophox/torch/generic.py](https://github.com/solgaardlab/neurophox/blob/61fb6f78441176ff8e82a41fcd3e7778b0809e99/neurophox/torch/generic.py) with the ones provided in our library.

For example, replace the following `cc_mul`

```python
arg0 = ...
arg1 = ...
cc_mul(arg0, arg1)
```

with `tqdnn.cc_mul`

```python
import tqdnn

arg0 = ...
arg1 = ...
tqdnn.cc_mul(arg0, arg1)
```

That's it!
