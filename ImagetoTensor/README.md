> Torch 에서 이미지를 사용하기 위해서는 numpy array 형태로 불러온 후 torch.Tensor로 변환해 주어야함

## 1. 이미지 파일을 텐서로 변환 및 정규화 
```python
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

trans = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))]
trainset = torchvision.datasets.ImageFolder(root = f'{DIR}', transform = trans)
```

## 2. DataLoader 네트워크에 올리기
```python
trainloader = DataLoader(trainset, batch_size=5, shuffle=False, num_workers=2)
dataiter = iter(trainloader)
images = dataiter.next()
print(images)
```
> Image folder로 부터 불러온 trainset을 Dataloader를 통해 batch형식으로 네트워크에 올리는 방법이다.

## Tip
```python
plt.ion() # 반응형 모드
```

## Tip Pandas
> Pandas DataFrame에서 특정 행,열을 선택하는 방법은 여러가지임
> 행번호(row number)로 선택하는 방법 (.iloc)[출처](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html)
> label or 조건표현으로 선택 (.loc)

### 1. 행번호(row number)
```python
data.iloc[0] # data의 첫번째 행만
data.iloc[1] # 두번째 행만
data.iloc[-1] # 마지막 행만
# Columns:
data.iloc[:,0] # 첫번째 열만
data.iloc[:,1] # 두번째 열만
data.iloc[:,-1] # 마지막 열만
data.iloc[0:5] # 첫 5개행만

data.iloc[:, 0:2] # 첫 2개열만
data.iloc[[0,3,6,24], [0,5,6]] # 1st, 4th, 7th, 25th 행과 + 1st 6th 7th 열만
data.iloc[0:5, 5:8] # 첫 5개 행과 5th, 6th, 7th 열만
```
> iloc를 사용할 때 Series 데이터가 될 수도 , Dataframe으로 출력 될 수도 있다
> 예시를 들자면

```python
import pandas as pd
mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
          {'a': 100, 'b': 200, 'c': 300, 'd': 400},
          {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
df = pd.DataFrame(mydict)
print(df)
```
```
# 출력
      a     b     c     d
0     1     2     3     4
1   100   200   300   400
2  1000  2000  3000  4000
```
```
print(type(df.iloc[0]))
print(df.iloc[0])
```
```
<class 'pandas.core.series.Series'>
a    1
b    2
c    3
d    4
Name: 0, dtype: int64
```
```
print(type(df.iloc[[0]]))
print(df.iloc[[0]])
```
```
<class 'pandas.core.frame.DataFrame'>

   a  b  c  d
0  1  2  3  4
```
> 위처럼 [[]]로 출력 하면 series 형태가 아닌 dataframe 형태로 출력이 된다

```python
df.iloc[[0, 1]]

"""
     a    b    c    d
0    1    2    3    4
1  100  200  300  400
"""
```



