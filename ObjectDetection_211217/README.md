```
Colab으로 진행하여고 했는데 데이터가 다 올라가지 않아 vscode + anaconda 환경에서 진행
```

## VSCODE에 Anaconda activate (CMD창 필요 Ctrl+shift+`)

# 코드구성
- * [Dataload](#dataload)
- EDA & 시각화
- * 데이터셋 정의
- * Model 구성
- * Class, 함수 정의
- * 학습
- 시각화

> * 는 필수요소*
------

## Dataload
1. path 설정
```python
train_files = sorted(glob('./data/train/*')) 
test_files = sorted(glob('./data/test/*'))
```
- sorted(glob.glob()) : 파일을 정렬해서 데려온다
2. json load
```python
train_json_list = []
for file in tqdm(train_files):
  with open(file, "r") as json_file:
    train_json_list.append(json.load(json_file))

test_json_list = []
for file in tqdm(test_file):
  with file in tqdm(test_files):
    test_json_list.append(json.load(json_file))
```
- tqdm 은 이터러블을 감싸면 진행 된 상태를 표시해준다
- tqdm($) : $로는 list 도 되고 100같은 숫자도 가능하고, dsec 같은 옵션도 가능하다 


##  EDA & 시각화
1. EDA
```python
label_count = {}
for data in train_json_list:
    for shape in data['shapes']:
        try:
            label_count[shape['label']]+=1
        except:
            label_count[shape['label']]=1
```

2. 시각화
```python
plt.figure(figsize=(25,30))
for i in range(30):
    plt.subplot(6,5,i+1)
    # base64 형식을 array로 변환
    img = Image.open(BytesIO(base64.b64decode(train_json_list[i]['imageData'])))
    img = np.array(img, np.uint8)
    title = []
    for shape in train_json_list[i]['shapes']:
        points = np.array(shape['points'], np.int32)
        cv2.polylines(img, [points], True, (0,255,0), 3)
        title.append(shape['label'])
    title = ','.join(title)
    plt.imshow(img)
    plt.subplot(6,5,i+1).set_title(title)
plt.show()
```

## 데이터셋

