```
Colab으로 진행하여고 했는데 데이터가 다 올라가지 않아 vscode + anaconda 환경에서 진행
```

## VSCODE에 Anaconda activate (CMD창 필요 Ctrl+shift+`)

# 코드구성
- *[Dataload](#dataload)*
- EDA & 시각화
- *데이터셋 정의*
- *Model 구성*
- *Class, 함수 정의*
- *학습*
- 시각화

> *는 필수요소*
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
- torch에서 데이터 셋을 사용하기 위해서는 PIL, openCV로 불러온 data를 numpy array로 변환 후 torch.Tensor로 변환해 주어야한다.
```python
class CustomDataset(Dataset):
    def __init__(self, json_list, mode='train'):
        self.mode = mode
        self.file_name = [json_file['file_name'] for json_file in json_list]
        if mode == 'train':
            self.labels = []
            for data in json_list:
                label = []
                for shapes in data['shapes']:
                    label.append(shapes['label'])
                self.labels.append(label)
            self.points = []
            for data in json_list:
                point = []
                for shapes in data['shapes']:
                    point.append(shapes['points'])
                self.points.append(point)
        self.imgs = [data['imageData'] for data in json_list]
        
        self.widths = [data['imageWidth'] for data in json_list]
        self.heights = [data['imageHeight'] for data in json_list]
        
        self.label_map ={
            '01_ulcer':1, '02_mass':2, '04_lymph':3, '05_bleeding':4
        }
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        file_name = self.file_name[i]
        img = Image.open(BytesIO(base64.b64decode(self.imgs[i])))
        img = self.transforms(img)
        
        target = {}
        if self.mode == 'train':
            boxes = []
            for point in self.points[i]:
                x_min = int(np.min(np.array(point)[:,0]))
                x_max = int(np.max(np.array(point)[:,0]))
                y_min = int(np.min(np.array(point)[:,1]))
                y_max = int(np.max(np.array(point)[:,1]))
                boxes.append([x_min, y_min, x_max, y_max])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            label = [self.label_map[label] for label in self.labels[i]]

            masks = []
            for box in boxes:
                mask = np.zeros([int(self.heights[i]), int(self.widths[i])], np.uint8)
                masks.append(cv2.rectangle(mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 1, -1))

            masks = torch.tensor(masks, dtype=torch.uint8)

            target["boxes"] = boxes
            target["labels"] = torch.tensor(label, dtype=torch.int64)
            target["masks"] = masks
            target["area"] = area
            target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([i], dtype=torch.int64)
        if self.mode == 'test':
            target["file_name"] = file_name
        return img, target
```

