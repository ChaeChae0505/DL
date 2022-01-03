- [이거 읽어보자](https://gaussian37.github.io/dl-pytorch-snippets/)
```

——————– 기본 문법 및 연산 관련 ——————–
Pytorch 란?
PyTorch 패키지의 구성 요소
PyTorch 기초 사용법
텐서의 생성과 변환
텐서의 인덱스 조작
텐서 연산
텐서의 차원 조작
Tensor 생성
Tensor 데이터 타입
Numpy to Tensor 또는 Tensor to Numpy
CPU 타입과 GPU 타입의 Tensor
Tensor 사이즈 확인하기
Index (slicing) 기능 사용방법
Join(cat, stack) 기능 사용 방법
slicing 기능 사용 방법
squeezing 기능 사용 방법
Initialization, 초기화 방법
Math Operation
Gradient를 구하는 방법
벡터와 텐서의 element-wise multiplication
gather 기능 사용 방법
expand와 repeat 기능 사용 방법
topk 기능 사용 방법


——————– 셋팅 및 문법 관련 ——————–
pytorch import 모음
pytorch 셋팅 관련 코드
GPU 셋팅 관련 코드
dataloader의 num_workers 지정
dataloader의 pin_memory
GPU 사용 시 data.cuda(non_blocking=True) 사용
optimizer.zero_grad(), loss.backward(), optimizer.step()
optimizer.step()을 통한 파라미터 업데이트와 loss.backward()와의 관계
gradient를 직접 zero로 셋팅하는 이유와 활용 방법
validation의 Loss 계산 시 detach 사용 관련
model.eval()와 torch.no_grad() 비교
Dropout 적용 시 Tensor 값 변경 메커니즘
재현을 위한 랜덤 seed값 고정
contiguous()의 의미


——————– 자주사용하는 함수 ——————–
torch.argmax(input, dim, keepdim)
Numpy → Tensor : torch.from_numpy(numpy.ndarray)
Tensor → Numpy
torch.unsqueeze(input, dim)
torch.squeeze(input, dim)
Variable(data)
F.interpolate()와 nn.Upsample()
block을 쌓기 위한 Module, Sequential, ModuleList, ModuleDict
shape 변경을 위한 transpose
permute를 이용한 shape 변경
nn.Dropout vs. F.dropout
nn.AvgPool2d vs. nn.AdaptiveAvgPool2d
optimizer.state_dict() 저장 결과
torch.einsum 함수 사용 예제
torch.softmax 함수 사용 예제
torch.repeat 함수 사용 예제
torch.scatter 함수 사용 예제
torch.split 함수 사용 예제
torch.nan_to_num 함수 사용 예제


——————– 자주 사용하는 응용 코드 모음 ——————–
파이썬 파일을 읽어서 네트워크 객체 생성
weight 초기화 방법
load와 save 방법
Dataloader 사용 방법
pre-trained model 사용 방법
pre-trained model 수정 방법
checkpoint 값 변경 후 저장
Learning Rate Scheduler 사용 방법
model의 parameter 확인 방법
Tensor 깊은 복사
일부 weight만 업데이트 하는 방법
OpenCV로 입력 받은 이미지 torch 형태로 변경
label을 이용하여 one hot 생성


——————– 효율적인 코드 사용 모음 ——————–
convolution - batchnorm 사용 시, convolution bias 사용 하지 않음
```
- [이 글](https://anweh.tistory.com/21) 과 유사함 이글을 보는 것을 추천 요고는 저를 위한 정리 ~!!!
- PoseCNN을 이해하는 과정에서 module을 어떻게 만드는지 궁금해서 찾아보게 됨


# Pytorch 신경망 모델의 정의
- pytorch로 신경망 모델을 생성할 때 크게 세가지 스템을 따르면 된다
```
1. Design your model using class with Variables
2. Construct loss and optim
3. Train cycle(forward, backward, update)
```
## 신경망 모델의 정의 방법
- 사용자 정의 nn 모듈
- nn.Module을 상속한 클래스 이용[참고](https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html)
