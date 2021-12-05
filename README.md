# Kubeflow Pipeline

## Train

기존 데이터 + 추가된 데이터로 학습을 한다.

- [x] Kubeflow Setting on Ubuntu 20.04
- [ ] 로컬에서 훈련 가능하게 만들기
  - [ ] 필요한 환경변수 기재
  - [ ] config, path -> argparse
- [ ] 데이터를 GCS에서 가져오고, 체크 포인트를 GCS에 저장하기 (input, output 데이터 위치 정의)
  - Opt1. Cloud Storage의 원본 데이터를 클러스터로 다운
  - Opt2. PV로 GCS 마운트
- [ ] Dockerize
  - Arguments

## Evaluation

테스트셋에 대해 평가를 하여 `mIoU`를 측정하고 기록한다.

## Model Checkpoint Update

1. 기존 학습했던 모델의 `mIoU`를 Cloud Storage에서 다운로드 한다.
2. 기존 학습했던 모델의 `mIoU`가 없거나, 신규 학습한 모델의 정확도가 일정 기준 이상 높다면 모델 자체와 정확도 정보를 Cloud Storage에 업로드 한다.
3. 클라우드 상의 모델 학습에서 정확도가 더 높아진 경우 TorchServe에 신규 버전으로 배포한다.

## Serving

## minikube setting

```
minikube start --cpus 4 --memory 7GB
```

## Port Forwarding

- `kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80`

## References

- [Kubeflow Manifest](https://github.com/kubeflow/manifests/tree/v1.4.0)
- [Kubeflow 1.4 설치 with Minikube](https://suwani.tistory.com/18)
- [AI Platform에서 구현하는 Kubeflow Pipelines 기반 ML 학습 및 배포 예제](https://medium.com/google-cloud-apac/gcp-ai-platform-%EC%97%90%EC%84%9C-%EA%B5%AC%ED%98%84%ED%95%98%EB%8A%94-kubeflow-pipelines-%EA%B8%B0%EB%B0%98-ml-%ED%95%99%EC%8A%B5-%EB%B0%8F-%EB%B0%B0%ED%8F%AC-%EC%98%88%EC%A0%9C-part-2-3-22b597f8d127)
- [Serving TorchServe in kubeflow pipelines](https://byeongjokim.github.io/posts/MLOps-Toy-Project-5/)
- [Kubeflow Pipelines – 파이프라인에서 외부 저장소를 이용하기](https://kangwoo.kr/2020/04/04/kubeflow-pipelines-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8%EC%97%90%EC%84%9C-%EC%99%B8%EB%B6%80-%EC%A0%80%EC%9E%A5%EC%86%8C%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%98%EA%B8%B0/)
