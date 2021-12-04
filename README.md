# Kubeflow Pipeline

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
