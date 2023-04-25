# Evaluation Framework for Argumentation Machines

This application holds test cases and calls argumentation microservices for the purpose of computing evaluation metrics.
Thus, one needs to set up the corresponding services according to their documentation first.
Then, we recommend to use docker to start this app.

```shell
docker-compose build
docker-compose up
```

The app is configured via [Hydra](https://hydra.cc).
To change the parameters, we recommend to create the file `arguelauncher/config/app.local.yaml` and put overrides there.
For instance, to evaluate an adaptation approach:

```yaml
defaults:
  - app
  - _self_
path:
  requests: data/requests/microtexts-generalization
retrieval:
  mac: false
  fac: false
nlp_config: STRF
adaptation:
  extras:
    type: openai-chat-hybrid
```

More documentation will follow for the final version.
