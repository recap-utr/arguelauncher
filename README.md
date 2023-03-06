# ReCAP: Argument Graph Retrieval

## System Requirements and Installation

This app is developed using [poetry](https://python-poetry.org).
In addition, we provide a docker-compose configuration for easy execution.
In the following, we will only provide the `docker-compose` commands.

To run the services, you need to create a file called `.env` in the project folder with the following contents:

```sh
ARGUEQUERY_NLP_HOST="nlp-service"
ARGUEQUERY_NLP_PORT="5678"
ARGUEQUERY_RETRIEVAL_HOST="retrieval-service"
ARGUEQUERY_RETRIEVAL_PORT="6789"
```

## Server

If you want to perform a case-base retrieval with your own dataset, you only need to start the retrieval server together with the NLP server:

`docker-compose up nlp-service retrieval-service`

It will then listen on the port `ARGUEQUERY_RETRIEVAL_PORT`.

## Optional: Evaluation with Built-In Client

In addition to the retrieval server, we also provide a client to automatically evaluate the retrieval results.
In this case, you can start the whole stack:

`docker-compose up`

The evaluation relies on some data to be available.
Details can be found in the following section.

### Case Base and Queries

All files need to be present in a format that can be parsed by the [arguebuf](https://pypi.org/project/arguebuf/) library.

### Queries and Benchmark Data

For each query with name `NAME.json`, a file `NAME.json` with benchmark ranking has to be created.
This data is used to calculate the evaluation metrics.

```json
{
  "candidates": [
    "microtexts/nodeset6366.json",
    "microtexts/nodeset6383.json",
    "microtexts/nodeset6387.json",
    "microtexts/nodeset6391.json",
    "microtexts/nodeset6450.json",
    "microtexts/nodeset6453.json",
    "microtexts/nodeset6464.json",
    "microtexts/nodeset6469.json"
  ],
  "rankings": {
    "microtexts/nodeset6366.json": 2,
    "microtexts/nodeset6383.json": 2,
    "microtexts/nodeset6387.json": 3,
    "microtexts/nodeset6391.json": 2,
    "microtexts/nodeset6450.json": 2,
    "microtexts/nodeset6453.json": 2,
    "microtexts/nodeset6464.json": 2,
    "microtexts/nodeset6469.json": 1
  }
}
```

_Please note:_ `candidates` and `rankings` do not need to contain the same filenames.

## Further Information

- [License](LICENSE)
- [Copyright](NOTICE.md)
