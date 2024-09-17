# Cognitive Kernel

<p align="center">
  <picture>
    <img alt="CK" src="docs/resources/CK_logo.png" width=20%>
  </picture>
</p>

<h3 align="center">
An Early-stage “Autopilot” System for Everyone
</h3>

<p align="center">
| <a href="https://tencent-ailab.github.io/CogKernel"><b>Project Page</b></a> | <a href="http://arxiv.org/abs/2409.10277"><b>Paper</b></a> |

</p>
Cognitive Kernel is an open-sourced agent system designed to achieve the goal of building general-purpose autopilot systems.
It has access to real-time and private information to finish real-world tasks.

In this repo, we release both the system and the backbone model to encourage further research on LLM-driven autopilot systems.
Following the guide, everyone should be able to deploy a private 'autopilot' system on their own machines.

## Installation

### Step 1: Start the model server

1. Download the model weights from the following resources.

<div style="margin-left: 60px;">

| Model Name              | Model Size |                                                  Model Link |
| :---------------------- | :--------: | ----------------------------------------------------------: |
| llama3_policy_model_8b  |     8B     |     [link](https://huggingface.co/CognitiveKernel/ck-8b-v1) |
| llama3_policy_model_70b |    70B     |    [link](https://huggingface.co/CognitiveKernel/ck-70b-v1) |
| helper_model_1b         |     1B     | [link](https://huggingface.co/CognitiveKernel/helper_1b_v1) |

</div>

2. Start the model servers

<div style="margin-left: 40px;">

- 2.1 Start the main policy model

  - Step 1: Install the vllm package.
    ```
    pip install vllm==0.5.4
    ```
  - Step 2: Launch the vllm service.
    ```
    python -m vllm.entrypoints.openai.api_server --model path/to/downloaded/policy/model --worker-use-ray --tensor-parallel-size 8 --port your_port --host 0.0.0.0 --trust-remote-code --max-model-len 8192 --served-model-name ck
    ```

- 2.2 Start the Helper LLM

  - Step 1: Install the TGI framework.

  - Step 2: Launch the TGI service:
    ```
    CUDA_VISIBLE_DEVICES=YOUR_GPU_ID text-generation-launcher --model-id PATH_TO_YOUR_HELPER_LLM_CHECKPOINT --port YOUR_PORT --num-shard 1 --disable-custom-kernels
    ```
  - Step 3: Configure service URLs.

    Update the `service_url_config.json` file. Replace the values for the following keys with the IP address and port of your Helper LLM instance:

    - `concept_perspective_generation`
    - `proposition_generation`
    - `concept_identification`
    - `filter_doc`
    - `dialog_summarization`

- 2.3 Start the text embedding model

  - Step 1: Install the required dependencies:

    Install `cherrypy` and `sentence_transformers`.

  - Step 2: Launch the embedding service:
    ```
    python text_embed_service.py --model gte_large --gpu YOUR_GPU_ID --port YOUR_PORT --batch_size 128
    ```
  - Step 3: Configure service URLs:

    Update the `service_url_config.json` file. Replace the value of `sentence_encoding` with the IP address and port of your text embedding service instance.

</div>

3. Edit the configuaration files:

   docker-compose.yml:

   - add the "IP:Port" of the policy model (8b/70b) servers to SERVICE_IP;

### Step 2: Start the system

1. Download and install the Docker desktop/engine

2. Go to the repo folder and then

<div style="margin-left: 40px;">

```
docker-compose build
```

or

```
docker compose build
```

</div>

3. Start the whole system with

<div style="margin-left: 40px;">

```
docker-compose up
```

or

```
docker compose build
```

</div>

Then, you should be able to play with the system from your local machine at (http://0.0.0.0:8080). PS: for the first time you start the system, you might observe an error on some machines. This is because the database docker takes some time to initiate. Just kill it and restart, the error should be gone.

## System Demonstration

Demo 1: Search for citations of an uploaded paper on Google Scholar.

<!-- ![Demo 1](https://raw.githubusercontent.com/tencent-ailab/CogKernel/main/docs/static/images/Case1.mp4) -->

<div align="center">
  <a href="https://youtu.be/mJ_ne4jxXM0">
    <img src="https://img.youtube.com/vi/mJ_ne4jxXM0/maxresdefault.jpg" style="width: 400px"/>
  </a>
</div>

<br>

Demo 2: Download a scientific paper and ask related questions.

<br>

<div align="center">
  <a href="https://youtu.be/vZ4GEwIas-o">
    <img src="https://img.youtube.com/vi/vZ4GEwIas-o/maxresdefault.jpg" style="width: 400px"/>
  </a>
</div>
<!-- ![Demo 2](https://github.com/tencent-ailab/CogKernel/docs/static/images/Case2.mp4) -->

<!-- <div>
<video width="85%" controls>
        <source src="https://raw.githubusercontent.com/tencent-ailab/CogKernel/main/docs/static/images/Case2.mp4" type="video/mp4">
</video>
</div> -->

## Contribution

We welcome and appreciate any contributions and collaborations from the community.

## Disclaimer

Cognitive Kernel is **not** a Tencent product and should only be used for research purpose.
Users should use Cognitive kernel carefully and be responsible for any potential consequences.

## Contact

For technical questions or suggestions, please use Github issues or Discussions.

For other issues, please contact us via [cognitivekernel AT gmail.com](cognitivekernel@gmail.com).
