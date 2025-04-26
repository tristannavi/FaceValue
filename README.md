# FaceValue

FaceValue is a general framework for assessing how Large Multimodal Models (LMMs) perceive human subjects based on
visual inputs. Our framework enables systematic probing of model perceptions across a range of user-defined social
traits and is lightweight, reproducible, and adaptable to different datasets and models.

The Jupyter notebook was designed for Google Colab. Anthropic and OpenAI API
keys are required to run the notebook. The easiest way to run the notebook is to use Google Colab, however, you can run
it locally as well. If you intend to run it locally, you will need to install the required packages and set up the
environment.

## Installation

Clone the repository and navigate to the directory:

```bash
git clone https://github.com/facevalue/facevalue.git
cd facevalue
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the notebook

To run the notebook locally, you will need to replace the API keys in the notebook with your own. You will also need to
remove references to the Google libraries.