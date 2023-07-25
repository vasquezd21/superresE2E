# An End-to-End Guide to Optimizing and Deploying Super-Resolution Models with Olive and ONNX Runtime Web

Olive serves as an advanced optimization tool, enabling developers to adapt and evaluate models by leveraging the most effective optimization techniques. This demonstration emphasizes how any developer can effortlessly optimize and convert a model using Olive, before deploying it via the ONNX Runtime Web application.

In this case, we chose a super-resolution model from the ONNX Zoo library to demonstrate the seamless end-to-end developer experience provided by Olive. With a single command line, we optimized a super-resolution CNN model through Olive passes, making it ready for integration into any application the developer wishes. The smooth optimization and evaluation process facilitated by Olive is versatile, applicable to any model.

For additional details about the super-resolution model, please visit [here](https://github.com/onnx/models/tree/main/vision/super_resolution/sub_pixel_cnn_2016).

## Setup
We recommend setting up a WSL development environment. You can follow these instructions on how to install WSL with Ubuntu [Install WSL | Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/install).

Once WSL is ready, install Olive from source in your Linux environment with the following commands:

```shell
git clone https://github.com/microsoft/Olive.git
cd Olive
python -m pip install .
```

Create a conda virtual environment with the required installations.

```shell
conda create -n oliveE2EDemo python=3.8
conda activate oliveE2EDemo
pip install olive-ai
```

## Olive Conversion and Optimization 

Once everything is set up, execute the following command to export the super-resolution PyTorch model to ONNX using Olive:

```
python3 -m olive.workflows.run --config config.json
```

This command executes the `oliveconfig.json` and `oliveloader.py` files. Olive offers users the flexibility to configure the optimization and exportation of a model according to its specifications and additional requirements using passes. The passes are located in the `oliveconfig.json` file, and are set to export the model using the OnnxConversion pass and optimize it with `AppendPrePostProcessingOps`. 

### Configuring for Olive using Passes
Olive requires comprehensive information about your model, such as the method to load the model and the name and shape of input tensors. Furthermore, you can specify your target hardware and the series of optimizations you intend to apply to the model. All this information can be conveniently provided in a JSON file, which then serves as the input to Olive.

Olive is capable of implementing various transformations and optimizations, known as passes, on the input model to produce an accelerated output model. For the full list of passes supported by Olive, please refer to the 'Passes' section [here](https://microsoft.github.io/Olive/api/passes.html#passes). The utilization of a JSON configuration file in conjunction with a helper file, which contains the model's specifics, is illustrated below.



```json
    "passes": {
        "exporter": {
            "type": "OnnxConversion",
            "config": {
                "target_opset": 14
            }
        },
        "prepost": {
            "type": "AppendPrePostProcessingOps",
            "config": {
                "tool_command": "superresolution",
                "tool_command_args": {
                    "output_format": "png"
                }
            }
        }
    },
```

The `oliveloader.py` file serves as a helper script, supplying Olive with necessary details and specifications about the super-resolution model.

```py
def load_pytorch_model(model_path: str) -> nn.Module:
    torch_model = SuperResolutionNet(upscale_factor=3)

    model_url = "https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth"

    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=lambda storage, loc: storage))

    torch_model.eval()

    return torch_model
```


## Inferencing and Deploying on ORT Web
Once the model has undergone conversion and optimization via Olive, it is ready to be utilized in an application, in this scenario, ORT Web. This application carries out inferencing within the browser, using the `onnxruntime-web` JS library. The two primary files involved in this process are `modelHelper.ts` and `ImageCanvas.tsx`, located in the utils and components folders, respectively.


### Inferencing
The `modelHelper.ts` file constructs and executes an `ort.InferenceSession` by providing the path to the super-resolution model and the `SessionOptions`. In this particular scenario, the execution provider chosen was `wasm` for CPU utilization, although `webgl` can alternatively be used for GPU.

```ts
import * as ort from 'onnxruntime-web';

export async function runSuperResModel(preprocessedData: any): Promise<[ort.Tensor, number]> {

  // Create session and set options. See the docs here for more options: 
  //https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#graphOptimizationLevel
  const session = await ort.InferenceSession
    .create('./_next/static/chunks/pages/superres_0_model.onnx',
      { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  console.log('Inference session created')

  var [results, inferenceTime] = await runInference(session, preprocessedData);

  return [results, inferenceTime];
}
```

The `runInference` method displays the high-definition image and the time it took for inference on the `ImageCanvas` web component

```ts
async function runInference(session: ort.InferenceSession, preprocessedData: ort.Tensor): Promise<[ort.Tensor, number]> {

  const start = new Date();

  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = preprocessedData;

  const outputData = await session.run(feeds);

  const end = new Date();

  const inferenceTime = (end.getTime() - start.getTime()) / 1000;

  const output = outputData[session.outputNames[0]];

  return [output, inferenceTime];
}
```
### Displaying High Resolution Images
The `ImageCanvas.tsx` file has the web components to display the run inference button and both low resolution and high resolution images. `ImageCanvas` also has the logic on how to convert the inputted png to a tensor so to run inferencing on and afterwards converting the tensor back into an image to display. 


<!-- Here the inputted png data is accessed and we normalize the data to meet the requirements of the model. Once we access the image data, we can then convert it to an ORT Tensor for inferencing. Afterwards the inputted low resolution image is displayed on the canvas and `submitInference` is called on the inputted tensor so that higher resolution image can be displayed. We do this so we can run inference on the main image. The file parses the image and sends it to submit inference and generate the higher-resolution image. The new higher-resolution image is displayed below the original image to serve as comparision.  -->

In this process, we take the inputted PNG data, and normalize it to align with our model's requirements. Once we've accessed the image data, it's then converted into an ORT Tensor, ready for inferencing. The next step involves displaying the original, low-resolution image on the canvas. At this stage, we call the submitInference function on the inputted tensor, enabling the creation of a higher-resolution version of the image. This is specifically done to allow us to run inference on the main image. 

The process continues with the parsing of the image, which is then passed to `submitInference`. This step leads to the creation of the high-resolution image. To show the improvements, we display the newly created high-resolution image below the original image. 

```tsx
  const displayImageAndRunInference = async () => {
    var sampleImage = getImage();
    var imgElement = new Image();
    imgElement.src = sampleImage;

    const canvas = canvasRef.current;
    const ctx = canvas!.getContext('2d');

    const imageData = ctx.getImageData(0, 0, 224, 224)

    const rgbData = imageData.data
    let normalizedData = new Float32Array(rgbData.length / 4)
    for (let i = 0; i < normalizedData.length; i++) {
      normalizedData[i] = Math.round(0.299 * rgbData[i * 4] + 0.587 * rgbData[i * 4 + 1] + 0.114 * rgbData[i * 4 + 2]) / 255.0;
    }

    const InputTensor = new Tensor('float32', normalizedData, [1, 1, 224, 224]); 


    imgElement.onload = async () => {
      ctx!.drawImage(imgElement, 0, 0, props.width, props.height);
      await submitInference(InputTensor);
    }
  };
```

From there, `submitInference` calls `runSuperResModel` which returns a promise that is awaited. The purpose of this function is to return array with two elements: the output tensor and the inference time. The output data is drawn onto a canvas using the `putImageData` method which utlimately displays the image post super-resolution. 

```ts
  const submitInference = async (imgElement) => {
    // Run inference
    await runSuperResModel(imgElement).then(async (res) => {
      const OutputTensor = res[0];
      const inferenceTime = res[1];
      const outputData = OutputTensor.toImageData();
      console.log(outputData)
      setInferenceTime(`Inference speed: ${inferenceTime} seconds`);
      const supercanvas = superResCanvasRef.current;
      const superctx = supercanvas!.getContext('2d');
      superctx.putImageData(outputData, 0, 0);
    })
  };
```

### Wrapping Up
This tutorial illustrates the capabilities of Olive and ONNX Runtime Web in enhancing the optimization and deployment of machine learning models. Leveraging Olive's capabilities, we are able to effectively streamline our models, ensuring they are as efficient  as possible. Coupling this with ONNX Runtime Web allows for an impressive end-to-end deployment directly within a web browser, simplifying the process and ensuring user accessability. 

