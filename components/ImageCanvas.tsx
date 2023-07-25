import { useRef, useState } from 'react';
import { getImageTensorFromPath } from '../utils/predict';
import styles from '../styles/Home.module.css';
import { runSuperResModel } from '../utils/modelHelper';
import { tensorToImageData } from '../utils/imageHelper';
import React from 'react';
import { Tensor } from 'onnxruntime-web';
// Import images

interface Props {
  height: number;
  width: number;
}

const ImageCanvas = (props: Props) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // new canvas ref for super res image
  const superResCanvasRef = useRef<HTMLCanvasElement>(null);
  const [inferenceTime, setInferenceTime] = useState("");

  //Create an array of images
  const imageCount = 91;
  const allImages = Array.from({ length: imageCount }, (_, i) => `public/t1${i + 1}.png`);

  // Randomly loads the image from the T91 list
  const getImage = () => {
    const imageIndex = Math.floor(Math.random() * 10) + 1;
    return `/t${imageIndex}.png`;
  }

  // Draw image and other UI elements then run inference
  const displayImageAndRunInference = async () => {
    // Get the image
    var sampleImage = getImage();
    var imgElement = new Image();
    imgElement.src = sampleImage;

    // Draw the inputted image on the canvas
    const canvas = canvasRef.current;
    const ctx = canvas!.getContext('2d');

    // get image data which is in uint8array format
    const imageData = ctx.getImageData(0, 0, 224, 224)

    const rgbData = imageData.data
    let normalizedData = new Float32Array(rgbData.length / 4)
    for (let i = 0; i < normalizedData.length; i++) {
      normalizedData[i] = Math.round(0.299 * rgbData[i * 4] + 0.587 * rgbData[i * 4 + 1] + 0.114 * rgbData[i * 4 + 2]) / 255.0;
    }

    const InputTensor = new Tensor('float32', normalizedData, [1, 1, 224, 224]); //Y = 0.299 * R + 0.587 * G + 0.114 * B


    imgElement.onload = async () => {
      ctx!.drawImage(imgElement, 0, 0, props.width, props.height);
      await submitInference(InputTensor);
    }
  };

  const submitInference = async (imgElement) => {
    // Run the inference
    await runSuperResModel(imgElement).then(async (res) => {
      const OutputTensor = res[0];
      const inferenceTime = res[1];
      const outputData = OutputTensor.toImageData();
      console.log(outputData)
      // let outputImageData = new ImageData(new Uint8ClampedArray(OutputTensor.bitmap.data.buffer), props.width, props.height);

      // Update the inference time
      setInferenceTime(`Inference speed: ${inferenceTime} seconds`);

      //put image data into canvas
      const supercanvas = superResCanvasRef.current;
      const superctx = supercanvas!.getContext('2d');
      superctx.putImageData(outputData, 0, 0);
    })
  };


  return (
    <>
      <button
        className={styles.grid}
        onClick={displayImageAndRunInference} >
        Run Super Resolution
      </button>
      <br />
      <canvas style={{ marginBottom: '20px' }} ref={canvasRef} width={props.width} height={props.height} />
      <canvas ref={superResCanvasRef} width={props.width} height={props.height} />
      <span>{inferenceTime}</span>
    </>
  )
};

export default ImageCanvas;
