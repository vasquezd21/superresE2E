import { useRef, useState } from 'react';
import { IMAGE_URLS } from '../data/sample-image-urls';
import { getImageTensorFromPath } from '../utils/predict';
import styles from '../styles/Home.module.css';
import { runSuperResModel } from '../utils/modelHelper';
import { tensorToImageData } from '../utils/imageHelper';
import React from 'react';

interface Props {
  height: number;
  width: number;
}

const ImageCanvas = (props: Props) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // new canvas ref for super res image
  const superResCanvasRef = useRef<HTMLCanvasElement>(null); 
  const [inferenceTime, setInferenceTime] = useState("");

  // Randomly loads the image from the IMAGE_URLS array
  const getImage = () => {
    var sampleImageUrls: Array<{ text: string; value: string }> = IMAGE_URLS;
    var random = Math.floor(Math.random() * (9 - 0 + 1) + 0);
    return sampleImageUrls[random];
  }

  // Draw image and other UI elements then run inference
  const displayImageAndRunInference = async () => { 
    // Get the image
    var sampleImage = getImage();
    var imgElement = new Image();
    imgElement.src = sampleImage.value;

    // Clear out previous values.
    //setInferenceTime("Inferencing...");

    // Draw the inputted image on the canvas
    const canvas = canvasRef.current;
    const ctx = canvas!.getContext('2d');
    imgElement.onload = async () => {
      ctx!.drawImage(imgElement, 0, 0, props.width, props.height);
      await submitInference(imgElement.src);
    }
  };

const submitInference = async (imgPath: string) => {
    // Convert image data to tensor
    const preprocessedData = await getImageTensorFromPath(imgPath);

    // Run the inference
    const [superResTensor, inferenceTime] = await runSuperResModel(preprocessedData);

    // Update the inference time
    setInferenceTime(`Inference speed: ${inferenceTime} seconds`);

    // Convert the output tensor back into an image
    const outputImage = await tensorToImageData(superResTensor, props.width, props.height);
    const outputImageData = new ImageData(new Uint8ClampedArray(outputImage.bitmap.data.buffer), props.width, props.height);

    // Get the canvas context for the super resolution image
    const canvas = superResCanvasRef.current;
    const ctx = canvas!.getContext('2d');

    // Draw the super resolution image on the canvas
    ctx!.putImageData(outputImageData, 0, 0);
};


  return (
    <>
      <button
        className={styles.grid}
        onClick={displayImageAndRunInference} >
        Run Super Resolution
      </button>
      <br/>
      <canvas ref={canvasRef} width={props.width} height={props.height} />
      <canvas ref={superResCanvasRef} width={props.width} height={props.height} />
      <span>{inferenceTime}</span>
    </>
  )
};

export default ImageCanvas;
