import { useRef, useState } from 'react';
import { IMAGE_URLS } from '../data/sample-image-urls';
import { getImageTensorFromPath } from '../utils/predict';
import styles from '../styles/Home.module.css';
import { runSuperResModel } from '../utils/modelHelper';
import { tensorToImageData } from '../utils/imageHelper';
import React from 'react';
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
    // import images into array and randomly select one to display
    // var imagesArray = Object.values(allImages);
    // var randomImage = Math.floor(Math.random() * imagesArray.length);
    // return imagesArray[randomImage];
    const imageIndex = Math.floor(Math.random() * 10) + 1; // Random number from 1 to 10
    return `/t${imageIndex}.png`;
  }

  // Draw image and other UI elements then run inference
  const displayImageAndRunInference = async () => {
    // Get the image
    var sampleImage = getImage();
    var imgElement = new Image();
    imgElement.src = sampleImage;

    // Clear out previous values.
    //setInferenceTime("Inferencing...");
    // setInferenceTime("");

    // Draw the inputted image on the canvas
    const canvas = superResCanvasRef.current;
    const ctx = canvas!.getContext('2d');
    imgElement.onload = async () => {
      ctx!.drawImage(imgElement, 0, 0, props.width, props.height);
      await submitInference(imgElement);
    }
  };

  const submitInference = async (imgElement: HTMLImageElement) => {
    // Convert image data to tensor
    const preprocessedData = await getImageTensorFromPath(imgElement.src);

    // Run the inference
    const [superResTensor, inferenceTime] = await runSuperResModel(preprocessedData);

    // Update the inference time
    setInferenceTime(`Inference speed: ${inferenceTime} seconds`);

    // Convert the output tensor back into an image
    const outputImage = await tensorToImageData(superResTensor, props.width, props.height);
    const outputImageData = new ImageData(new Uint8ClampedArray(outputImage.bitmap.data.buffer), props.width, props.height);
    
    // Get the canvas context for the super resolution image
    const canvas = canvasRef.current;
    const ctx = canvas!.getContext('2d');
    ctx!.putImageData(outputImageData, props.width, props.height)
    // ctx!.putImageData(outputImageData, 0, 0)
    // Draw the super resolution image on the canvas
    //   imgElement.onload = async () => {
    //   ctx!.drawImage(imgElement, 0, 0, props.width, props.height);
    //   await submitInference(imgElement.src);
    // }
  };


  return (
    <>
      <button
        className={styles.grid}
        onClick={displayImageAndRunInference} >
        Run Super Resolution
      </button>
      <br />
      <canvas ref={canvasRef} width={props.width} height={props.height} />
      <canvas ref={superResCanvasRef} width={props.width} height={props.height} />
      <span>{inferenceTime}</span>
    </>
  )
};

export default ImageCanvas;
