import Jimp from 'jimp';
import { Tensor } from 'onnxruntime-web';

export async function getImageTensorFromPath(path: string, dims: number[] =  [1, 3, 224, 224]): Promise<Tensor> {
  // 1. load the image  
  var image = await loadImageFromPath(path, dims[2], dims[3]);
  // 2. convert to tensor
  var imageTensor = imageDataToTensor(image, dims);
  // 3. return the tensor
  return imageTensor;
}

//Only need to convert the image to greyscale before converting it to tensors
async function loadImageFromPath(path: string, width: number = 224, height: number= 224): Promise<Jimp> {
  // Use Jimp to load the image, convert to grayscale and resize it.
  var imageData = await Jimp.read(path).then((imageBuffer: Jimp) => {
    return imageBuffer.greyscale().resize(width, height);
});


  return imageData;
}


function imageDataToTensor(image: Jimp, dims: number[]): Tensor {
  // 1. Get buffer data from image and create a grayscale array.
  var imageBufferData = image.bitmap.data;
  const grayArray = new Array<number>();

  // 2. Loop through the image buffer and extract the grayscale values
  for (let i = 0; i < imageBufferData.length; i += 4) {
    grayArray.push(imageBufferData[i]);
    // we only need one channel for grayscale image, so skip other channels and the alpha channel
  }

  // 3. convert to float32
  const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
  for (let i = 0; i < grayArray.length; i++) {
    float32Data[i] = grayArray[i] / 255.0; // convert to float
  }

  // 4. create the tensor object from onnxruntime-web.
  const inputTensor = new Tensor("float32", float32Data, dims);
  return inputTensor;
}


export async function tensorToImageData(tensor: Tensor, width: number = 224, height: number = 224): Promise<Jimp> {
  // Check if tensor.data is Float32Array
  if (!(tensor.data instanceof Float32Array)) {
    throw new Error('The tensor data must be a Float32Array');
  }

  // 1. Convert tensor values back to pixel intensities
  const normalized = Array.from(tensor.data, (value) => value * 255);

  // 2. array to hold the pixel data
  let pixelData = new Array<number>();
  for (let i = 0; i < normalized.length; i++) {
    // only have one channel so use the same value for RGB to create a grayscale image
    pixelData.push(normalized[i], normalized[i], normalized[i], 255);
  }

  // 3. Convert pixel data array to image
  let imageData = new Jimp({ data: Uint8Array.from(pixelData), width, height });

  return imageData;
}



