import * as ort from 'onnxruntime-web';
import _ from 'lodash';

export async function runSuperResModel(preprocessedData: any): Promise<[ort.Tensor, number]> {

  // Create session and set options. See the docs here for more options: 
  //https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#graphOptimizationLevel
  const session = await ort.InferenceSession
    .create('./_next/static/chunks/pages/superres_0_model.onnx',
      { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  console.log('Inference session created')

  // Run inference and get results.
  var [results, inferenceTime] = await runInference(session, preprocessedData);

  return [results, inferenceTime];
}

async function runInference(session: ort.InferenceSession, preprocessedData: ort.Tensor): Promise<[ort.Tensor, number]> {
  // Get start time to calculate inference time.
  const start = new Date();

  // create feeds with the input name from model export and the preprocessed data.
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = preprocessedData;

  // Run the session inference.
  const outputData = await session.run(feeds);

  // Get the end time to calculate inference time.
  const end = new Date();

  // Convert to seconds.
  const inferenceTime = (end.getTime() - start.getTime()) / 1000;

  // Get output results with the output name from the model export.
  const output = outputData[session.outputNames[0]];

  return [output, inferenceTime];
}

